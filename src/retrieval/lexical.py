"""B2 (keyword half): in-memory field-weighted BM25.

Hand-rolled rather than pulled from a library for three reasons. It runs
entirely in memory with numpy as the only dependency, which is what B4
asks for; it supports scoring a *restricted* candidate pool, which the
category scoping in :mod:`.category` depends on; and it lets each catalog
field carry its own weight, since a term appearing in ``title`` means far
more than the same term buried in ``description``.

Index layout is a classic inverted index held in flat numpy arrays:

    postings[term] -> (doc_ids, weighted_term_frequencies)

Flat arrays keep the memory footprint predictable and make scoring a
vectorised gather rather than a Python loop over documents.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .catalog import Catalog, INDEXED_FIELDS
from .text import tokenise

# Per-field multipliers applied to term frequency at index time.
# Rationale: the customer simulator quotes `features` and `details`
# verbatim when it states a requirement, and names the category in its
# opening line, so those three fields carry the most retrieval signal.
# `description` is long and repetitive, so it is damped.
FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "categories": 2.0,
    "features": 2.5,
    "details": 2.5,
    "store": 1.5,
    "description": 1.0,
}

BM25_K1 = 1.2
BM25_B = 0.6  # Below the usual 0.75: catalog documents vary hugely in
              # length and over-penalising long ones buries rich listings.


class LexicalIndex:
    """Field-weighted BM25 over the frozen catalog, scored in numpy."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.size = catalog.size
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        self._doc_len = np.zeros(self.size, dtype=np.float32)
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        # term -> {doc_id: weighted tf}. Built as dicts then frozen into
        # numpy arrays, which roughly halves the resident size.
        raw: dict[str, dict[int, float]] = defaultdict(dict)
        for field in INDEXED_FIELDS:
            weight = FIELD_WEIGHTS[field]
            column = self.catalog.fields[field]
            for doc_id in range(self.size):
                text = column[doc_id]
                if not text:
                    continue
                for token in tokenise(text):
                    bucket = raw[token]
                    bucket[doc_id] = bucket.get(doc_id, 0.0) + weight
                    self._doc_len[doc_id] += weight

        self._avg_len = float(self._doc_len.mean()) or 1.0
        total_docs = float(self.size)
        for token, bucket in raw.items():
            doc_ids = np.fromiter(bucket.keys(), dtype=np.int32, count=len(bucket))
            freqs = np.fromiter(bucket.values(), dtype=np.float32, count=len(bucket))
            order = np.argsort(doc_ids)
            self._postings[token] = (doc_ids[order], freqs[order])
            # Standard BM25 idf with the +1 guard so common terms stay
            # weakly positive instead of flipping negative.
            df = float(len(bucket))
            self._idf[token] = float(np.log(1.0 + (total_docs - df + 0.5) / (df + 0.5)))

    # ------------------------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    def score(self, query_terms: list[str], pool: np.ndarray | None = None) -> np.ndarray:
        """BM25 score for every document in ``pool`` (or the whole catalog).

        Returns an array parallel to ``pool``. Scoring inside a pool is the
        hot path: the category scope hands us ~194 candidates, so the
        gather below touches a fraction of the postings list.
        """
        if pool is None:
            pool = np.arange(self.size, dtype=np.int32)
        scores = np.zeros(len(pool), dtype=np.float32)
        if not len(pool) or not query_terms:
            return scores

        # Position of each pooled doc, so postings can be mapped onto the
        # output array without scanning. -1 marks documents outside the pool.
        position = np.full(self.size, -1, dtype=np.int32)
        position[pool] = np.arange(len(pool), dtype=np.int32)

        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * self._doc_len[pool] / self._avg_len)

        # Repeated query terms genuinely count more, so tf is not deduped.
        for term in query_terms:
            entry = self._postings.get(term)
            if entry is None:
                continue
            doc_ids, freqs = entry
            slots = position[doc_ids]
            keep = slots >= 0
            if not keep.any():
                continue
            slots = slots[keep]
            tf = freqs[keep]
            scores[slots] += self._idf[term] * (tf * (BM25_K1 + 1.0)) / (tf + norm[slots])
        return scores

    def idf(self, term: str) -> float:
        """Rarity of a term, used by the engine to weight phrase evidence."""
        return self._idf.get(term, 0.0)
