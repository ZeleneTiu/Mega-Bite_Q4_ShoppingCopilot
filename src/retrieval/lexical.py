"""B2 (keyword half) + B4 (memory): in-memory field-weighted BM25.

Hand-rolled rather than pulled from a library for three reasons. It runs
entirely in memory with numpy as the only dependency, which is what B4
asks for; it supports scoring a *restricted* candidate pool, which the
category scoping in :mod:`.category` depends on; and it lets each catalog
field carry its own weight, since a term appearing in ``title`` means far
more than the same term buried in ``description``.

Storage is a CSR-style inverted index in three flat arrays:

    offsets[t] .. offsets[t+1]   slice of postings belonging to term t
    doc_ids[...]                 which documents
    freqs[...]                   weighted term frequency in each

The obvious implementation, ``dict[term][doc_id] = weight``, was measured
at 561MB resident for 4.3M postings whose actual data is 34MB. Python
dict entries cost roughly 100 bytes each, and the allocator never returns
the arenas afterwards. Flat arrays hold the same information with no
per-entry object overhead, and the build accumulates into arrays rather
than dicts so the peak is bounded too.
"""
from __future__ import annotations

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

# Documents per flush when building. Bounds the transient arrays without
# making the concatenation list long enough to matter.
_BUILD_CHUNK = 4096


class LexicalIndex:
    """Field-weighted BM25 over the frozen catalog, scored in numpy."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.size = catalog.size
        self._vocab: dict[str, int] = {}
        self._offsets = np.zeros(1, dtype=np.int64)
        self._doc_ids = np.zeros(0, dtype=np.int32)
        self._freqs = np.zeros(0, dtype=np.float32)
        self._idf_values = np.zeros(0, dtype=np.float32)
        self._doc_len = np.zeros(self.size, dtype=np.float32)
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        vocab = self._vocab
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        term_buf: list[int] = []
        doc_buf: list[int] = []
        weight_buf: list[float] = []

        def flush() -> None:
            if not term_buf:
                return
            chunks.append((
                np.asarray(term_buf, dtype=np.int32),
                np.asarray(doc_buf, dtype=np.int32),
                np.asarray(weight_buf, dtype=np.float32),
            ))
            term_buf.clear(); doc_buf.clear(); weight_buf.clear()

        for doc_id in range(self.size):
            for field in INDEXED_FIELDS:
                text = self.catalog.fields[field][doc_id]
                if not text:
                    continue
                weight = FIELD_WEIGHTS[field]
                for token in tokenise(text):
                    term_id = vocab.get(token)
                    if term_id is None:
                        term_id = len(vocab)
                        vocab[token] = term_id
                    term_buf.append(term_id)
                    doc_buf.append(doc_id)
                    weight_buf.append(weight)
                    self._doc_len[doc_id] += weight
            if doc_id % _BUILD_CHUNK == _BUILD_CHUNK - 1:
                flush()
        flush()

        if not chunks:
            self._offsets = np.zeros(1, dtype=np.int64)
            self._avg_len = 1.0
            return

        terms = np.concatenate([c[0] for c in chunks])
        docs = np.concatenate([c[1] for c in chunks])
        weights = np.concatenate([c[2] for c in chunks])
        chunks.clear()

        # Group identical (term, doc) pairs and sum their weights, which is
        # what the dict version was doing implicitly. Sorting by term then
        # doc also lays the postings out in the CSR order we want.
        order = np.lexsort((docs, terms))
        terms, docs, weights = terms[order], docs[order], weights[order]
        del order
        boundary = np.empty(len(terms), dtype=bool)
        boundary[0] = True
        np.not_equal(terms[1:], terms[:-1], out=boundary[1:])
        np.logical_or(boundary[1:], docs[1:] != docs[:-1], out=boundary[1:])
        starts = np.flatnonzero(boundary)
        self._doc_ids = docs[starts]
        self._freqs = np.add.reduceat(weights, starts).astype(np.float32)
        grouped_terms = terms[starts]
        del terms, docs, weights, boundary, starts

        # offsets[t] is where term t's postings begin. searchsorted over the
        # sorted term column gives every boundary in one pass.
        vocab_size = len(vocab)
        self._offsets = np.searchsorted(
            grouped_terms, np.arange(vocab_size + 1, dtype=np.int32), side="left"
        ).astype(np.int64)
        del grouped_terms

        document_frequency = np.diff(self._offsets).astype(np.float32)
        total_docs = float(self.size)
        # Standard BM25 idf with the +1 guard so common terms stay weakly
        # positive instead of flipping negative.
        self._idf_values = np.log(
            1.0 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        self._avg_len = float(self._doc_len.mean()) or 1.0

    # ------------------------------------------------------------------
    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    @property
    def postings_count(self) -> int:
        return int(len(self._doc_ids))

    def score(
        self,
        query_terms: list[str],
        pool: np.ndarray | None = None,
        term_weights: list[float] | None = None,
    ) -> np.ndarray:
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
        # term_weights, when given, carries B4's slot decay: an older or
        # revoked constraint contributes a fraction of a fresh one.
        for position_index, term in enumerate(query_terms):
            weight = 1.0 if term_weights is None else term_weights[position_index]
            if weight <= 0.0:
                continue
            term_id = self._vocab.get(term)
            if term_id is None:
                continue
            start, end = self._offsets[term_id], self._offsets[term_id + 1]
            if start == end:
                continue
            slots = position[self._doc_ids[start:end]]
            keep = slots >= 0
            if not keep.any():
                continue
            slots = slots[keep]
            tf = self._freqs[start:end][keep]
            idf = float(self._idf_values[term_id])
            scores[slots] += weight * idf * (tf * (BM25_K1 + 1.0)) / (tf + norm[slots])
        return scores

    def idf(self, term: str) -> float:
        """Rarity of a term, used by the engine to weight phrase evidence."""
        term_id = self._vocab.get(term)
        return 0.0 if term_id is None else float(self._idf_values[term_id])
