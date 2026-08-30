"""Category scoping: the primary candidate-pool reducer.

The customer simulator opens every session with the target product's own
coarse category, e.g. "I'm looking for Earrings Hoop". Recovering that
label and scoping to it takes the candidate pool from 50,000 items to a
median of roughly 180, before any relevance scoring happens at all.

Scoping is never allowed to be fatal. If the stated label is unknown, or
the bucket it selects is too small to contain a plausible answer, the
ladder in :meth:`CategoryIndex.candidates` widens step by step and
ultimately falls back to the whole catalog. A wrong category costs us
ranking quality; it must never cost us the item entirely.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .catalog import Catalog
from .text import normalise, tokenise

# The simulator's three opening templates all place the category directly
# after "looking for", terminated by a comma or full stop.
_CATEGORY_RE = re.compile(r"looking for\s+(.+?)\s*(?:[.,]|$)", re.IGNORECASE)


class CategoryIndex:
    """Maps spoken category labels to candidate document ids."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        # Exact coarse label (as the simulator would say it) -> doc ids.
        self._coarse: dict[str, list[int]] = defaultdict(list)
        # Individual path segment -> doc ids, used to widen a thin bucket.
        self._segment: dict[str, list[int]] = defaultdict(list)
        for doc_id in range(catalog.size):
            self._coarse[normalise(catalog.coarse_category(doc_id))].append(doc_id)
            for segment in catalog.category_paths[doc_id]:
                self._segment[normalise(segment)].append(doc_id)
        self._coarse_arrays = {k: np.asarray(v, dtype=np.int32) for k, v in self._coarse.items()}
        self._segment_arrays = {k: np.asarray(v, dtype=np.int32) for k, v in self._segment.items()}
        self._all = np.arange(catalog.size, dtype=np.int32)

    # ------------------------------------------------------------------
    @staticmethod
    def parse_label(message: str) -> str | None:
        """Pull the stated category out of an opening customer message."""
        match = _CATEGORY_RE.search(message)
        if not match:
            return None
        label = normalise(match.group(1))
        # Guard against the phrase running away into a long sentence when
        # the simulator's category itself contained no terminator.
        return label if label and len(label) < 120 else None

    # ------------------------------------------------------------------
    def scope(self, label: str | None, min_pool: int = 10) -> "Scope":
        """Resolve a stated label into a tight primary pool and a wider tail.

        Measured on the 200 public sessions, the exact bucket contains the
        target every time, with a median size of 182. So the primary pool
        stays deliberately tight and widening is demoted to a fallback the
        engine only draws on when the primary cannot fill top_k. That keeps
        ranking precision high without ever risking zero recall on a
        private session whose category wording we fail to recognise.

        ``min_pool`` defaults to top_k: a bucket only has to be able to
        fill the result list to be worth ranking inside. Raising it to 30
        pushed the worst-case pool from 1,478 to 29,521 items for no
        recall gain, so the tight setting wins on both axes.
        """
        if not label:
            return Scope(self._all, self._all, "all")

        exact = self._coarse_arrays.get(label)
        tokens = tokenise(label, keep_stopwords=True)
        token_hits = [h for h in (self._segment_token_hits(t) for t in tokens) if h is not None]

        # Documents whose category path carries every token of the label.
        # Catches wording or ordering drift between label and stored path.
        intersected: np.ndarray | None = None
        if token_hits:
            intersected = token_hits[0]
            for hits in token_hits[1:]:
                intersected = np.intersect1d(intersected, hits)

        # Widest sensible tail: anything sharing any segment of the label,
        # deepest segment first since it is the most specific.
        wide = np.zeros(0, dtype=np.int32)
        for hits in reversed(token_hits):
            wide = np.union1d(wide, hits).astype(np.int32)
        fallback = wide if len(wide) else self._all

        # Primary pool: tightest option that is big enough to rank inside.
        if exact is not None and len(exact) >= min_pool:
            return Scope(exact, fallback, "exact")
        merged = exact if exact is not None else np.zeros(0, dtype=np.int32)
        if intersected is not None and len(intersected):
            merged = np.union1d(merged, intersected).astype(np.int32)
        if len(merged) >= min_pool:
            return Scope(merged, fallback, "exact+token" if exact is not None else "token")
        if len(fallback) and fallback is not self._all:
            return Scope(fallback, self._all, "widened")
        return Scope(self._all, self._all, "all")

    # ------------------------------------------------------------------
    def _segment_token_hits(self, token: str) -> np.ndarray | None:
        """Doc ids for every category segment containing this token."""
        direct = self._segment_arrays.get(token)
        if direct is not None:
            return direct
        parts = [ids for segment, ids in self._segment_arrays.items() if token in segment.split()]
        if not parts:
            return None
        return np.unique(np.concatenate(parts)).astype(np.int32)


@dataclass(frozen=True)
class Scope:
    """A resolved candidate pool.

    ``primary`` is ranked first. ``fallback`` is only drawn on to top up
    the result list when the primary pool cannot supply enough hits, so a
    mis-parsed category degrades quality rather than returning nothing.
    """

    primary: np.ndarray
    fallback: np.ndarray
    mode: str
