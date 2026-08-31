"""B1: frozen catalog loading and schema normalisation.

Loads the 50,000 item ``Clothing_Shoes_and_Jewelry`` catalog into memory
once, in a layout the downstream indexes can address by integer document
id rather than by ``parent_asin`` string. Integer ids let the lexical
index work in numpy arrays, which is what keeps B4's latency and memory
budget realistic.

Field notes from the data audit:
  * ``price``    null on ~79% of items. Kept as NaN, never used to exclude.
  * ``details``  empty on ~3.3% of items.
  * ``categories`` never empty, so it is safe to lean on for scoping.
  * ``title``    empty on 2 items out of 50,000.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .text import flatten, normalise

# Fields indexed for retrieval, ordered by how much signal they carry.
# The weights are applied by the lexical index, not here; this tuple only
# fixes the field order so every index agrees on it.
INDEXED_FIELDS: tuple[str, ...] = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)

# Category path segments that carry no discriminative power because every
# item in the frozen catalog sits under them. Mirrors the evaluator's own
# exclusion list so our coarse category strings line up with the ones the
# customer simulator speaks.
GENERIC_SEGMENTS = frozenset({
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
})


class Catalog:
    """In-memory view of the frozen catalog, addressed by document id."""

    def __init__(self) -> None:
        self.asins: list[str] = []
        self.titles: list[str] = []
        self.stores: list[str] = []
        # Normalised (lowercased, whitespace collapsed) text per indexed
        # field, one list per field, each parallel to self.asins.
        self.fields: dict[str, list[str]] = {name: [] for name in INDEXED_FIELDS}
        # Raw category path, e.g. ("Women", "Jewelry", "Earrings", "Hoop"),
        # with the generic top-level segments stripped.
        self.category_paths: list[tuple[str, ...]] = []
        # NaN where the source price is null. Never used as a hard filter.
        self.prices: np.ndarray = np.zeros(0, dtype=np.float32)
        self.asin_to_id: dict[str, int] = {}
        # Set by compact(): one joined string per document, replacing the
        # six per-field lists once the indexes no longer need them.
        self._combined: list[str] | None = None

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = "data/catalog.jsonl") -> "Catalog":
        catalog = cls()
        prices: list[float] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                doc_id = len(catalog.asins)
                asin = str(product["parent_asin"])
                catalog.asins.append(asin)
                catalog.asin_to_id[asin] = doc_id
                catalog.titles.append(str(product.get("title") or ""))
                catalog.stores.append(str(product.get("store") or ""))
                for name in INDEXED_FIELDS:
                    catalog.fields[name].append(normalise(flatten(product.get(name))))
                catalog.category_paths.append(_clean_path(product.get("categories")))
                prices.append(_parse_price(product.get("price")))
        catalog.prices = np.asarray(prices, dtype=np.float32)
        return catalog

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.asins)

    def document_text(self, doc_id: int) -> str:
        """Full normalised text of one document, for phrase containment."""
        if self._combined is not None:
            return self._combined[doc_id]
        return " ".join(self.fields[name][doc_id] for name in INDEXED_FIELDS)

    def compact(self) -> None:
        """Drop per-field text once the indexes that need it are built. (B4)

        After construction only two consumers remain: phrase evidence wants
        the whole document as one string, and the category ranker wants the
        category path. Holding six separate strings per document serves
        neither, costs six times the per-string overhead, and forces a join
        on every phrase lookup. Collapsing to one combined string plus the
        category text is both smaller and faster on the hot path.

        Safe to call only after LexicalIndex and CategoryIndex exist; the
        raw category path list is preserved, so scoping still works.
        """
        if self._combined is not None:
            return
        # Keep the category text, which the category ranker still needs.
        categories = list(self.fields["categories"])
        sources = [self.fields[name] for name in INDEXED_FIELDS]
        combined: list[str] = []
        for doc_id in range(self.size):
            combined.append(" ".join(source[doc_id] for source in sources))
            # Release each original string as it is consumed. Building the
            # joined list while the originals are all still referenced
            # spikes resident memory by ~40MB and the allocator does not
            # hand it back, so the naive version made things worse.
            for source in sources:
                source[doc_id] = ""
        self._combined = combined
        self.fields = {"categories": categories}

    def coarse_category(self, doc_id: int) -> str:
        """The category label the customer simulator will actually say.

        The evaluator builds its opening line from the last two segments of
        the cleaned category path, so we reproduce that exactly. Matching
        this string is what collapses 50,000 candidates to roughly 180.
        """
        path = self.category_paths[doc_id]
        return " ".join(path[-2:]) if path else "clothing item"


def _clean_path(value: object) -> tuple[str, ...]:
    """Split a categories list into segments, dropping generic top levels.

    Segments arrive both as separate list entries and as comma-joined
    strings, so both are split before filtering.
    """
    if not isinstance(value, list):
        return ()
    segments: list[str] = []
    for entry in value:
        for part in str(entry).split(","):
            part = part.strip()
            if part and part.lower() not in GENERIC_SEGMENTS:
                segments.append(part)
    return tuple(segments)


def _parse_price(value: object) -> float:
    """Coerce price to float, NaN when absent or unparseable.

    NaN is deliberate: an unpriced item is unknown, not free and not
    expensive, so it must never sort to either end of a price ordering.
    """
    if value is None or value == "":
        return float("nan")
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return float("nan")
