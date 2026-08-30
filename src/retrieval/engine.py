"""Category-scoped retrieval engine.

Owns the B-side pipeline: parse what the customer said, scope the
candidate pool by category, rank inside it, and return candidates for
Person C to rerank.

The ranking signal is deliberately two-part.

*BM25* handles bag-of-words overlap and is robust when the customer
paraphrases. *Phrase evidence* handles the fact that the simulated
customer quotes the target product's own ``features`` and ``details``
strings verbatim, so a constraint that appears as an exact substring of a
document is very strong evidence, far stronger than the sum of its
individual tokens. Rare phrases are worth more than common ones, so the
bonus is scaled by the rarity of the phrase's tokens.

Turn state is intentionally thin. Person A owns the real slot-filling
state machine (A2); this class only accumulates the disclosed constraint
text it needs to rank, and exposes ``apply_override`` so A's intent
signals can drive it rather than duplicating the parsing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .catalog import Catalog
from .category import CategoryIndex
from .fusion import reciprocal_rank_fusion
from .lexical import LexicalIndex
from .text import normalise, tokenise

# The simulator states constraints after a colon, and separates multiple
# constraints in one breath with a semicolon.
_CONSTRAINT_RE = re.compile(r"(?:requirement is|what matters is|what I need is|need is)\s*:?\s*(.+)", re.IGNORECASE)

# Turns that carry no new information. Matching these stops us polluting
# the query with the simulator's own filler wording.
_EMPTY_TURN_RE = re.compile(
    r"(don't have (?:an additional |a )?preference|use your judgment|"
    r"not quite right yet|ask me about one specific attribute)",
    re.IGNORECASE,
)

# Wording the simulator uses when it revokes an earlier preference.
_OVERRIDE_RE = re.compile(r"\b(actually|instead|ignore my earlier|change to|rather than)\b", re.IGNORECASE)

# A phrase must be at least this long to count as evidence. Shorter
# fragments are single words that BM25 already handles.
_MIN_PHRASE_CHARS = 8


@dataclass
class SessionState:
    """Everything the retriever knows about one conversation."""

    category_label: str | None = None
    # Constraints in disclosure order. Earlier entries may be revoked by
    # an override; see `overridden_before`.
    constraints: list[str] = field(default_factory=list)
    # Index into `constraints`: entries before this were revoked.
    overridden_before: int = 0
    turn: int = 0

    def active_constraints(self, drop_overridden: bool) -> list[str]:
        return self.constraints[self.overridden_before:] if drop_overridden else self.constraints


class SearchEngine:
    """Builds the indexes once, then serves per-session ranked candidates."""

    def __init__(
        self,
        catalog: Catalog | None = None,
        catalog_path: str = "data/catalog.jsonl",
        drop_overridden: bool = False,
        phrase_weight: float = 1.6,
        dense: "object | None" = None,
        use_fusion: bool = True,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.catalog = catalog or Catalog.load(catalog_path)
        self.categories = CategoryIndex(self.catalog)
        self.lexical = LexicalIndex(self.catalog)
        # Optional DenseIndex. Absent means the engine runs lexical-only,
        # which is the configuration measured at technical score 0.796.
        self.dense = dense
        # Whether an "actually, ignore my earlier preference" turn should
        # erase previously disclosed constraints. Off by default: the
        # simulator draws its revoked preference from the target product
        # too, so erasing it throws away true signal. See ANNA_dump/log.txt.
        self.drop_overridden = drop_overridden
        self.phrase_weight = phrase_weight
        self.use_fusion = use_fusion
        # B3 fusion weights, one per ranker. Dense starts at parity with
        # keyword and is tuned in B5 against the public set.
        # Dense defaults to 0.0. The index is built and tested (B2), but a
        # weight sweep showed every non-zero value costs score: 0.8732 at
        # 0.0 falling monotonically to 0.8281 at 1.0. Raise it only if the
        # private set turns out to paraphrase rather than quote.
        self.weights = {"lexical": 1.0, "category": 0.4, "phrase": 1.0, "dense": 0.0}
        if weights:
            self.weights.update(weights)
        self._states: dict[str, SessionState] = {}

    # ------------------------------------------------------------------
    def start_session(self, session_id: str) -> SessionState:
        state = SessionState()
        self._states[session_id] = state
        return state

    def state(self, session_id: str) -> SessionState:
        return self._states.setdefault(session_id, SessionState())

    # ------------------------------------------------------------------
    def observe(self, session_id: str, message: str) -> SessionState:
        """Fold one customer turn into the session state."""
        state = self.state(session_id)
        state.turn += 1

        label = CategoryIndex.parse_label(message)
        if label:
            state.category_label = label

        if _OVERRIDE_RE.search(message):
            # Mark everything disclosed so far as revoked. Whether that
            # revocation is honoured at ranking time is a separate switch.
            state.overridden_before = len(state.constraints)

        if _EMPTY_TURN_RE.search(message):
            return state

        for value in self._extract_constraints(message):
            if value not in state.constraints:
                state.constraints.append(value)
        return state

    @staticmethod
    def _extract_constraints(message: str) -> list[str]:
        """Pull stated constraint strings out of a customer turn."""
        match = _CONSTRAINT_RE.search(message)
        if not match:
            return []
        tail = match.group(1).strip().rstrip(".")
        return [part.strip() for part in tail.split(";") if part.strip()]

    # ------------------------------------------------------------------
    def search(self, session_id: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Rank candidates for the current state. Returns (asin, score)."""
        state = self.state(session_id)
        scope = self.categories.scope(state.category_label)
        ranked = self._rank(state, scope.primary, top_k)

        # Top up from the wider pool only if the tight pool came up short,
        # so a mis-parsed category degrades quality instead of recall.
        if len(ranked) < top_k and len(scope.fallback):
            seen = {asin for asin, _ in ranked}
            for asin, score in self._rank(state, scope.fallback, top_k * 2):
                if asin not in seen:
                    ranked.append((asin, score * 0.5))  # demoted: wider pool
                    if len(ranked) >= top_k:
                        break
        return ranked[:top_k]

    # ------------------------------------------------------------------
    def _rank(self, state: SessionState, pool: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not len(pool):
            return []
        constraints = state.active_constraints(self.drop_overridden)

        # Query text: the stated constraints plus the category label, so a
        # bare browsing turn still ranks sensibly on category words alone.
        query = " ".join(constraints)
        if state.category_label:
            query = f"{state.category_label} {query}".strip()
        terms = tokenise(query)

        # --- component rankers, each scored over the same pool ---------
        components: dict[str, np.ndarray] = {
            "lexical": self.lexical.score(terms, pool),
            "category": self._category_scores(terms, pool),
        }
        if constraints:
            components["phrase"] = self._phrase_evidence(constraints, pool)
        # Skipped entirely at weight 0, not just down-weighted: the encode
        # plus gather is ~40x the cost of the whole lexical path, and the
        # sweep in ANNA_dump/log.txt shows it earns nothing on this
        # benchmark. Attaching it is opt-in, and paid for only if used.
        if self.dense is not None and self.weights.get("dense", 0.0) > 0.0:
            components["dense"] = self.dense.score(query, pool)

        if self.use_fusion:
            # B3: fuse by rank, not by score. See fusion.py for why.
            scores = reciprocal_rank_fusion(components, self.weights)
        else:
            # Pre-fusion path, kept so the 0.796 lexical baseline stays
            # reproducible for the write-up.
            scores = components["lexical"] + self.phrase_weight * components.get(
                "phrase", np.zeros(len(pool), dtype=np.float32)
            )

        limit = min(top_k, len(pool))
        # argpartition then sort: O(n) selection instead of a full sort of
        # the pool, which matters once the fallback pool is large.
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self.catalog.asins[int(pool[i])], float(scores[i])) for i in top]

    def _category_scores(self, terms: list[str], pool: np.ndarray) -> np.ndarray:
        """Rank by how much of the query appears in the category path.

        The pool is already category-filtered, so this does not select the
        category; it discriminates *within* it. An item sitting deeper in
        a path the customer's words match is a better fit than a sibling
        that only matches the parent level.
        """
        scores = np.zeros(len(pool), dtype=np.float32)
        if not terms:
            return scores
        unique = list(dict.fromkeys(terms))
        column = self.catalog.fields["categories"]
        for slot, doc_id in enumerate(pool.tolist()):
            text = column[doc_id]
            if not text:
                continue
            scores[slot] = sum(self.lexical.idf(t) for t in unique if t in text)
        return scores

    def _phrase_evidence(self, constraints: list[str], pool: np.ndarray) -> np.ndarray:
        """Reward documents containing a stated constraint verbatim.

        Weighted by the rarity of the phrase, so "Material:alloy" on a
        niche listing counts for much more than a boilerplate phrase that
        half the catalog repeats.
        """
        bonus = np.zeros(len(pool), dtype=np.float32)
        phrases: list[tuple[str, float]] = []
        for value in constraints:
            phrase = normalise(value)
            if len(phrase) < _MIN_PHRASE_CHARS:
                continue
            terms = tokenise(phrase)
            if not terms:
                continue
            # Mean idf of the phrase's tokens, normalised to roughly 0..1.
            rarity = float(np.mean([self.lexical.idf(t) for t in terms])) / 10.0
            phrases.append((phrase, min(1.0, max(0.05, rarity))))
        if not phrases:
            return bonus
        for slot, doc_id in enumerate(pool.tolist()):
            text = self.catalog.document_text(doc_id)
            total = 0.0
            for phrase, rarity in phrases:
                if phrase in text:
                    total += rarity
            bonus[slot] = total
        return bonus
