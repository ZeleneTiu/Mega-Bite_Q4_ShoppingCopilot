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

# Leading clauses that announce a constraint without being part of it.
# Stripped so phrase matching compares product text against product text,
# not against the customer's framing. Deliberately covers many phrasings:
# the public simulator uses one, the private set may use others.
_LEAD_MARKER_RE = re.compile(
    r"^.{0,80}?\b(?:"
    r"requirement is|requirements are|key requirement|what matters is|"
    r"what i need is|things that matter to me are|matters? to me (?:is|are)|"
    r"needs? to be|must (?:have|be)|has to be|have to be|"
    r"i(?:'m| am)?\s*(?:want|need|after|looking for|really want)|"
    r"prefer(?:ably)?|ideally|specifically"
    r")\b\s*:?\s*",
    re.IGNORECASE,
)

# Turns that carry no new information. Matching these stops us polluting
# the query with the simulator's filler. Broadened well beyond the public
# templates, since a false negative here costs far less than a false
# positive: at worst we add a little noise to the query.
_EMPTY_TURN_RE = re.compile(
    r"(don'?t have (?:an additional |a )?preference|no (?:strong )?(?:feelings|preference)|"
    r"does ?n'?t matter|no opinion|use your judg?ment|your call|up to you|"
    r"not quite (?:right)?|nothing (?:else|more)(?: to add)?|"
    r"ask me (?:about )?(?:one |some)?(?:thing|specific)|^\s*nope\b)",
    re.IGNORECASE,
)

# Wording that revokes an earlier preference.
_OVERRIDE_RE = re.compile(
    r"\b(actually|instead|ignore my earlier|ignore what i said|change to|"
    r"rather than|scratch that|forget what i said|on second thought)\b",
    re.IGNORECASE,
)

# Primary separators: a semicolon, or a sentence boundary. These reliably
# divide one constraint from the next.
_SEGMENT_RE = re.compile(r";|(?<=[a-z0-9])\.\s+(?=[A-Z])")

# Secondary separator. Conversational phrasing joins constraints with "and",
# but so do plenty of real product phrases ("Drop and Dangle", "Shoes and
# Jewelry"): 19 of 80 sampled constraints contain it. So "and" only ever
# produces EXTRA candidates alongside the whole segment, never instead of
# it. Splitting on it outright cost 0.8732 -> 0.8543 on the public set.
_SUBSPLIT_RE = re.compile(r"\band\b", re.IGNORECASE)

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
    # Every candidate form of every constraint, including the customer's
    # framing. Used only for phrase evidence, where a candidate that never
    # occurs in any document simply contributes nothing.
    phrases: list[str] = field(default_factory=list)
    # Turn on which each entry of `constraints` was disclosed, parallel to
    # it. Drives B4's slot decay.
    constraint_turns: list[int] = field(default_factory=list)
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
        slot_decay: float = 1.0,
        override_penalty: float = 1.0,
        dense: "object | None" = None,
        use_fusion: bool = True,
        weights: dict[str, float] | None = None,
        compact_catalog: bool = True,
    ) -> None:
        self.catalog = catalog or Catalog.load(catalog_path)
        self.categories = CategoryIndex(self.catalog)
        self.lexical = LexicalIndex(self.catalog)
        # B4: both indexes are built, so the per-field text can go. Frees
        # roughly 60MB and removes a six-way join from the phrase path.
        if compact_catalog:
            self.catalog.compact()
        # Optional DenseIndex. Absent means the engine runs lexical-only,
        # which is the configuration measured at technical score 0.796.
        self.dense = dense
        # Whether an "actually, ignore my earlier preference" turn should
        # erase previously disclosed constraints. Off by default: the
        # simulator draws its revoked preference from the target product
        # too, so erasing it throws away true signal. See ANNA_dump/log.txt.
        self.drop_overridden = drop_overridden
        self.phrase_weight = phrase_weight
        # B4 slot decay. A constraint contributes slot_decay ** age, where
        # age is how many turns ago it was disclosed. 1.0 disables it.
        self.slot_decay = slot_decay
        # Multiplier for constraints the customer has since revoked. This
        # is the middle ground between keeping them at full strength and
        # deleting them: erasure measured at 0.7186 vs 0.8702 for keeping,
        # because the simulator draws the revoked preference from the
        # target product too, so it stays true. 1.0 disables the penalty.
        self.override_penalty = override_penalty
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
    def observe(self, session_id: str, message: str,
                constraints: dict | None = None) -> SessionState:
        """Fold one customer turn into the session state.

        ``constraints`` is Person A's parsed slot dict (IntentRouter's
        ``detected_constraints``). It is a HINT and additive only: its textual
        values join the phrase-evidence candidates, and phrase evidence scores
        a document only when the phrase actually occurs in it, so a wrong or
        redundant slot costs one lookup and nothing else.

        It deliberately does NOT touch the BM25 query or the evidence-word
        count. Both feed the hold gate, and A's regexes are tuned for intent
        classification rather than recall, so letting them move the gate would
        make the hold decision depend on a second, differently-tuned parser.
        """
        state = self.state(session_id)
        state.turn += 1
        is_override = bool(_OVERRIDE_RE.search(message))

        # Category is sticky. It is established once, from the opening
        # turn, and only revisited when the customer explicitly revokes
        # something. Without this, a later turn mentioning "leather" gets
        # mis-read as a request for the leather jackets category and the
        # scope silently moves off the target.
        if state.category_label is None:
            state.category_label = self.categories.find_label(message)
        elif is_override:
            # An override revokes a preference, not usually the category.
            # Only a category named verbatim may replace an established
            # one: allowing the fuzzy matcher to act here cost intent
            # override 0.967 -> 0.833, because the override turn names a
            # constraint and no category, so the matcher invented one.
            replacement = self.categories.find_label(message, verbatim_only=True)
            if replacement:
                state.category_label = replacement

        if is_override:
            # Mark everything disclosed so far as revoked. Whether that
            # revocation is honoured at ranking time is a separate switch.
            state.overridden_before = len(state.constraints)

        # A's slots ride along as extra phrase candidates. Done before the
        # empty-turn return so a turn B reads as contentless can still carry
        # a slot A recognised.
        if constraints:
            for value in self._router_phrases(constraints):
                if value not in state.phrases:
                    state.phrases.append(value)

        if _EMPTY_TURN_RE.search(message):
            return state

        preferred, candidates = self._parse_turn(message, state.category_label)
        for value in preferred:
            if value not in state.constraints:
                state.constraints.append(value)
                state.constraint_turns.append(state.turn)
        for value in candidates:
            if value not in state.phrases:
                state.phrases.append(value)
        return state

    @staticmethod
    def _router_phrases(constraints: dict) -> list[str]:
        """Textual slot values from A's router, as phrase-evidence candidates.

        Numeric slots (price_max, price_min, min_rating) are skipped on
        purpose: they are filters rather than text, and B4 measured price as
        carrying no usable signal on this catalog (null on a large share of
        items, and absent from 0 of 200 public sessions).
        """
        out: list[str] = []
        for value in constraints.values():
            if not isinstance(value, str):
                continue
            value = value.strip()
            if len(value) >= 3:
                out.append(value)
        return list(dict.fromkeys(out))

    @classmethod
    def _extract_constraints(cls, message: str, category_label: str | None = None) -> list[str]:
        """Preferred constraint forms only. Thin wrapper kept for tests."""
        return cls._parse_turn(message, category_label)[0]

    @staticmethod
    def _parse_turn(message: str, category_label: str | None = None) -> tuple[list[str], list[str]]:
        """Pull constraint text out of a turn without relying on templates.

        Rather than hunting for a marker phrase and giving up when it is
        absent, this takes the whole turn as constraint text and removes
        what is known not to be a constraint: the category the customer
        already stated, and any leading clause that announces rather than
        states. Both the stripped and the unstripped form are kept, because
        phrase evidence only ever adds score when a phrase actually occurs
        in a document, so a redundant candidate costs nothing but a lookup.
        """
        text = message.strip()
        if category_label:
            # Remove the category span, case-insensitively, so it is not
            # re-counted as a constraint phrase.
            text = re.sub(re.escape(category_label), " ", text, flags=re.IGNORECASE)

        preferred: list[str] = []   # feeds the word-matching query
        candidates: list[str] = []  # feeds phrase evidence
        for segment in _SEGMENT_RE.split(text):
            if segment is None:
                continue
            segment = segment.strip().strip(".,;:!? ").strip()
            if len(segment) < 3:
                continue
            candidates.append(segment)
            stripped = _LEAD_MARKER_RE.sub("", segment).strip(".,;:!? ").strip()
            if stripped and stripped != segment and len(stripped) >= 3:
                candidates.append(stripped)
            # Extra candidates either side of an "and", in case the turn
            # really was joining two separate constraints. Additive only.
            base = stripped if stripped else segment
            if _SUBSPLIT_RE.search(base):
                for piece in _SUBSPLIT_RE.split(base):
                    piece = piece.strip(".,;:!? ").strip()
                    if len(piece) >= 4:
                        candidates.append(piece)
            # The query gets the announcing clause removed. Keeping it in
            # dilutes BM25 with the customer's framing, which measured as
            # 0.8732 -> 0.8543 on the public templates.
            preferred.append(stripped if stripped and len(stripped) >= 3 else segment)
        return list(dict.fromkeys(preferred)), list(dict.fromkeys(candidates))

    # ------------------------------------------------------------------
    def evidence_words(self, session_id: str) -> int:
        """How many distinct informative words the customer has given us.

        Measured in words rather than in constraint segments on purpose.
        Segment counts move with phrasing ("leather and 100% Leather" is one
        segment conversationally and two tersely), and the whole point of
        the P1 work was that behaviour must not depend on wording. Word
        counts are stable across all three tested phrasings.
        """
        state = self.state(session_id)
        words: set[str] = set()
        for value in state.active_constraints(self.drop_overridden):
            words.update(tokenise(value))
        return len(words)

    def should_hold(
        self,
        session_id: str,
        turn: int,
        min_words: int = 4,
        max_hold_turn: int = 3,
    ) -> bool:
        """Suggest withholding recommendations and asking a question instead.

        This is a SIGNAL, not a policy. Person A owns the decision (tasks A3
        over-generality detection and A4 proactive guidance); retrieval is
        simply the layer that can see how thin the evidence is, so it
        reports that rather than guessing on A's behalf.

        Why holding pays: answering a vague first turn tends to place the
        target around rank 2, which ends the session at reciprocal rank 0.5.
        Asking once more and answering at rank 1 scores 1.0. The extra turn
        costs efficiency, but efficiency carries weight 0.2 against MRR's
        0.3, so the trade is favourable. Measured on the public set:
        technical 0.8702 -> 0.9092, MRR 0.6727 -> 0.8362, with hit rate
        unchanged at 0.985.

        ``max_hold_turn`` is the safety net. Hit rate is weighted 0.5, so
        the agent must never keep stalling in pursuit of a better rank;
        past this turn it answers with whatever it has.
        """
        return turn <= max_hold_turn and self.evidence_words(session_id) < min_words

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
    def _constraint_weight(self, state: SessionState, index: int) -> float:
        """B4 slot decay: how much a given constraint still counts for."""
        weight = 1.0
        if self.slot_decay != 1.0:
            age = max(0, state.turn - state.constraint_turns[index])
            weight *= self.slot_decay ** age
        if index < state.overridden_before:
            weight *= self.override_penalty
        return weight

    def _rank(self, state: SessionState, pool: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not len(pool):
            return []
        constraints = state.active_constraints(self.drop_overridden)

        # Query text: the stated constraints plus the category label, so a
        # bare browsing turn still ranks sensibly on category words alone.
        # Terms carry the weight of the constraint they came from, so slot
        # decay reaches BM25 rather than only the phrase signal.
        terms: list[str] = []
        term_weights: list[float] = []
        if state.category_label:
            for token in tokenise(state.category_label):
                terms.append(token)
                term_weights.append(1.0)
        offset = state.overridden_before if self.drop_overridden else 0
        for index in range(offset, len(state.constraints)):
            weight = self._constraint_weight(state, index)
            for token in tokenise(state.constraints[index]):
                terms.append(token)
                term_weights.append(weight)

        # --- component rankers, each scored over the same pool ---------
        components: dict[str, np.ndarray] = {
            "lexical": self.lexical.score(terms, pool, term_weights),
            "category": self._category_scores(terms, pool),
        }
        if state.phrases:
            components["phrase"] = self._phrase_evidence(state.phrases, pool)
        # Skipped entirely at weight 0, not just down-weighted: the encode
        # plus gather is ~40x the cost of the whole lexical path, and the
        # sweep in ANNA_dump/log.txt shows it earns nothing on this
        # benchmark. Attaching it is opt-in, and paid for only if used.
        if self.dense is not None and self.weights.get("dense", 0.0) > 0.0:
            components["dense"] = self.dense.score(" ".join(constraints), pool)

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
