"""D2: session memory + context distillation.

This is the object that crosses module boundaries. Person A writes slot state
into it; Person C reads a clean, compact view out of it. Person D owns it.

Two things live here per session:
  * the LONG-TERM user_profile handed to us by reset() (contract-guaranteed
    shape: purchase_frequency, average_prior_rating, rating_style,
    preference_tags, summary) -- passed straight through to the reranker as a
    soft signal.
  * the SHORT-TERM conversational state (Person A's ConversationState, holding
    the accumulated slots) -- distilled by session_state() into the dict the
    reranker's prompt builder expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.intent.state_router import ConversationState


@dataclass
class SessionMemory:
    session_id: str
    user_profile: dict = field(default_factory=dict)   # long-term, from reset()
    state: ConversationState | None = None             # short-term, owned by A's tracker
    history: list = field(default_factory=list)        # raw turns, newest last

    def record_turn(self, role: str, text: str) -> None:
        self.history.append({"role": role, "text": text})

    def session_state(self) -> dict:
        """Distill A's raw slots into the compact dict C's prompt consumes.

        This IS the 'context distillation' deliverable: it turns the state
        machine's internal representation into a stable, documented shape that
        the reranker depends on, so neither side has to know the other's
        internals. Empty slots are dropped to keep the prompt (and token cost)
        tight.
        """
        if self.state is None:
            return {}
        s = self.state.slots
        distilled = {
            "category": s.get("main_category"),
            "categories": s.get("categories"),
            "store": s.get("store"),
            "price_max": s.get("price_max"),
            "price_min": s.get("price_min"),
            "min_rating": s.get("min_rating"),
            "details": s.get("details", {}),   # size / color / material
        }
        return {k: v for k, v in distilled.items() if v not in (None, [], {}, "")}


class MemoryStore:
    """Holds every session. A and B keep their own internal state; this is the
    shared layer D controls and threads through the pipeline."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}

    def start(self, session_id: str, user_profile: dict, state: ConversationState) -> SessionMemory:
        mem = SessionMemory(session_id=session_id, user_profile=dict(user_profile or {}), state=state)
        self._sessions[session_id] = mem
        return mem

    def get(self, session_id: str) -> SessionMemory:
        return self._sessions[session_id]
