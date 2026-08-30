from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

# Import routers from the parent's Conversational Logic folder
conversational_logic_path = Path(__file__).parent.parent / "Conversational Logic"
intent_router_path = conversational_logic_path / "intent_router.py"
state_router_path = conversational_logic_path / "state_router.py"

# Load intent_router module
spec = spec_from_file_location("intent_router", intent_router_path)
intent_router_module = module_from_spec(spec)
spec.loader.exec_module(intent_router_module)
IntentRouter = intent_router_module.IntentRouter
IntentResult = intent_router_module.IntentResult

# Load state_router module
spec = spec_from_file_location("state_router", state_router_path)
state_router_module = module_from_spec(spec)
spec.loader.exec_module(state_router_module)
StateTracker = state_router_module.StateTracker
ConversationState = state_router_module.ConversationState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    """Agent with intent routing, state tracking, and over-generality detection."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        
        # Initialize routers
        self.intent_router = IntentRouter()
        self.state_tracker = StateTracker()
        
        # Session state management
        self._session_states: dict[str, ConversationState] = {}
        
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)
        # Initialize conversation state for this session
        self._session_states[session_id] = ConversationState(session_id=session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        
        state = self._session_states[session_id]
        
        # Step 1: Detect intent from user message with dynamic weighting based on filled slots
        intent_result: IntentResult = self.intent_router.route(user_message, state.slots)
        
        # Step 2: Update conversation state with constraints and intent
        state = self.state_tracker.update_state(
            state, 
            user_message, 
            intent_result.detected_constraints,
            intent_result.track.value
        )
        
        # Step 3: Check for over-generality before retrieval
        # First, get candidate count to check threshold
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        
        if expression:
            candidate_rows = self.connection.execute(
                "SELECT COUNT(*) FROM products WHERE products MATCH ?",
                (expression,),
            ).fetchone()
            candidate_count = candidate_rows[0] if candidate_rows else 0
        else:
            candidate_count = 0
        
        # Check if query is over-generalized
        if self.state_tracker.check_over_generality(state, candidate_count):
            clarification_prompt = self.state_tracker.generate_clarification_prompt(state)
            missing_attributes = self.state_tracker._get_prioritized_missing_attributes(state)
            ask_attribute = missing_attributes[0] if missing_attributes else "other"
            if ask_attribute == "price range":
                ask_attribute = "budget"
            return {
                "message": clarification_prompt,
                "ask_attribute": ask_attribute,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        
        # Step 4: Perform retrieval
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
