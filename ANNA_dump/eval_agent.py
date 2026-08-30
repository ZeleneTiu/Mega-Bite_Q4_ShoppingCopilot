"""Local harness agent: wraps the B-side retrieval engine in the contest API.

This exists so the retrieval work can be measured on its own, without
touching starter/agent.py (Person D owns that file for D1 integration).
The conversational policy here is a deliberate placeholder standing in for
Person A's router and Person C's reranker.
"""
from __future__ import annotations

from pathlib import Path

from src.retrieval.catalog import Catalog
from src.retrieval.engine import SearchEngine


class Agent:
    # Overridden by run_eval.py so the ask policy can be ablated.
    ASK: str | None = "other"

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl",
                 enable_dense: bool = False, **kwargs) -> None:
        catalog = Catalog.load(str(catalog_path))
        dense = None
        if enable_dense:
            from src.retrieval.dense import DenseIndex
            dense = DenseIndex(catalog)
        self.engine = SearchEngine(catalog=catalog, dense=dense, **kwargs)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.engine.start_session(session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self.engine.observe(session_id, user_message)
        ranked = self.engine.search(session_id, top_k)
        return {
            "message": "Here are the closest matches I found.",
            # Placeholder policy, handed to Person A/C. See run_eval.py notes.
            "ask_attribute": self.ASK,
            "recommendations": [
                {"parent_asin": asin, "score": round(score, 6)} for asin, score in ranked
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
