"""Integrated agent: thin adapter over src/pipeline.py.

The evaluator imports THIS Agent class and calls reset()/respond(). All real
logic lives in the pipeline; this file only adapts the contract and guarantees
respond() never raises -- the evaluator zeroes an entire session on any
exception, so a single bad turn must not take the whole session down.
"""
from __future__ import annotations

from pathlib import Path

from src.pipeline import Pipeline


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        # enable_rerank=True is safe: with no ANTHROPIC_API_KEY the reranker
        # falls back to B's ordering, so we never drop below the retrieval
        # baseline. Flip enable_clarification on only after measuring it.
        self.pipeline = Pipeline(str(catalog_path), enable_rerank=True, enable_clarification=False)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.pipeline.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self.pipeline.step(session_id, user_message, turn, top_k)
        except Exception:
            return {
                "message": "",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
