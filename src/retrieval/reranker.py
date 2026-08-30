"""
reranker.py
-----------
Task C1 (Candidate Scoring & Reranking) + Task C2 (MRR Optimization).

Input:  the candidate pool Person B's hybrid retrieval already produced
        (their job: make sure the right item is SOMEWHERE in this list — Hit Rate@10).
Output: the SAME items, reordered so the correct one is as close to position 1
        as possible (your job — MRR).

Hard rule: never drop or add parent_asins. Only reorder. Dropping an item
that B correctly retrieved breaks Hit Rate@10 too, which isn't your metric
to break.
"""

import json
import re

from .llm_client import LLMClient, LLMResult
from .prompts import RERANK_SYSTEM_PROMPT, build_rerank_user_prompt


class Reranker:
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()

    def rerank(self, candidates: list, session_state: dict, message_history: list, user_profile: dict = None) -> dict:
        """
        candidates:      B's pool for this turn, e.g. [{"parent_asin": ..., "title": ..., ...}]
        session_state:   A's slot-filled per-turn constraints (category/price/details/etc.)
        message_history: recent conversation turns
        user_profile:    the LONG-TERM profile from reset_request — optional, soft signal
                          (purchase_frequency, average_prior_rating, rating_style,
                          preference_tags, summary). Distinct from session_state.

        Returns:
        {
            "ranked": [ {"parent_asin": ..., "score": float}, ... ]  # same set as input, reordered
            "confident": bool,                        # True if LLM felt sure about #1
            "usage": {"prompt_tokens": int, "completion_tokens": int}
        }
        """
        if not candidates:
            return {"ranked": [], "confident": False, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        try:
            user_prompt = build_rerank_user_prompt(candidates, session_state, message_history, user_profile)
            result: LLMResult = self.llm_client.call_llm(RERANK_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_llm_json(result.text)
            ranked = self._reconcile_with_input(parsed.get("ranked", []), candidates)

            return {
                "ranked": ranked,
                "confident": bool(parsed.get("confident", False)),
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            }
        except Exception:
            # Fallback: preserve B's original order untouched. Never let a
            # reranking failure turn into a missing item or a crashed turn.
            return {
                "ranked": [{"parent_asin": c.get("parent_asin")} for c in candidates],
                "confident": False,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """LLMs sometimes wrap JSON in markdown fences or add stray text — strip that."""
        text = text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(match.group(0))

    @staticmethod
    def _reconcile_with_input(llm_ranked: list, original_candidates: list) -> list:
        """
        Safety net for C1/C2: guarantee output is exactly the same set of
        parent_asins as the input, just reordered by the LLM's scores.
        - Drops any asin the LLM hallucinated that wasn't in the input.
        - Appends any asin the LLM forgot to rank, at the end (never lose one).
        """
        original_ids = [c.get("parent_asin") for c in original_candidates]
        original_id_set = set(original_ids)

        seen = set()
        ordered = []
        for item in llm_ranked:
            asin = item.get("parent_asin")
            if asin in original_id_set and asin not in seen:
                entry = {"parent_asin": asin}
                if "score" in item and item["score"] is not None:
                    entry["score"] = item["score"]
                ordered.append(entry)
                seen.add(asin)

        # append anything the LLM missed, preserving B's original order for those
        for asin in original_ids:
            if asin not in seen:
                ordered.append({"parent_asin": asin})
                seen.add(asin)

        return ordered  # top_k truncation happens in contract.build_response, not here
