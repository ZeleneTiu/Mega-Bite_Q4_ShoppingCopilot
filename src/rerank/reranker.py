import concurrent.futures
import json
import re

from .llm_client import LLMClient, LLMResult
from .prompts import RERANK_SYSTEM_PROMPT, build_rerank_user_prompt


class Reranker:
    """C2: LLM reordering of B's candidate pool, with a hard failure budget.

    Three guarantees, because the evaluator zeroes a whole session on any
    exception and counts a hung turn exactly like a failed one:

      1. No key, no work. ``LLMClient.available`` is checked before a prompt
         is built, so an offline graded run pays nothing per turn.
      2. A hung call cannot outlive ``timeout_seconds`` of wall clock. The
         request timeout alone is not enough -- it does not bound DNS, TLS or
         a retry -- so the call also runs on a worker with a deadline.
      3. Any failure returns B's ordering WITH B's scores intact, so falling
         back is indistinguishable from never having been asked.

    ``stats`` records what actually happened, so an integration run can prove
    whether the LLM path executed rather than assuming it did.
    """

    def __init__(self, llm_client: LLMClient = None, timeout_seconds: float | None = 8.0):
        self.llm_client = llm_client or LLMClient()
        self.timeout_seconds = timeout_seconds
        self.stats = {"llm_ok": 0, "no_key": 0, "failed": 0, "timed_out": 0}
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def available(self) -> bool:
        return bool(getattr(self.llm_client, "available", True))

    def rerank(self, candidates: list, session_state: dict, message_history: list, user_profile: dict = None) -> dict:
        """
        candidates:      B's pool for this turn, e.g. [{"parent_asin": ..., "title": ..., ...}]
        session_state:   A's slot-filled per-turn constraints (category/price/details/etc.)
        message_history: recent conversation turns
        user_profile:    the LONG-TERM profile from reset_request -- optional, soft signal
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

        # Cheapest possible offline path: no prompt built, no import, no raise.
        if not self.available:
            self.stats["no_key"] += 1
            return self._fallback(candidates)

        try:
            user_prompt = build_rerank_user_prompt(candidates, session_state, message_history, user_profile)
            result: LLMResult = self._call_with_deadline(RERANK_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_llm_json(result.text)
            ranked = self._reconcile_with_input(parsed.get("ranked", []), candidates)
            self.stats["llm_ok"] += 1

            return {
                "ranked": ranked,
                "confident": bool(parsed.get("confident", False)),
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                },
            }
        except concurrent.futures.TimeoutError:
            self.stats["timed_out"] += 1
            return self._fallback(candidates)
        except Exception:
            # Fallback: preserve B's original order untouched. Never let a
            # reranking failure turn into a missing item or a crashed turn.
            self.stats["failed"] += 1
            return self._fallback(candidates)

    # ------------------------------------------------------------------
    def _call_with_deadline(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """Wall-clock bound on the whole call, not just the socket read.

        The worker thread is left to finish on its own if it overruns; killing
        a thread is not safely possible in Python, and the turn has already
        been served from B's ordering by then.
        """
        if not self.timeout_seconds:
            return self.llm_client.call_llm(system_prompt, user_prompt)
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rerank"
            )
        future = self._pool.submit(self.llm_client.call_llm, system_prompt, user_prompt)
        return future.result(timeout=self.timeout_seconds)

    @staticmethod
    def _fallback(candidates: list) -> dict:
        """B's order, B's scores. Indistinguishable from never having asked."""
        ranked = []
        for c in candidates:
            entry = {"parent_asin": c.get("parent_asin")}
            score = c.get("score")
            if score is not None:
                entry["score"] = score
            ranked.append(entry)
        return {"ranked": ranked, "confident": False, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """LLMs sometimes wrap JSON in markdown fences or add stray text - strip that."""
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
        - Appends any asin the LLM forgot to rank, at the end (never lose one),
          carrying B's score so an unranked item is still explainable.
        """
        original_ids = [c.get("parent_asin") for c in original_candidates]
        original_id_set = set(original_ids)
        original_scores = {c.get("parent_asin"): c.get("score") for c in original_candidates}

        seen = set()
        ordered = []
        for item in llm_ranked:
            asin = item.get("parent_asin")
            if asin in original_id_set and asin not in seen:
                entry = {"parent_asin": asin}
                if "score" in item and item["score"] is not None:
                    entry["score"] = item["score"]
                elif original_scores.get(asin) is not None:
                    entry["score"] = original_scores[asin]
                ordered.append(entry)
                seen.add(asin)

        # append anything the LLM missed, preserving B's original order for those
        for asin in original_ids:
            if asin not in seen:
                entry = {"parent_asin": asin}
                if original_scores.get(asin) is not None:
                    entry["score"] = original_scores[asin]
                ordered.append(entry)
                seen.add(asin)

        return ordered  # top_k truncation happens in contract.build_response, not here
