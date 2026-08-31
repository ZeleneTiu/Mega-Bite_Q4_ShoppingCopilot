"""Submission safety: contract validation and graceful degradation.

Two risks in the submission rules drive this module.

First, ``docs/submission_rules.md``: "For official final scoring, organizer
policy may disable network access." Any component that calls an LLM API
will fail in that environment.

Second, ``docs/competition_specification.md``: "Exceptions, invalid output,
and timeouts may count as a miss." That is the part worth understanding
precisely. The evaluator already wraps every ``respond`` call:

    try:
        response = agent.respond(...)
    except Exception:
        response = {"message": "", "ask_attribute": None, "recommendations": []}

So a raised exception does not crash the run. It silently substitutes an
EMPTY recommendation list for that turn. If the failure is systemic, such
as no network, it happens on every turn of every session and the score
goes to roughly zero without a single error message in the output.

:class:`ResilientAgent` moves that failure boundary inside ``respond``, so
the fallback is the offline retrieval ranking rather than nothing. Same
failure, different outcome: a network-disabled run degrades to the
retrieval-only score instead of collapsing.

Nothing here imports anything outside the standard library.
"""
from __future__ import annotations

import concurrent.futures
from typing import Any, Callable

# From docs/agent_api_contract.json. Duplicated as a literal rather than
# parsed from the JSON so this module has no file or schema dependency and
# cannot itself become a failure mode.
ALLOWED_ATTRIBUTES: frozenset[str] = frozenset({
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
})

EMPTY_RESPONSE: dict[str, Any] = {
    "message": "",
    "ask_attribute": None,
    "recommendations": [],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}


def validate_response(response: Any, top_k: int = 10) -> list[str]:
    """Check a turn response against the contract. Empty list means valid.

    Returns the violations rather than raising, so callers can log what was
    wrong and still serve something. Task C4 asks for contract compliance;
    this is the check that makes compliance verifiable rather than assumed.
    """
    problems: list[str] = []
    if not isinstance(response, dict):
        return ["response is not a dict"]

    if not isinstance(response.get("message"), str):
        problems.append("message must be a string")

    attribute = response.get("ask_attribute", "__missing__")
    if attribute == "__missing__":
        problems.append("ask_attribute key is required (may be null)")
    elif attribute is not None and attribute not in ALLOWED_ATTRIBUTES:
        problems.append(f"ask_attribute {attribute!r} is not an allowed value")

    recommendations = response.get("recommendations")
    if not isinstance(recommendations, list):
        problems.append("recommendations must be a list")
    else:
        if len(recommendations) > 100:
            problems.append("recommendations exceeds the 100 item cap")
        seen: set[str] = set()
        for position, item in enumerate(recommendations[:top_k]):
            if not isinstance(item, dict):
                problems.append(f"recommendation {position} is not an object")
                continue
            asin = item.get("parent_asin")
            if not isinstance(asin, str) or not asin:
                problems.append(f"recommendation {position} has no valid parent_asin")
                continue
            if asin in seen:
                problems.append(f"recommendation {position} duplicates {asin}")
            seen.add(asin)
            extra = set(item) - {"parent_asin", "score"}
            if extra:
                problems.append(f"recommendation {position} has extra keys {sorted(extra)}")

    usage = response.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            problems.append("usage must be an object when present")
        else:
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    problems.append(f"usage.{key} must be a non-negative integer")
    return problems


def coerce_response(response: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a contract-valid response, repairing what can be repaired."""
    fallback = fallback or EMPTY_RESPONSE
    if not isinstance(response, dict):
        return dict(fallback)
    safe: dict[str, Any] = {
        "message": response["message"] if isinstance(response.get("message"), str) else "",
        "ask_attribute": None,
        "recommendations": [],
    }
    attribute = response.get("ask_attribute")
    if isinstance(attribute, str) and attribute in ALLOWED_ATTRIBUTES:
        safe["ask_attribute"] = attribute

    seen: set[str] = set()
    for item in response.get("recommendations") or []:
        asin = item.get("parent_asin") if isinstance(item, dict) else item
        if not isinstance(asin, str) or not asin or asin in seen:
            continue
        seen.add(asin)
        entry: dict[str, Any] = {"parent_asin": asin}
        if isinstance(item, dict) and isinstance(item.get("score"), (int, float)):
            entry["score"] = float(item["score"])
        safe["recommendations"].append(entry)
        if len(safe["recommendations"]) >= 100:
            break

    usage = response.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
               for v in (prompt, completion)):
            safe["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion}
    safe.setdefault("usage", {"prompt_tokens": 0, "completion_tokens": 0})
    return safe


class ResilientAgent:
    """Wraps the pipeline so no downstream failure can cost a session.

    ``offline_agent`` must be the retrieval-only agent. It has no network,
    no model and no credentials, so it is the thing that always works.

    ``enhance`` is the optional layer that can fail: Person C's LLM
    reranker, Person A's generated clarification message, anything using an
    API. It is called with the baseline response and may return an improved
    one. If it raises, times out, or returns something the contract
    rejects, the baseline is served instead and the session continues.

    Intended for Person D's D1 integration. It is not retrieval logic; it
    is the seatbelt around whatever the integrated agent ends up being.
    """

    def __init__(
        self,
        offline_agent: Any,
        enhance: Callable[..., dict] | None = None,
        timeout_seconds: float | None = 8.0,
    ) -> None:
        self.offline_agent = offline_agent
        self.enhance = enhance
        # A hung API call costs as much as a failed one, and the spec counts
        # timeouts as a miss, so the enhancement gets a wall clock budget.
        self.timeout_seconds = timeout_seconds
        self.failures: list[str] = []
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        self.offline_agent.reset(session_id, user_profile)
        reset_enhancer = getattr(self.enhance, "reset", None)
        if callable(reset_enhancer):
            try:
                reset_enhancer(session_id, user_profile)
            except Exception as error:  # noqa: BLE001 - never fatal
                self.failures.append(f"enhance.reset: {type(error).__name__}: {error}")

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        # The baseline is computed first and unconditionally. If anything at
        # all goes wrong after this point, we already hold a valid answer.
        try:
            baseline = coerce_response(
                self.offline_agent.respond(session_id, user_message, turn, top_k)
            )
        except Exception as error:  # noqa: BLE001
            self.failures.append(f"offline: {type(error).__name__}: {error}")
            return dict(EMPTY_RESPONSE)

        if self.enhance is None:
            return baseline

        try:
            enhanced = self._call_enhance(session_id, user_message, turn, top_k, baseline)
            problems = validate_response(enhanced, top_k)
            if problems:
                self.failures.append(f"enhance contract: {problems[0]}")
                return baseline
            return enhanced
        except Exception as error:  # noqa: BLE001
            self.failures.append(f"enhance: {type(error).__name__}: {error}")
            return baseline

    # ------------------------------------------------------------------
    def _call_enhance(self, session_id, user_message, turn, top_k, baseline) -> dict:
        if not self.timeout_seconds:
            return self.enhance(session_id, user_message, turn, top_k, baseline)
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = self._pool.submit(
            self.enhance, session_id, user_message, turn, top_k, baseline
        )
        # A TimeoutError here propagates to respond's handler, which serves
        # the baseline. The worker thread is left to finish on its own; the
        # alternative, killing it, is not safely possible in Python.
        return future.result(timeout=self.timeout_seconds)

    def failure_summary(self) -> dict[str, int]:
        """Counts by failure kind, for the report's disclosure section."""
        summary: dict[str, int] = {}
        for entry in self.failures:
            key = entry.split(":", 1)[0]
            summary[key] = summary.get(key, 0) + 1
        return summary
