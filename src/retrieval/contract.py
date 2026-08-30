ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other", None,
}

# turn_request.top_k is always const 10 in the schema, but the schema itself
# allows up to 100 recommendation items. Default to the request's top_k.
DEFAULT_TOP_K = 10
SCHEMA_MAX_RECOMMENDATIONS = 100


class ContractViolation(Exception):
    pass


def build_response(message: str, ask_attribute, ranked: list, usage: dict = None, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Assembles and validates the final dict `respond()` should return.
    Call this as the LAST step before returning from agent.py, so nothing
    downstream can accidentally produce a malformed response.

    `ranked` items may optionally carry a "score" (float) — passed through
    if present, since the schema allows it and it's useful for debugging/judging.
    """
    if ask_attribute not in ALLOWED_ATTRIBUTES:
        raise ContractViolation(f"ask_attribute '{ask_attribute}' not in allowed set")

    if not isinstance(ranked, list):
        raise ContractViolation("recommendations must be a list")

    cap = min(top_k, SCHEMA_MAX_RECOMMENDATIONS)
    recommendations = []
    seen = set()
    for item in ranked:
        asin = item.get("parent_asin") if isinstance(item, dict) else None
        if not asin or asin in seen:
            continue
        rec = {"parent_asin": asin}
        if isinstance(item, dict) and "score" in item and item["score"] is not None:
            rec["score"] = float(item["score"])
        recommendations.append(rec)
        seen.add(asin)
        if len(recommendations) >= cap:
            break

    response = {
        "message": message or "",
        "ask_attribute": ask_attribute,
        "recommendations": recommendations,
        "usage": {
            "prompt_tokens": int((usage or {}).get("prompt_tokens", 0)),
            "completion_tokens": int((usage or {}).get("completion_tokens", 0)),
        },
    }
    return response


def validate_response(response: dict) -> None:
    """Sanity check you can call in tests/ before wiring into agent.py."""
    required_keys = {"message", "ask_attribute", "recommendations"}  # usage is NOT required by schema
    missing = required_keys - response.keys()
    if missing:
        raise ContractViolation(f"missing required keys: {missing}")

    allowed_top_level = {"message", "ask_attribute", "recommendations", "usage"}
    extra = response.keys() - allowed_top_level
    if extra:
        raise ContractViolation(f"response has disallowed extra keys: {extra}")

    if response["ask_attribute"] not in ALLOWED_ATTRIBUTES:
        raise ContractViolation("bad ask_attribute")

    if len(response["recommendations"]) > SCHEMA_MAX_RECOMMENDATIONS:
        raise ContractViolation("too many recommendations (schema max is 100)")

    for r in response["recommendations"]:
        allowed_rec_keys = {"parent_asin", "score"}
        extra_rec = r.keys() - allowed_rec_keys
        if extra_rec:
            raise ContractViolation(f"recommendation has disallowed keys: {extra_rec}")
        if "parent_asin" not in r or not r["parent_asin"]:
            raise ContractViolation("recommendation missing parent_asin")

    if "usage" in response:
        usage = response["usage"]
        if "prompt_tokens" not in usage or "completion_tokens" not in usage:
            raise ContractViolation("usage present but missing token fields")
