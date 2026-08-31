RERANK_SYSTEM_PROMPT = (
    "You are a shopping recommendation reranker. You are given a customer's stated "
    "preferences and a list of candidate products. Reorder ALL candidates from best "
    "to worst fit. Use only the information provided; do not invent products.\n\n"
    "Return ONLY a JSON object, no prose, no markdown fences, of the form:\n"
    '{"ranked": [{"parent_asin": "<id>", "score": <float 0..1>}, ...], "confident": <true|false>}\n\n'
    "Include every candidate exactly once."
)
def build_rerank_user_prompt(candidates: list, session_state: dict, message_history: list, user_profile: dict = None) -> str:
    """
    candidates:      list of dicts from Person B, e.g. [{"parent_asin": "...", "title": "...",
                     "store": "...", "price": ..., "categories": [...], "details": {...}}, ...]
    session_state:   slot-filled constraints from Person A's state tracker for THIS session
                     e.g. {"category": "sneakers", "price": "<50", "details": {"size": "9"}}
    message_history: list of recent turns, e.g. [{"role": "user", "text": "..."}, ...]
    user_profile:    optional LONG-TERM signal from reset_request, e.g.
                     {"purchase_frequency": "...", "average_prior_rating": 4.2,
                      "rating_style": "...", "preference_tags": [...], "summary": "..."}
                     This is soft/background context — session_state (explicit, this-turn
                     constraints) should always win if the two ever conflict.
    """
    # Keep candidate info compact — only what's needed to judge fit, to save tokens.
    compact_candidates = [
        {
            "parent_asin": c.get("parent_asin"),
            "title": c.get("title"),
            "store": c.get("store"),
            "price": c.get("price"),
            "details": c.get("details"),
        }
        for c in candidates
    ]

    recent_turns = message_history[-4:] if message_history else []

    profile_block = ""
    if user_profile:
        # Only pull the soft-signal fields — keep it short, these are hints not rules.
        tags = user_profile.get("preference_tags") or []
        summary = user_profile.get("summary") or ""
        if tags or summary:
            profile_block = (
                f"\nBackground on this customer (soft signal only — explicit preferences "
                f"below always take priority if they conflict):\n"
                f"tags: {tags}\nsummary: {summary}\n"
            )

    return (
        f"Known customer preferences so far (explicit, this session):\n{session_state}\n"
        f"{profile_block}\n"
        f"Recent conversation:\n{recent_turns}\n\n"
        f"Candidate products to rank:\n{compact_candidates}\n\n"
        f"Rank all {len(compact_candidates)} candidates now."
    )
