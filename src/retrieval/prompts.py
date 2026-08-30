"""
prompts.py
----------
Task C3 (Prompt Engineering for Low MTTC).

Goal: the reranking prompt should make the LLM DECISIVE, not hedge-y.
Every extra turn spent asking a question the LLM could've resolved with
what it already has costs MTTC. So the prompt explicitly asks the model to:
  1. Score every candidate against everything known so far (session_state).
  2. Only flag "low confidence" if it genuinely cannot distinguish the top item.
  3. Justify its #1 pick briefly (helps debugging, costs few tokens).

Keep this prompt SHORT. Long system prompts burn prompt_tokens which counts
against the disclosed token-usage/cost metric, even though it isn't part of
TechnicalScore directly — still bad to bloat it for no reason.
"""

RERANK_SYSTEM_PROMPT = """You are a product-ranking assistant for an e-commerce search agent.
You will be given a customer's known preferences and a pool of candidate products
(already pre-filtered by a search engine — do not invent new items, only rank the ones given).

Your job:
- Rank the candidates from BEST fit to WORST fit for what the customer wants.
- Put the single best match first. Being right at position 1 matters most.
- Give each candidate a confidence score from 0.0 to 1.0 for how well it matches.
- If your top pick is well above the rest, mark "confident": true.
- If several items are close and you genuinely cannot tell which is correct,
  mark "confident": false — but only do this when it's truly ambiguous, not out of caution.

Respond with ONLY valid JSON, no other text, in this exact shape:
{
  "ranked": [
    {"parent_asin": "...", "score": 0.0, "reason": "short phrase"},
    ...
  ],
  "confident": true,
  "note": "one short sentence, optional"
}
"""


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
