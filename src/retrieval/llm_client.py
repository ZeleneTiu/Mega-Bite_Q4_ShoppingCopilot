"""
llm_client.py
--------------
Thin wrapper around whichever LLM API the team picks (Anthropic, OpenAI, local, etc.)
Its only job: call the model, return the text, and report token usage so it can be
plugged straight into the `usage` field of the agent's API contract response.

Swap out `_call_anthropic` for whatever provider you actually use — keep the
call_llm() signature the same so reranker.py doesn't need to change.
"""

import json
import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 500):
        self.model = model
        self.max_tokens = max_tokens
        # NOTE: never commit real keys. Read from env / .env as the README instructs.
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def call_llm(self, system_prompt: str, user_prompt: str) -> LLMResult:
        """
        Returns an LLMResult with .text plus token counts.
        Raises on hard failure — callers (reranker.py) MUST catch this and fall back.
        """
        import requests  # keep import local so this file doesn't hard-crash if unused

        if not self.api_key:
            raise RuntimeError("No API key set (check .env / ANTHROPIC_API_KEY)")

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )
        usage = data.get("usage", {})

        return LLMResult(
            text=text,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )
