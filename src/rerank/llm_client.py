import json
import os
from dataclasses import dataclass
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
_ENV_LOADED = False


def load_dotenv_once() -> None:
    """Read the repo-root .env into os.environ, once, without a dependency.

    The code used to say "read from env / .env as the README instructs" while
    nothing anywhere read .env, so a key placed there would have been ignored
    in silence -- and a silently keyless run is exactly the failure that made
    C invisible in the first place. Existing environment variables always win,
    so an exported key still overrides the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        text = _ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and value and name not in os.environ:
            os.environ[name] = value


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


# Split so a blackholed network cannot burn the full budget on the connect.
# The graded run may have no network at all (docs/submission_rules.md); a
# dropped packet there looks like a hang, not a refusal.
CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 12.0


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 500):
        self.model = model
        self.max_tokens = max_tokens
        # NOTE: never commit real keys. .env is gitignored; see .env.example.
        load_dotenv_once()
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        """Whether a call could possibly succeed.

        Callers check this BEFORE building a prompt, so an offline graded run
        costs nothing per turn instead of building a prompt, importing
        requests and raising, 438 times over.
        """
        return bool(self.api_key)

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
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
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
