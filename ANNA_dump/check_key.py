"""Preflight: is the key live, and is the configured model real?

Run this BEFORE run_integrated.py --compare-key. A 281-call run against a bad
key or a wrong model id does not error out -- every call fails, the reranker
falls back to B's order, and the score comes back looking exactly like a
keyless run. That is the same class of silent failure that let C sit inert for
a fortnight, so it gets its own two-second check.

  python ANNA_dump/check_key.py
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rerank.llm_client import LLMClient  # noqa: E402


def main() -> int:
    client = LLMClient()
    if not client.available:
        print("NO KEY FOUND.")
        print("  Put it in .env as   ANTHROPIC_API_KEY=sk-ant-...")
        print("  or in PowerShell:   $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        return 1

    key = client.api_key
    print("key found: %s...%s  (%d chars)" % (key[:14], key[-4:], len(key)))
    if not key.startswith("sk-ant-"):
        print("  WARNING: does not start with sk-ant- -- wrong key pasted?")

    request = urllib.request.Request(
        "https://api.anthropic.com/v1/models?limit=50",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            models = [m["id"] for m in json.load(response).get("data", [])]
    except urllib.error.HTTPError as error:
        detail = error.read().decode()[:200]
        print("AUTH FAILED: HTTP %d %s" % (error.code, detail))
        if error.code == 401:
            print("  -> the key is not recognised. Get a fresh one at")
            print("     console.anthropic.com -> API keys.")
        return 1
    except Exception as error:                       # noqa: BLE001
        print("NO NETWORK: %s: %s" % (type(error).__name__, str(error)[:160]))
        print("  -> the graded run may be offline too; that path is already")
        print("     covered, the agent scores 0.909154 with no key at all.")
        return 1

    print("auth OK. %d models visible." % len(models))
    ok = client.model in models
    print("configured model: %s   %s" % (client.model, "VALID" if ok else "NOT IN THE LIST"))
    if not ok:
        print("  every call would 404 and fall back silently. Pick one of:")
        for name in models[:8]:
            print("    ", name)
        print("  then set it in src/rerank/llm_client.py (LLMClient.model).")
        return 1

    print("\nready. Next:  python ANNA_dump/run_integrated.py --compare-key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
