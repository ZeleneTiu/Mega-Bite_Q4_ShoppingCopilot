"""Proves the submission survives a network-disabled grading run.

docs/submission_rules.md: "For official final scoring, organizer policy may
disable network access." docs/competition_specification.md: "Exceptions,
invalid output, and timeouts may count as a miss."

Without these tests, "we have an offline fallback" is a claim in a report.
With them it is a fact, and any future change that reintroduces a network
dependency fails here rather than silently in the graded run.

Run with:  python -m unittest tests.test_offline_safety -v
"""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ANNA_dump.eval_agent import Agent  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.safety import ResilientAgent, coerce_response, validate_response  # noqa: E402

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
# The retrieval-only pipeline scores 0.9092 on the public set. The floor is
# set below that to absorb sampling and tuning drift without being so loose
# that a real regression slips through.
MINIMUM_TECHNICAL_SCORE = 0.85


class _NetworkDisabled:
    """Context manager that makes any socket creation fail, as a locked-down
    grading environment would."""

    def __enter__(self):
        self._socket = socket.socket
        self._create = socket.create_connection
        self._getaddrinfo = socket.getaddrinfo

        def blocked(*args, **kwargs):
            raise OSError("network access is disabled")

        socket.socket = blocked
        socket.create_connection = blocked
        socket.getaddrinfo = blocked
        return self

    def __exit__(self, *exc):
        socket.socket = self._socket
        socket.create_connection = self._create
        socket.getaddrinfo = self._getaddrinfo
        return False


class OfflineSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not Path(CATALOG).exists():
            raise unittest.SkipTest(f"{CATALOG} not present")
        cls.samples = load_jsonl(DATASET)
        cls.ids, cls.cats, cls.prods = catalog_index(CATALOG)

    # -- the headline guarantee ----------------------------------------
    def test_full_evaluation_runs_with_network_disabled(self) -> None:
        """The whole 200-session run must complete and score, with no network."""
        with _NetworkDisabled():
            agent = Agent(CATALOG)
            result = evaluate(agent, self.samples, self.ids, self.cats, self.prods)
        score = result["recommended_technical_score"]
        self.assertGreaterEqual(
            score, MINIMUM_TECHNICAL_SCORE,
            f"offline technical score {score:.4f} fell below {MINIMUM_TECHNICAL_SCORE}",
        )
        self.assertEqual(result["reported_token_usage"]["total_tokens"], 0,
                         "the offline path must not report token usage")

    def test_every_turn_response_satisfies_the_contract(self) -> None:
        """Invalid output may count as a miss, so check every turn, not one."""
        with _NetworkDisabled():
            agent = Agent(CATALOG)
            for sample in self.samples[:40]:
                session = sample["sample_id"]
                agent.reset(session, sample["user_profile"])
                for turn in (1, 2, 3):
                    response = agent.respond(session, "I'm looking for Jewelry Necklaces.", turn, 10)
                    problems = validate_response(response, 10)
                    self.assertEqual(problems, [], f"{session} turn {turn}: {problems}")

    # -- the seatbelt ---------------------------------------------------
    def test_enhancer_that_raises_falls_back_to_retrieval(self) -> None:
        """A failing LLM layer must cost nothing but the enhancement."""
        def always_fails(session_id, message, turn, top_k, baseline):
            raise ConnectionError("no network at scoring time")

        with _NetworkDisabled():
            agent = ResilientAgent(Agent(CATALOG), enhance=always_fails)
            result = evaluate(agent, self.samples, self.ids, self.cats, self.prods)
        self.assertGreaterEqual(result["recommended_technical_score"], MINIMUM_TECHNICAL_SCORE)
        self.assertTrue(agent.failures, "the failure should have been recorded, not hidden")

    def test_enhancer_returning_invalid_output_is_rejected(self) -> None:
        """A malformed enhancement must not reach the evaluator."""
        def returns_rubbish(session_id, message, turn, top_k, baseline):
            return {"message": 42, "ask_attribute": "not_an_attribute", "recommendations": "nope"}

        agent = ResilientAgent(Agent(CATALOG), enhance=returns_rubbish)
        agent.reset("s", {})
        response = agent.respond("s", "I'm looking for Jewelry Necklaces.", 1, 10)
        self.assertEqual(validate_response(response, 10), [])
        self.assertTrue(any("contract" in f for f in agent.failures))

    def test_enhancer_that_hangs_is_timed_out(self) -> None:
        """A hung call costs a session just like a failed one."""
        import time

        def hangs(session_id, message, turn, top_k, baseline):
            time.sleep(30)

        agent = ResilientAgent(Agent(CATALOG), enhance=hangs, timeout_seconds=0.5)
        agent.reset("s", {})
        started = time.time()
        response = agent.respond("s", "I'm looking for Jewelry Necklaces.", 1, 10)
        self.assertLess(time.time() - started, 5.0, "the timeout did not fire")
        self.assertEqual(validate_response(response, 10), [])

    def test_retrieval_modules_import_nothing_network_capable(self) -> None:
        """Guards against a future import quietly reintroducing a dependency."""
        import importlib
        for name in ("catalog", "category", "engine", "fusion", "lexical", "text"):
            module = importlib.import_module(f"src.retrieval.{name}")
            source = Path(module.__file__).read_text(encoding="utf-8")
            for banned in ("import requests", "import httpx", "import urllib",
                           "import openai", "from openai", "sentence_transformers"):
                self.assertNotIn(banned, source, f"{name}.py references {banned}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
