"""Proves the three integration gaps are closed and stay closed.

Before these, deleting Person A and Person C from src/pipeline.py changed the
public score by exactly nothing (0.870196 either way), the evidence gate was
built but never called, and Agent() raised FileNotFoundError from any working
directory but the repo root -- a failure the evaluator swallows into an empty
recommendation list for every turn of every session.

Each test here is the negative control for one of those: it fails if the wire
comes loose again.

Run with:  python -m unittest tests.test_integration_flow -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Pipeline, resolve_catalog_path  # noqa: E402
from src.rerank import Reranker  # noqa: E402
from src.rerank.llm_client import LLMResult  # noqa: E402
from src.rerank.contract import validate_response  # noqa: E402
from starter.agent import Agent  # noqa: E402

CATALOG = "data/catalog.jsonl"

# A vague opener: two informative words, below the gate's four-word floor.
THIN_TURN = "I want a jacket"
# Enough disclosed detail that the gate must stand down.
RICH_TURN = "navy blue waterproof rain shell, size medium, sealed seams"


class _FakeClient:
    """A stand-in for a real key. No network, same code path."""

    available = True

    def __init__(self, payload: str = '{"ranked": [], "confident": false}',
                 delay: float = 0.0) -> None:
        self.payload = payload
        self.delay = delay
        self.calls = 0

    def call_llm(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return LLMResult(self.payload, 100, 25)


class EvidenceGateTest(unittest.TestCase):
    """Gap (c): the gate existed on the engine and nothing called it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = Pipeline(CATALOG, enable_rerank=False)

    def test_thin_turn_withholds_and_asks(self):
        """Withholding IS the mechanism: worth +0.039 technical, measured."""
        self.pipeline.reset("thin", {})
        response = self.pipeline.step("thin", THIN_TURN, 1, 10)
        self.assertEqual(response["recommendations"], [])
        self.assertIn(response["ask_attribute"], {"category", "material", "color",
                                                  "size", "style", "brand", "budget",
                                                  "feature", "use_case", "other"})
        self.assertTrue(response["message"].strip(), "a held turn must still say something")
        validate_response(response)

    def test_rich_turn_answers_immediately(self):
        """The gate must not stall a customer who already said enough."""
        self.pipeline.reset("rich", {})
        response = self.pipeline.step("rich", RICH_TURN, 1, 10)
        self.assertEqual(len(response["recommendations"]), 10)

    def test_gate_stands_down_after_turn_three(self):
        """Hit rate carries weight 0.5. The agent must never stall forever."""
        self.pipeline.reset("stall", {})
        for turn in (1, 2, 3):
            self.assertEqual(self.pipeline.step("stall", THIN_TURN, turn, 10)["recommendations"], [])
        late = self.pipeline.step("stall", THIN_TURN, 4, 10)
        self.assertEqual(len(late["recommendations"]), 10)

    def test_gate_can_be_switched_off(self):
        ungated = Pipeline(CATALOG, enable_rerank=False, enable_evidence_gate=False)
        ungated.reset("off", {})
        self.assertEqual(len(ungated.step("off", THIN_TURN, 1, 10)["recommendations"]), 10)


class RerankSafetyTest(unittest.TestCase):
    """Gap (b): C never executed, and a graded run may have no network."""

    CANDIDATES = [
        {"parent_asin": "B0AAAAAAAA", "score": 2.0, "title": "one"},
        {"parent_asin": "B0BBBBBBBB", "score": 1.0, "title": "two"},
    ]

    def test_no_key_costs_nothing_and_keeps_b_scores(self):
        """An offline turn must not build a prompt, and must not lose scores."""
        reranker = Reranker()
        if reranker.available:                      # a key is present in this shell
            self.skipTest("ANTHROPIC_API_KEY is set; this asserts the offline path")
        result = reranker.rerank(self.CANDIDATES, {}, [])
        self.assertEqual(reranker.stats["no_key"], 1)
        self.assertEqual([r["parent_asin"] for r in result["ranked"]],
                         ["B0AAAAAAAA", "B0BBBBBBBB"])
        self.assertEqual(result["ranked"][0]["score"], 2.0)
        self.assertEqual(result["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_hung_call_is_bounded_and_falls_back(self):
        """A hung turn costs a session exactly as much as a failed one."""
        reranker = Reranker(llm_client=_FakeClient(delay=5.0), timeout_seconds=0.5)
        started = time.time()
        result = reranker.rerank(self.CANDIDATES, {}, [])
        self.assertLess(time.time() - started, 2.0)
        self.assertEqual(reranker.stats["timed_out"], 1)
        self.assertEqual(result["ranked"][0]["parent_asin"], "B0AAAAAAAA")

    def test_unparseable_reply_falls_back(self):
        reranker = Reranker(llm_client=_FakeClient(payload="I am not JSON"))
        result = reranker.rerank(self.CANDIDATES, {}, [])
        self.assertEqual(reranker.stats["failed"], 1)
        self.assertEqual([r["parent_asin"] for r in result["ranked"]],
                         ["B0AAAAAAAA", "B0BBBBBBBB"])

    def test_hallucinated_and_missing_asins_are_reconciled(self):
        """The output set must equal the input set, whatever the model says."""
        payload = ('{"ranked": [{"parent_asin": "B0ZZZZZZZZ", "score": 9.0}, '
                   '{"parent_asin": "B0BBBBBBBB", "score": 5.0}], "confident": true}')
        reranker = Reranker(llm_client=_FakeClient(payload=payload))
        result = reranker.rerank(self.CANDIDATES, {}, [])
        self.assertEqual([r["parent_asin"] for r in result["ranked"]],
                         ["B0BBBBBBBB", "B0AAAAAAAA"])
        # The asin the model never ranked keeps B's score, so it stays explainable.
        self.assertEqual(result["ranked"][1]["score"], 2.0)

    def test_llm_path_actually_runs_when_a_client_is_available(self):
        """Guards against C going inert again without anyone noticing."""
        payload = '{"ranked": [{"parent_asin": "B0BBBBBBBB", "score": 5.0}], "confident": true}'
        client = _FakeClient(payload=payload)
        reranker = Reranker(llm_client=client)
        result = reranker.rerank(self.CANDIDATES, {}, [])
        self.assertEqual(client.calls, 1)
        self.assertEqual(reranker.stats["llm_ok"], 1)
        self.assertEqual(result["usage"]["prompt_tokens"], 100)


class GradingSafetyTest(unittest.TestCase):
    """Gap: Agent() died outside the repo root, and rebuilt the index every time."""

    def test_agent_constructs_from_a_foreign_working_directory(self):
        script = (
            "import sys; sys.path.insert(0, r'%s')\n"
            "from starter.agent import Agent\n"
            "a = Agent()\n"
            "a.reset('s', {})\n"
            "r = a.respond('s', %r, 1, 10)\n"
            "print(len(r['recommendations']))\n" % (str(ROOT), RICH_TURN)
        )
        proc = subprocess.run([sys.executable, "-c", script], cwd=os.path.dirname(str(ROOT)),
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(proc.stdout.strip().splitlines()[-1], "10")

    def test_relative_path_resolves_against_the_repo_root(self):
        self.assertTrue(Path(resolve_catalog_path("data/catalog.jsonl")).is_file())

    def test_second_construction_reuses_the_built_index(self):
        """800 sessions x a 6.9s rebuild is 92 minutes of doing nothing."""
        first = Agent(CATALOG)
        started = time.time()
        second = Agent(CATALOG)
        self.assertLess(time.time() - started, 1.0)
        self.assertIs(first.pipeline.engine, second.pipeline.engine)


if __name__ == "__main__":
    unittest.main()
