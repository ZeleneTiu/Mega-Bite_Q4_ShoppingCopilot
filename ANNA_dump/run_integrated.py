"""Run the public-set evaluator against the INTEGRATED agent (starter.agent).

run_eval.py measures the B-side engine alone. This one measures what actually
gets graded: A's router, B's retrieval, B's evidence gate and C's reranker, in
the pipeline they ship in.

It exists because the merged pipeline scored 0.870196 while A and C
contributed literally nothing -- deleting both changed the score by 0.000000.
So this harness prints WHAT RAN, not just what scored: the reranker's own tally
of LLM calls, fallbacks and timeouts, plus reported token usage. A keyless run
is visibly keyless instead of quietly keyless.

  python ANNA_dump/run_integrated.py                     # shipped config
  python ANNA_dump/run_integrated.py --ablate            # every flag, one table
  python ANNA_dump/run_integrated.py --fake-llm reverse  # C's worst case, no network
  python ANNA_dump/run_integrated.py --compare-key       # keyless vs key, side by side

--compare-key is the one to run once ANTHROPIC_API_KEY is set: it scores the
same 200 sessions twice, once with the key hidden from the reranker and once
with it, so C's contribution is a measured number rather than a hope.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.rerank import Reranker  # noqa: E402
from src.rerank.llm_client import LLMResult  # noqa: E402
from src.safety import coerce_response  # noqa: E402

# Retrieval-only reference: B's engine with the evidence gate on. The
# integrated agent must not score below this, ever. If it does, the
# conversation layer is taking something away rather than adding to it.
RETRIEVAL_ONLY_REFERENCE = 0.909154

EMPTY = {"message": "", "ask_attribute": None, "recommendations": [],
         "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


class IntegratedAgent:
    """The shipped adapter, with the pipeline flags exposed for ablation."""

    def __init__(self, catalog_path: str, **flags) -> None:
        self.pipeline = Pipeline(catalog_path, **flags)
        self.sessions_done = 0
        self.progress_total = 0     # set by score() when a live run is expected

    def reset(self, session_id: str, user_profile: dict) -> None:
        # A live reranker run is ~281 sequential API calls. Without a tick the
        # harness looks hung for a quarter of an hour, which is how you end up
        # killing a run that was working.
        if self.progress_total:
            self.sessions_done += 1
            elapsed = time.time() - self._started
            rate = elapsed / max(self.sessions_done, 1)
            left = rate * (self.progress_total - self.sessions_done)
            print("\r  session %d/%d  %.0fs elapsed, ~%.0fs left   " % (
                self.sessions_done, self.progress_total, elapsed, left),
                end="", file=sys.stderr, flush=True)
        self.pipeline.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return coerce_response(self.pipeline.step(session_id, user_message, turn, top_k))
        except Exception:
            return dict(EMPTY)


class StubClient:
    """A reranker that reorders without a network, to bound C's risk.

    identity  agrees with B  -> C's realistic ceiling
    reverse   inverts B      -> C's floor, the cost of a confidently wrong model
    shuffle   ignores B      -> what a model that adds no signal actually costs
    """

    available = True

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls = 0

    def call_llm(self, system_prompt: str, user_prompt: str) -> LLMResult:
        self.calls += 1
        asins = list(dict.fromkeys(re.findall(r"\b(B0[A-Z0-9]{8})\b", user_prompt)))
        if self.mode == "reverse":
            asins = list(reversed(asins))
        elif self.mode == "shuffle":
            random.Random(0).shuffle(asins)
        payload = {"ranked": [{"parent_asin": a, "score": 1.0 - i * 0.01}
                              for i, a in enumerate(asins)],
                   "confident": True}
        return LLMResult(json.dumps(payload), 120, 40)


def score(label, agent, samples, ids, cats, prods, quiet=False, progress=False):
    started = time.time()
    if progress:
        agent.progress_total = len(samples)
        agent.sessions_done = 0
        agent._started = started
    result = evaluate(agent, samples, ids, cats, prods)
    if progress:
        print("\r" + " " * 60 + "\r", end="", file=sys.stderr, flush=True)
    reranker = agent.pipeline.reranker
    result["_meta"] = {
        "label": label,
        "eval_seconds": round(time.time() - started, 2),
        "rerank_stats": dict(reranker.stats) if reranker else None,
    }
    if not quiet:
        print("%-34s tech=%.6f hit=%.4f mrr=%.6f mttc=%.4f  (%.1fs)" % (
            label, result["recommended_technical_score"], result["hit_rate_at_10"],
            result["mrr"], result["mttc"], result["_meta"]["eval_seconds"]))
    return result


def report(result):
    meta = result["_meta"]
    delta = result["recommended_technical_score"] - RETRIEVAL_ONLY_REFERENCE
    print("\n== %s ==  eval %.1fs" % (meta["label"], meta["eval_seconds"]))
    print("%-18s %10s" % ("metric", "value"))
    print("%-18s %10.4f" % ("hit_rate@10", result["hit_rate_at_10"]))
    print("%-18s %10.6f" % ("mrr", result["mrr"]))
    print("%-18s %10.4f" % ("mttc", result["mttc"]))
    print("%-18s %10.4f" % ("efficiency", result["efficiency"]))
    print("%-18s %10.6f" % ("technical_score", result["recommended_technical_score"]))
    verdict = "OK" if delta >= -1e-6 else "REGRESSION -- the conversation layer is costing you"
    print("%-18s %+10.6f   %s" % ("vs retrieval-only", delta, verdict))

    tokens = result["reported_token_usage"]["total_tokens"]
    stats = meta["rerank_stats"]
    print("\nwhat actually ran:  tokens=%d" % tokens)
    if stats is None:
        print("  reranker OFF by default, matching the shipped agent.")
        print("  pass --rerank to exercise C, or --fake-llm to rehearse free.")
    else:
        print("  llm_ok=%d  no_key=%d  failed=%d  timed_out=%d" % (
            stats["llm_ok"], stats["no_key"], stats["failed"], stats["timed_out"]))
        if stats["llm_ok"] == 0 and stats["no_key"] > 0:
            print("  -> no key: C did not run. Scores are retrieval-only.")
        elif stats["failed"] or stats["timed_out"]:
            print("  -> C degraded on some turns; the fallback served B's order there.")

    print("\nby scenario:")
    for name, m in result["scenario_metrics"].items():
        print("  %-16s n=%-4d hit=%.3f mrr=%.3f mttc=%.2f" % (
            name, m["sample_count"], m["hit_rate_at_10"], m["mrr"], m["mttc"]))


KEYS = ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="ANNA_dump/results_integrated.json")
    parser.add_argument("--label", default="integrated")
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N sessions. A live --compare-key "
                             "over all 200 is ~281 API calls and 10-20 minutes; "
                             "--limit 40 gives the direction in about two.")
    parser.add_argument("--no-gate", action="store_true",
                        help="disable B's evidence gate (costs 0.039, measured)")
    parser.add_argument("--router-constraints", action="store_true",
                        help="feed A's parsed slots into phrase evidence (measured: 0.000000)")
    parser.add_argument("--clarify", action="store_true",
                        help="enable A's over-generality clarification (costs 0.005, measured)")
    parser.add_argument("--rerank", action="store_true",
                        help="turn Person C's LLM reranker ON. OFF by default, to "
                             "match the shipped agent -- with a key in .env this "
                             "makes ~281 live API calls and takes 10-20 minutes.")
    parser.add_argument("--no-rerank", action="store_true",
                        help="deprecated no-op; C is off by default now")
    parser.add_argument("--fake-llm", choices=("identity", "reverse", "shuffle"),
                        help="exercise C's path with a stand-in client, no network, no spend")
    parser.add_argument("--ablate", action="store_true", help="run the whole flag matrix")
    parser.add_argument("--compare-key", action="store_true",
                        help="score twice, with the key hidden and with it visible")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit:
        samples = samples[:args.limit]
        print("scoring the first %d of 200 sessions (--limit). Directional only: "
              "a subset moves the score by a few thousandths on its own." % len(samples))
    ids, cats, prods = catalog_index(args.catalog)

    def build(stub=True, **overrides):
        flags = {
            "enable_rerank": args.rerank and not args.no_rerank,
            "enable_clarification": args.clarify,
            "enable_evidence_gate": not args.no_gate,
            "use_router_constraints": args.router_constraints,
        }
        flags.update(overrides)
        if args.compare_key or args.fake_llm:
            flags["enable_rerank"] = True
        agent = IntegratedAgent(args.catalog, **flags)
        if stub and args.fake_llm and agent.pipeline.reranker is not None:
            agent.pipeline.reranker = Reranker(llm_client=StubClient(args.fake_llm))
        return agent

    started = time.time()
    build()   # first construction pays the index build; later ones are cached
    print("index ready in %.1fs (cached; later constructions are ~0s)\n" % (time.time() - started))

    if args.ablate:
        rows = [
            ("gate off, no hints (pre-fix)", dict(enable_evidence_gate=False, use_router_constraints=False)),
            ("gate on  <-- SHIPPED", dict(enable_evidence_gate=True, use_router_constraints=False)),
            ("gate on + A's slot hints", dict(enable_evidence_gate=True, use_router_constraints=True)),
            ("gate on + clarification", dict(enable_evidence_gate=True, enable_clarification=True)),
            ("gate on, C removed", dict(enable_evidence_gate=True, enable_rerank=False)),
        ]
        results = {label: score(label, build(**flags), samples, ids, cats, prods)
                   for label, flags in rows}
        Path(args.output).write_text(json.dumps(
            {k: {m: v[m] for m in KEYS} for k, v in results.items()}, indent=2) + "\n",
            encoding="utf-8")
        print("\nwritten to %s" % args.output)
        return

    if args.compare_key:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key and not args.fake_llm:
            print("No ANTHROPIC_API_KEY set, so both arms would be identical.")
            print("Export the key first, or use --fake-llm to rehearse without spending.")
            return
        os.environ.pop("ANTHROPIC_API_KEY", None)
        # The control arm must be genuinely C-less: no key AND no stand-in,
        # otherwise --fake-llm would rehearse both arms identically and the
        # delta would always read zero.
        without = score("C off  (key hidden)", build(stub=False), samples, ids, cats, prods)
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
        if key and not args.fake_llm:
            print("  (live: ~%d API calls, sequential -- allow 10-20 min for 200 "
                  "sessions)" % (len(samples) * 1.4))
        with_key = score("C on   (key visible)", build(), samples, ids, cats, prods,
                         progress=bool(key) and not args.fake_llm)
        delta = with_key["recommended_technical_score"] - without["recommended_technical_score"]
        stats = with_key["_meta"]["rerank_stats"]
        print("\nC is worth %+.6f technical (%d tokens, %d successful calls, "
              "%d fallbacks, %d timeouts)" % (
                  delta, with_key["reported_token_usage"]["total_tokens"],
                  stats["llm_ok"], stats["failed"], stats["timed_out"]))
        # A run where C never actually executed produces delta 0.000000, which
        # reads exactly like "C is neutral". It is not the same claim at all,
        # so say which one happened.
        if stats["llm_ok"] == 0:
            print("C DID NOT RUN -- %d fallbacks, %d timeouts, 0 successful calls."
                  % (stats["failed"], stats["timed_out"]))
            print("  The delta above is meaningless. Run ANNA_dump/check_key.py "
                  "before reading anything into it.")
        elif stats["failed"] or stats["timed_out"]:
            print("PARTIAL: C ran on %d turns, fell back on %d, timed out on %d."
                  % (stats["llm_ok"], stats["failed"], stats["timed_out"]))
            # A failed turn serves B's ordering, so the failures pull the C-on
            # arm TOWARDS the C-off arm from whichever side it sits on. Which
            # way that biases the delta depends on its sign, and getting this
            # backwards turns "C is hurting you" into "needs more data".
            if delta >= 0:
                print("  Those turns served B's order, so the delta UNDERSTATES C's")
                print("  benefit. Fix the failures and the gain should grow.")
            else:
                print("  Those turns served B's order -- which is the BETTER order --")
                print("  so the delta FLATTERS C. Its true cost is larger than shown.")
                print("  Fixing the timeouts would make this worse, not better.")
        elif delta > 0.002:
            print("ship C: +%.6f over retrieval-only, on %d clean calls."
                  % (delta, stats["llm_ok"]))
        else:
            print("do NOT ship C: it is not paying for itself on this set.")
        Path(args.output).write_text(json.dumps({
            "without_key": {m: without[m] for m in KEYS},
            "with_key": {m: with_key[m] for m in KEYS},
            "delta_technical": round(delta, 6),
        }, indent=2) + "\n", encoding="utf-8")
        print("written to %s" % args.output)
        return

    result = score(args.label, build(), samples, ids, cats, prods, quiet=True)
    report(result)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nwritten to %s" % args.output)


if __name__ == "__main__":
    main()
