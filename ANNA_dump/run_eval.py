"""Run the public-set evaluator against the B-side engine and diff vs baseline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from ANNA_dump.eval_agent import Agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="ANNA_dump/results.json")
    parser.add_argument("--label", default="run")
    parser.add_argument("--drop-overridden", action="store_true")
    parser.add_argument("--ask-attribute", default="other")
    parser.add_argument("--phrase-weight", type=float, default=1.6)
    parser.add_argument("--no-category", action="store_true")
    parser.add_argument("--no-fusion", action="store_true", help="use the pre-B3 additive scorer")
    parser.add_argument("--dense", action="store_true", help="enable the dense/FAISS ranker")
    parser.add_argument("--weights", default="", help='e.g. "dense=1.5,category=0.2"')
    args = parser.parse_args()

    started = time.time()
    weights = {}
    for pair in filter(None, args.weights.split(",")):
        name, _, value = pair.partition("=")
        weights[name.strip()] = float(value)
    agent = Agent(
        args.catalog,
        drop_overridden=args.drop_overridden,
        phrase_weight=args.phrase_weight,
        use_fusion=not args.no_fusion,
        weights=weights or None,
        enable_dense=args.dense,
    )
    build_seconds = time.time() - started

    import ANNA_dump.eval_agent as mod
    mod.Agent.ASK = None if args.ask_attribute == "none" else args.ask_attribute
    if args.no_category:
        # Ablation: disable category scoping by never recognising a label,
        # which drops the engine back to whole-catalog ranking.
        from src.retrieval.category import CategoryIndex
        CategoryIndex.parse_label = staticmethod(lambda message: None)

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    started = time.time()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    eval_seconds = time.time() - started

    baseline = json.loads(Path("docs/baseline_results.json").read_text(encoding="utf-8"))
    result["_meta"] = {
        "label": args.label,
        "index_build_seconds": round(build_seconds, 2),
        "eval_seconds": round(eval_seconds, 2),
        "drop_overridden": args.drop_overridden,
        "phrase_weight": args.phrase_weight,
        "fusion": not args.no_fusion,
        "dense": args.dense,
        "weights": weights,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def row(name: str, key: str, base_key: str | None = None) -> str:
        ours = result[key]
        base = baseline[base_key or key]
        delta = ours - base
        return f"{name:<18} {base:>9.4f} -> {ours:>9.4f}   {delta:+9.4f}"

    print(f"\n== {args.label} ==  index {build_seconds:.1f}s | eval {eval_seconds:.1f}s")
    print(f"{'metric':<18} {'baseline':>9}    {'ours':>9}   {'delta':>9}")
    print(row("hit_rate@10", "hit_rate_at_10"))
    print(row("mrr", "mrr"))
    print(row("mttc", "mttc"))
    print(row("efficiency", "efficiency"))
    print(row("technical_score", "recommended_technical_score", "technical_score"))
    print("\nby scenario:")
    for name, metrics in result["scenario_metrics"].items():
        print(f"  {name:<16} n={metrics['sample_count']:<4} hit={metrics['hit_rate_at_10']:.3f} "
              f"mrr={metrics['mrr']:.3f} mttc={metrics['mttc']:.2f}")


if __name__ == "__main__":
    main()
