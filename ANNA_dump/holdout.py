"""Overfitting check: how much of 0.9092 is fitted to the public 200?

Every weight, threshold and switch in the engine was chosen by looking at
the same 200 public sessions, and the graded set is 800 private ones. This
splits the public set in half (stratified by scenario type so both folds
keep the 40/40/15/5 mix), tunes on one fold, and reports the score on the
other. The gap between in-sample best and held-out score is the honest
estimate of how much of our number is real.

Both directions are run, because a single split is itself a sample.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import local_evaluator as ev  # noqa: E402
from src.retrieval.catalog import Catalog  # noqa: E402
from src.retrieval.engine import SearchEngine  # noqa: E402
from src.retrieval.text import tokenise  # noqa: E402


class HarnessAgent:
    def __init__(self, engine, words, cap):
        self.engine, self.words, self.cap = engine, words, cap

    def reset(self, session_id, user_profile):
        self.engine.start_session(session_id)

    def respond(self, session_id, user_message, turn, top_k):
        self.engine.observe(session_id, user_message)
        ranked = self.engine.search(session_id, top_k)
        if turn <= self.cap and self.engine.evidence_words(session_id) < self.words:
            ranked = []
        return {"message": "", "ask_attribute": "other",
                "recommendations": [{"parent_asin": a} for a, _ in ranked],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def stratified_halves(samples):
    """Split preserving the scenario mix, deterministically."""
    buckets = defaultdict(list)
    for sample in samples:
        buckets[sample["scenario_type"]].append(sample)
    fold_a, fold_b = [], []
    for scenario in sorted(buckets):
        for index, sample in enumerate(sorted(buckets[scenario], key=lambda s: s["sample_id"])):
            (fold_a if index % 2 == 0 else fold_b).append(sample)
    return fold_a, fold_b


# The knobs that were actually tuned by looking at the public set.
GRID = [
    (phrase, category, words, cap)
    for phrase in (0.4, 1.0, 2.0)
    for category in (0.2, 0.4, 0.7)
    for words in (2, 4, 6)
    for cap in (2, 3)
]
SHIPPED = (1.0, 0.4, 4, 3)


def main() -> None:
    catalog = Catalog.load("data/catalog.jsonl")
    engine = SearchEngine(catalog=catalog)
    samples = ev.load_jsonl("data/public_set.jsonl")
    ids, cats, prods = ev.catalog_index("data/catalog.jsonl")
    fold_a, fold_b = stratified_halves(samples)
    print(f"fold A n={len(fold_a)}  fold B n={len(fold_b)}  grid={len(GRID)} configs\n")

    def run(config, fold):
        phrase, category, words, cap = config
        engine.weights["phrase"] = phrase
        engine.weights["category"] = category
        engine._states.clear()
        r = ev.evaluate(HarnessAgent(engine, words, cap), fold, ids, cats, prods)
        return r["recommended_technical_score"]

    scores = {}
    started = time.time()
    for index, config in enumerate(GRID, 1):
        scores[config] = (run(config, fold_a), run(config, fold_b))
        elapsed = time.time() - started
        print("\r  config %d/%d  %.0fs elapsed, ~%.0fs left   " % (
            index, len(GRID), elapsed, elapsed / index * (len(GRID) - index)),
            end="", file=sys.stderr, flush=True)
    print("\r" + " " * 60 + "\r", end="", file=sys.stderr, flush=True)

    print(f"{'':<26}{'tuned on A':>12}{'held out B':>12}{'gap':>9}")
    best_a = max(GRID, key=lambda c: scores[c][0])
    best_b = max(GRID, key=lambda c: scores[c][1])
    print(f"{'best config on A':<26}{scores[best_a][0]:>12.4f}{scores[best_a][1]:>12.4f}"
          f"{scores[best_a][1]-scores[best_a][0]:>9.4f}   {best_a}")
    print(f"{'best config on B':<26}{scores[best_b][1]:>12.4f}{scores[best_b][0]:>12.4f}"
          f"{scores[best_b][0]-scores[best_b][1]:>9.4f}   {best_b}")
    print(f"{'SHIPPED config':<26}{scores[SHIPPED][0]:>12.4f}{scores[SHIPPED][1]:>12.4f}"
          f"{scores[SHIPPED][1]-scores[SHIPPED][0]:>9.4f}   {SHIPPED}")
    print()
    opt_a = scores[best_a][0] - scores[SHIPPED][0]
    opt_b = scores[best_b][1] - scores[SHIPPED][1]
    print(f"optimism (best-on-fold minus shipped, same fold): A {opt_a:+.4f}   B {opt_b:+.4f}")
    print(f"shipped config rank on A: {sorted(GRID, key=lambda c: -scores[c][0]).index(SHIPPED)+1}/{len(GRID)}")
    print(f"shipped config rank on B: {sorted(GRID, key=lambda c: -scores[c][1]).index(SHIPPED)+1}/{len(GRID)}")
    print(f"\nspread across all {len(GRID)} configs:")
    all_a = [scores[c][0] for c in GRID]; all_b = [scores[c][1] for c in GRID]
    print(f"  fold A  min {min(all_a):.4f}  max {max(all_a):.4f}  range {max(all_a)-min(all_a):.4f}")
    print(f"  fold B  min {min(all_b):.4f}  max {max(all_b):.4f}  range {max(all_b)-min(all_b):.4f}")


if __name__ == "__main__":
    main()
