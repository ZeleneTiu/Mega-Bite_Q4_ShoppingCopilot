"""Robustness harness: does the engine survive differently-worded customers?

The public simulator speaks in fixed templates ("I'm looking for X. A key
requirement is: Y."). The organiser holds 800 private sessions we cannot
inspect. If those are phrased differently, any parser tuned to the public
templates degrades silently, and the public score becomes a lie.

This harness swaps the simulator's phrasing while preserving exactly the
same information content and disclosure logic, so any score drop is
attributable to parsing brittleness and nothing else.

Styles, roughly in order of how hostile they are:
  clean    the organiser's own templates, as a control
  natural  conversational rewording, markers still present but different
  terse    markers stripped entirely, constraints stated bare
"""
from __future__ import annotations

import argparse
import json
import random
import zlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import local_evaluator as ev  # noqa: E402
from ANNA_dump.eval_agent import Agent  # noqa: E402

ALLOWED = ev.ALLOWED_ATTRIBUTES


def _matches(sample: dict, attribute: str, disclosed: set[str]) -> list[str]:
    """Same disclosure rule as the organiser's customer_reply."""
    constraints = [
        *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
        *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
    ]
    found = [
        v for v in constraints
        if v not in disclosed
        and (attribute == "other" or ev.classify_constraint(v) == attribute)
    ][:2]
    disclosed.update(found)
    return found


def make_natural():
    def initial_message(sample, category, disclosed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            c = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(c)
            return f"Hey, I'm after {category}. It really needs to be {c}."
        if scenario == "intent_override":
            return f"Hey, I'm after {category}. {sample['behavior']['override']['old_value']}"
        return f"Just browsing {category} at the moment, nothing specific in mind."

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        attr = ask_attribute if isinstance(ask_attribute, str) else None
        if sample["scenario_type"] == "boundary" and not boundary_used and attr:
            return f"No strong feelings on {attr}, your call there.", True
        if not attr:
            return "Not quite there yet. Ask me about something specific.", boundary_used
        if attr not in ALLOWED:
            attr = "other"
        found = _matches(sample, attr, disclosed)
        if not found:
            return f"Nothing else to add on {attr}.", boundary_used
        return "The things that matter to me are " + " and ".join(found) + ".", boundary_used

    def override_message(new_value: str) -> str:
        return f"Scratch that, forget what I said before. Really I want {new_value}."

    return initial_message, customer_reply, override_message


def make_terse():
    def initial_message(sample, category, disclosed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            c = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(c)
            return f"{category}. Must have {c}."
        if scenario == "intent_override":
            return f"{category}. {sample['behavior']['override']['old_value']}"
        return f"{category}, just browsing."

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        attr = ask_attribute if isinstance(ask_attribute, str) else None
        if sample["scenario_type"] == "boundary" and not boundary_used and attr:
            return f"{attr} doesn't matter to me.", True
        if not attr:
            return "Nope. Ask me something specific.", boundary_used
        if attr not in ALLOWED:
            attr = "other"
        found = _matches(sample, attr, disclosed)
        if not found:
            return f"Nothing more on {attr}.", boundary_used
        # Bare constraints, no marker phrase at all. The hardest case.
        return "; ".join(found) + ".", boundary_used

    def override_message(new_value: str) -> str:
        return f"{new_value}."

    return initial_message, customer_reply, override_message


# ---------------------------------------------------------------------------
# The three styles above reword the WRAPPER but quote each constraint
# verbatim, because that is what the organiser's own simulator does. They
# therefore prove the parser is not template-bound -- and prove nothing about
# what happens if the organiser paraphrases the constraint TEXT.
#
# That distinction matters here more than anywhere else: the intent card is
# built from the product's own `features` and `details`
# (local_evaluator.py:57), and removing `features` from the index costs 0.441
# technical. The whole score rests on lexical overlap with those fields, so
# the honest stress test is one that breaks the overlap itself.
#
# paraphrase()  swaps common descriptors for wordier equivalents and drops a
#               quarter of the remaining words, deterministically.
# hostile       applies that on top of the terse wrapper: no marker phrases,
#               no verbatim constraint, nothing to match on but leftovers.

_SYNONYMS = {
    "waterproof": "water resistant", "breathable": "lets air through",
    "lightweight": "light and easy to carry", "durable": "built to last",
    "adjustable": "can be adjusted", "comfortable": "comfy",
    "hypoallergenic": "kind to sensitive skin", "elastic": "stretchy",
    "zipper": "zip fastening", "pockets": "pouches", "hood": "hooded part",
    "sleeve": "arm section", "stainless": "rust free", "sterling": "solid",
    "genuine": "real", "machine": "in the washer", "washable": "can be washed",
    "insulated": "keeps warmth in", "cushioned": "padded underfoot",
    "leather": "hide material", "cotton": "soft woven fabric",
}


def paraphrase(text: str, seed: int, drop: float = 0.25) -> str:
    """Reword a constraint so its exact wording no longer matches the catalog."""
    rng = random.Random(seed)
    words = str(text).split()
    out: list[str] = []
    for word in words:
        key = word.lower().strip(".,;:()")
        if key in _SYNONYMS:
            out.extend(_SYNONYMS[key].split())
        elif rng.random() >= drop:
            out.append(word)
    return " ".join(out) or str(text)


def make_paraphrase(drop=0.25, terse=False):
    base_initial, base_reply, base_override = (
        make_terse() if terse else make_natural()
    )

    def initial_message(sample, category, disclosed):
        text = base_initial(sample, category, disclosed)
        # zlib.crc32, not hash(): Python randomises str hashing per process, so
        # hash() made these styles produce a different score on every run. A
        # robustness harness whose numbers move on their own is worse than none.
        seed = zlib.crc32(str(sample["sample_id"]).encode()) & 0xFFFF
        return paraphrase(text, seed=seed, drop=drop)

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        text, boundary_used = base_reply(sample, ask_attribute, disclosed, boundary_used)
        return paraphrase(text, seed=len(disclosed) * 7 + 3, drop=drop), boundary_used

    return initial_message, customer_reply, base_override


STYLES = {
    "clean": None,
    "natural": make_natural,
    "terse": make_terse,
    "paraphrase": lambda: make_paraphrase(drop=0.25, terse=False),
    "hostile": lambda: make_paraphrase(drop=0.40, terse=True),
}


def apply_style(name: str) -> None:
    """Monkeypatch the simulator's phrasing in place."""
    factory = STYLES[name]
    if factory is None:
        return
    initial_message, customer_reply, override_message = factory()
    ev.initial_message = initial_message
    ev.customer_reply = customer_reply

    # The override turn's wording lives inside behavior_for, so wrap it.
    original_behavior_for = ev.behavior_for

    def patched_behavior_for(scenario, card, rng):
        behavior = original_behavior_for(scenario, card, rng)
        if "override" in behavior:
            behavior["override"]["message"] = override_message(
                str(behavior["override"].get("new_value", ""))
            )
        return behavior

    ev.behavior_for = patched_behavior_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles", default="clean,natural,terse,paraphrase,hostile")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--integrated", action="store_true",
        help="Perturb the SHIPPED pipeline (starter.agent.Agent) instead of the "
             "retrieval-only harness agent. The private set is scored against "
             "the shipped agent, so this is the run that actually matters; the "
             "retrieval-only default stays for attributing a drop to B alone.",
    )
    parser.add_argument("--output", default="ANNA_dump/robustness.json")
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    ids, cats, prods = ev.catalog_index(args.catalog)
    if args.integrated:
        from starter.agent import Agent as IntegratedAgent  # noqa: PLC0415
        agent = IntegratedAgent(args.catalog)
        clear_states = agent.pipeline.engine._states.clear
    else:
        agent = Agent(args.catalog)
        clear_states = agent.engine._states.clear

    print(f"{'style':<10} {'hit@10':>8} {'mrr':>8} {'mttc':>7} {'technical':>10}")
    results = {}
    for style in args.styles.split(","):
        style = style.strip()
        apply_style(style)
        clear_states()
        started = time.time()
        r = ev.evaluate(agent, samples, ids, cats, prods)
        results[style] = {k: r[k] for k in
                          ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score")}
        print(f"{style:<10} {r['hit_rate_at_10']:>8.4f} {r['mrr']:>8.4f} "
              f"{r['mttc']:>7.2f} {r['recommended_technical_score']:>10.4f}"
              f"   ({time.time()-started:.1f}s)", flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
