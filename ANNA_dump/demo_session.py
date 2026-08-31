"""Play one real evaluation session out loud, turn by turn.

Final Deliverables asks for "one demonstrated multi-turn session". The
evaluator proves multi-turn behaviour statistically -- MTTC 2.71 means the
agent converts in 2.71 turns on average -- but it prints aggregate metrics,
not a conversation. This prints the conversation.

Nothing here is scripted. The customer's words come from the organiser's own
simulator (evaluator.local_evaluator), driven by the hidden intent card, with
the same disclosure policy and the same override timing the graded run uses.
The agent is starter.agent.Agent exactly as submitted. The only thing added is
the printing.

  python ANNA_dump/demo_session.py                    a good intent-override session
  python ANNA_dump/demo_session.py --scenario buying  pick the scenario
  python ANNA_dump/demo_session.py --slow 3           pause 3s per turn, to narrate over
  python ANNA_dump/demo_session.py --list             what is available
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS, TOP_K, catalog_index, coarse_category, customer_reply,
    initial_message, load_jsonl, materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

WIDTH = 78
SHOW_RECS = 5

# Default take for the camera: the evidence gate withholds on turn 1, the
# customer reverses on turn 3, and the target lands at rank 1. It is a
# representative session, not a flattering one -- 75% of all 200 sessions
# converge at rank 1. Use --list and --index to pick another.
DEFAULT_DEMO = "public_0052"


def rule(char="-"):
    print(char * WIDTH)


def wrap(text, indent):
    """Hard wrap so nothing runs off the edge of a screen recording."""
    words, lines, line = str(text).split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > WIDTH - indent:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    pad = " " * indent
    return ("\n" + pad).join(lines) if lines else ""


def pause(slow, wait, prompt="        [ press ENTER for the next turn ]"):
    """Hold the screen. --wait beats --slow for filming: a fixed timer desyncs
    from a spoken script within the first block, because a three-turn session
    only has two inter-turn gaps."""
    if wait:
        try:
            input(prompt)
        except EOFError:
            pass
    elif slow:
        time.sleep(slow)


def play(sample, agent, catalog_ids, categories, products, slow=0.0, wait=False):
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    session_id = "demo_" + str(sample["sample_id"])

    print()
    rule("=")
    print("  SESSION  %s" % sample["sample_id"])
    print("  scenario %s        difficulty %s"
          % (sample["scenario_type"], sample.get("difficulty_bucket", "-")))
    print("  hidden target      %s" % target)
    print("  %s" % wrap(str(products[target].get("title", ""))[:120], 2))
    print()
    print("  The agent is never told any of the above. It sees only the")
    print("  customer's words, one turn at a time.")
    rule("=")

    pause(slow, wait, "        [ ENTER to begin the conversation ]")

    agent.reset(session_id, sample["user_profile"])
    disclosed, boundary_used = set(), False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print()
        rule()
        print("  TURN %d" % turn)
        rule()
        print("  customer   %s" % wrap(message, 13))

        started = time.perf_counter()
        response = agent.respond(session_id, message, turn, TOP_K)
        latency_ms = (time.perf_counter() - started) * 1000

        print("  agent      %s" % wrap(response.get("message", ""), 13))
        print("             ask_attribute: %r          [%.1f ms]"
              % (response.get("ask_attribute"), latency_ms))

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print()
        if not ranked:
            print("             recommendations: NONE")
            print("             The evidence gate is withholding. Fewer than four")
            print("             informative words so far, so the agent asks instead")
            print("             of guessing. Answering here tends to land the target")
            print("             around rank 2 and end the session at RR 0.5; asking")
            print("             once more ends it at 1.0. Worth +0.039 technical.")
        else:
            for position, asin in enumerate(ranked[:SHOW_RECS], 1):
                title = str(products.get(asin, {}).get("title", ""))[:58]
                mark = "  <== TARGET" if asin == target else ""
                print("             %2d. %-12s %-58s%s" % (position, asin, title, mark))
            if len(ranked) > SHOW_RECS:
                print("             ... %d more (%d returned, top 10 are scored)"
                      % (len(ranked) - SHOW_RECS, len(ranked)))

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            pause(slow, wait, "        [ ENTER for the result ]")
            print()
            rule("=")
            print("  HIT on turn %d at rank %d.  reciprocal rank %.3f" % (turn, rank, 1.0 / rank))
            print("  Session over: the evaluator stops the moment the target appears.")
            rule("=")
            return

        if turn == MAX_TURNS:
            break

        pause(slow, wait)

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, ignore my earlier preference."))
            print()
            print("  >>> INTENT OVERRIDE next turn: the customer revokes what they")
            print("      said earlier. Everything before this is now stale, and the")
            print("      session cannot convert until the new intent is served.")
        else:
            message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print()
    rule("=")
    print("  No hit within %d turns." % MAX_TURNS)
    rule("=")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--scenario", default="intent_override",
                        choices=("buying", "browsing", "intent_override", "boundary"))
    parser.add_argument("--sample-id", default=None,
                        help="play one exact session (default: %s)" % DEFAULT_DEMO)
    parser.add_argument("--index", type=int, default=0, help="nth session of that scenario")
    parser.add_argument("--slow", type=float, default=0.0,
                        help="fixed seconds between turns; fine for a quick look")
    parser.add_argument("--wait", action="store_true",
                        help="hold at every beat until you press ENTER. Use this for "
                             "filming: you control the pace, so a spoken script cannot "
                             "drift out of sync with the screen.")
    parser.add_argument("--list", action="store_true", help="list available sessions")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.list:
        for scenario in ("buying", "browsing", "intent_override", "boundary"):
            matching = [s for s in samples if s["scenario_type"] == scenario]
            print("%-16s %3d sessions   first ids: %s"
                  % (scenario, len(matching),
                     ", ".join(str(s["sample_id"]) for s in matching[:4])))
        return

    wants_default = (args.sample_id is None and args.index == 0
                     and args.scenario == "intent_override")
    if wants_default:
        args.sample_id = DEFAULT_DEMO

    if args.sample_id:
        chosen = [s for s in samples if str(s["sample_id"]) == str(args.sample_id)]
        if not chosen:
            print("no session with sample_id %r" % args.sample_id)
            return
        sample = chosen[0]
    else:
        matching = [s for s in samples if s["scenario_type"] == args.scenario]
        sample = matching[args.index % len(matching)]

    print("building the index (once, ~7s) ...", flush=True)
    started = time.time()
    agent = Agent(args.catalog)
    catalog_ids, categories, products = catalog_index(args.catalog)
    print("ready in %.1fs. No network, no API key, no model." % (time.time() - started))

    play(sample, agent, catalog_ids, categories, products, slow=args.slow, wait=args.wait)


if __name__ == "__main__":
    main()
