# Demo narration — ~3 min 30 s

Run this in one window, full screen, font size up:

    python ANNA_dump/demo_session.py --wait

**Use `--wait`, not `--slow`.** It holds at four points and waits for you to
press ENTER, so you set the pace and the script can never drift out of sync
with the screen. A fixed timer cannot work here: a three-turn session has only
two inter-turn gaps, against roughly three minutes of narration.

The four hold points line up with the four sections below:

    [ ENTER to begin the conversation ]   after the header      -> read [0:25]
    [ press ENTER for the next turn ]     after turn 1          -> read [0:45]
    [ press ENTER for the next turn ]     after turn 2          -> read [1:30]
    [ ENTER for the result ]              before the hit        -> read [2:00]

Read the block, press ENTER, read the next. Sections [2:30] and [3:15] come
after the session ends, so take as long as you like there.

Have the index already built (run it once before recording) so there is no dead
air, and turn wifi off before you start — you will refer to it.

---

## [0:00] Before you press enter

> "This is Megabyte's shopping copilot. It finds one hidden product in a frozen
> fifty-thousand-item Amazon catalog, by having a conversation. Technical score
> 0.909, against a starter baseline of 0.107.
>
> One thing before I run it — my wifi is off. There's no model in this, no API
> key, no network. It's numpy, and it answers in under two milliseconds a turn.
> I'll come back to why that was a deliberate choice."

Press enter.

## [0:25] The header

> "The script is showing me the hidden target so you can follow along — a
> polyester tunic top. The agent never sees any of that. It only sees what the
> customer types, one turn at a time. And the customer here isn't me, it's the
> organiser's own simulator, running the same disclosure rules the graded run
> uses."

## [0:45] TURN 1 — the important beat

Say this *before* the empty list lands, or it reads as a bug.

> "Watch turn one carefully. The customer says something vague — tees and
> blouses. And the agent returns **nothing**. No recommendations at all.
>
> That's deliberate, and it's the single biggest thing we built. The customer
> has given fewer than four informative words, so instead of guessing, the agent
> asks a question. Guessing here typically lands the target around rank two,
> which ends the session at a reciprocal rank of a half. Asking one more
> question ends it at one.
>
> That trade — spend a turn, gain a rank — is worth 0.039 on the technical
> score. We measured it both ways: 0.870 without, 0.909 with, and hit rate
> identical at 0.985. The gate switches itself off after turn three, so it can
> never stall a session."

## [1:30] TURN 2 — retrieval working

> "Now the customer discloses the material, and the target comes straight in at
> rank one. That's category scoping doing the work: naming the coarse category
> cuts fifty thousand products down to a median of a hundred and ninety-four,
> and we recover the target in two hundred out of two hundred sessions. On top
> of that a hand-rolled BM25 in numpy, fused with reciprocal rank fusion."

Point at the timing on screen.

> "Three milliseconds."

## [2:00] TURN 3 — the override

> "Here's the hard scenario. The customer reverses themselves — 'actually,
> ignore what I said'. Fifteen percent of graded sessions do this, and the
> session can't convert until the new intent is served.
>
> We measured what to do with the revoked preference, and the answer was
> counterintuitive: **keeping** it scores 0.853 on override MRR, deleting it
> scores 0.305. So we keep it. Target holds at rank one, hit on turn three,
> reciprocal rank 1.0."

## [2:30] What we removed — your strongest section

> "The part I actually want to show you is what we took *out*.
>
> We built dense embeddings. Swept six weights. Every one was worse — 0.909 down
> to 0.828, at forty times the latency.
>
> We wired the intent router's parsed constraints into retrieval. Bit-identical.
> Zero.
>
> We built an LLM reranker and ran it live. Minus 0.060, eight seconds a
> session, a quarter of the calls timing out.
>
> We reweighted the index fields. Flat.
>
> Four honest negatives — and they have one cause. The simulator builds its
> constraints from the product's own features field. We proved it by ablation:
> take the features field out of the index and the score collapses by 0.44,
> where taking the description out costs 0.008. So lexical overlap isn't a proxy
> for relevance here — it's very nearly ground truth, and every semantic layer
> we added diluted it.
>
> On the reranker specifically: it can only reorder inside the top ten, so it
> can't change hit rate at all — only MRR. Seventy-five percent of our targets
> are already at rank one. A *perfect* oracle is worth plus 0.045. The floor is
> minus 0.25. We didn't like that bet, so it ships disabled with a tested
> fallback."

## [3:15] Close

> "So: 0.909 technical. Not overfitted — tuned on one half of the public set, it
> ranks tenth of fifty-four configurations on the held-out half. Robust to
> rephrasing — 0.88 when you swap the wording out entirely. Twenty-one
> regression tests. And it runs offline in one second, on numpy, for nothing.
>
> Our honest estimate on the private set is 0.88 to 0.92."

---

## If you have 60 seconds, not 210

Run `--wait` and say only:

1. Wifi is off. No model, no key, 1.8 ms a turn. Score 0.909 from a 0.107 baseline.
2. Turn one returns nothing **on purpose** — it asks instead of guessing. Worth +0.039, measured.
3. Turn three, the customer reverses; target holds at rank one.
4. We also built embeddings, an LLM reranker and router integration. All three measured worse. They ship disabled, and the report says exactly why.

---

## Notes

- Don't read the metric tables aloud. Put them on screen, talk over them.
- Don't apologise for the templated `message` text. If asked, it's listed in
  the report's limitations; the contract field that the simulator actually reads
  is `ask_attribute`, and that one is chosen from conversation state.
- Rehearse the run at least twice. It is deterministic — same session, same
  turns, same rank — so what you rehearse is what you film.
- You can read this start to finish without looking up, as long as you press
  ENTER at the end of each marked block. The screen waits for you, not the
  other way round.
- Other sessions: `--scenario buying --index 0` hits at turn 2 rank 1, and
  `--scenario boundary --index 1` shows the "no preference" path. `--list` shows
  everything.
