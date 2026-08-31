# Megabyte — Q4 Conversational Shopping Copilot

**Technical score 0.909154** on the 200-session public set, against a starter
baseline of 0.1067. The submitted agent uses **no model, no network and no
credentials**: numpy only, 1.8 ms per turn, zero tokens, zero cost.

| metric | value |
|---|---|
| HitRate@10 | 0.9850 |
| MRR | 0.836181 |
| MTTC | 2.7100 |
| Efficiency | 0.8290 |
| **TechnicalScore** | **0.909154** |

| scenario | n | hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.9875 | 0.8271 | 2.28 |
| browsing | 80 | 0.9875 | 0.8248 | 2.61 |
| intent override | 30 | 0.9667 | 0.8528 | 3.83 |
| boundary | 10 | 1.0000 | 0.9500 | 3.60 |

---

## 1. Architecture

One turn flows: **intent (A) → state update (A) → retrieve (B) → evidence gate
(B) → rerank (C, disabled) → contract response (C)**, orchestrated by
`src/pipeline.py` (D).

- `src/intent/` — intent routing and conversation state, slot accumulation,
  intent-override detection.
- `src/retrieval/` — the score engine. `catalog.py` parsing, `category.py`
  scoping, `lexical.py` a hand-rolled numpy BM25, `dense.py` optional
  embeddings (ships at weight 0.0), `fusion.py` reciprocal rank fusion,
  `engine.py` ranking, turn state and the evidence gate.
- `src/rerank/` — LLM reranking and contract validation. Reranking is
  **disabled in the submitted agent** (§4).
- `src/memory/` — per-session memory and context distillation.
- `src/safety.py` — contract validation, response coercion, resilience wrapper.
- `starter/agent.py` — the submitted `Agent`.

### What makes the score

**Category scoping is the engine.** The simulator names the target's own coarse
category. Matching that against the closed set of 1,115 catalog labels cuts
50,000 candidates to a median of 194, with 200/200 recall.

**The evidence gate is the second lever.** When the customer has given fewer
than four distinct informative words and it is still turn 1–3, the agent
returns no recommendations and asks a question instead. Answering a vague turn
tends to place the target near rank 2, ending the session at reciprocal rank
0.5; asking once more ends it at 1.0. Efficiency carries weight 0.20 against
MRR's 0.30, so the extra turn pays.

    gate off   0.870196   MRR 0.672653   MTTC 2.21
    gate on    0.909154   MRR 0.836181   MTTC 2.71   +0.0390

Hit rate is unchanged at 0.985 either way, and the gate self-disables past turn
3 so hit rate — weight 0.50 — is never at risk.

**The clarification policy is the largest single lever.** Which attribute the
agent asks for decides what the simulator discloses next:

    ask_attribute "other"      0.909154
    ask_attribute "feature"    0.765357   -0.144
    ask_attribute null         0.326283   -0.583

---

## 2. Models, cost, latency and token usage

**The submitted agent uses no model.** Per the specification, `usage` is
optional when no model is used; the agent reports zeros.

| | |
|---|---|
| Model | none in the submitted path |
| Dependencies | `numpy` only (verified: the graded agent imports nothing else) |
| Network | none required; run verified with networking disabled |
| Credentials | none |
| Index build | 6.61 s, once, cached across `Agent()` constructions |
| Full public run | 1.03 s for 200 sessions / 539 turns |
| Latency per turn | mean 1.81 ms, p50 1.35 ms, p95 4.82 ms, max 9.11 ms |
| Tokens | 0 prompt, 0 completion |
| Estimated cost | $0.00 |

**Prototyped and rejected:** `claude-sonnet` for candidate reranking. Measured
live on 40 public sessions: −0.060 technical, −0.200 MRR, 8.3 s per session,
1,311 tokens per call, 25 % timeout rate against an 8 s deadline. See §4.

**Fallback behaviour.** With no key the reranker short-circuits before building
a prompt. Any failure — no key, transport error, unparseable reply, or a call
exceeding its wall-clock deadline — returns the retrieval ordering and its
scores unchanged, so a degraded run is indistinguishable from never having
asked. `starter/agent.py` additionally coerces every response to the contract
and cannot raise. Covered by `tests/test_offline_safety.py` and
`tests/test_integration_flow.py` (21 tests).

---

## 3. Results

    starter baseline                    0.1067
    retrieval engine alone              0.909154
    merged pipeline, gate off           0.870196
    SUBMITTED                           0.909154

### Robustness to rephrasing

`ANNA_dump/perturb_eval.py --integrated` rewords the simulator while preserving
information content and disclosure logic.

| style | technical |
|---|---|
| clean (organiser's templates) | 0.9092 |
| natural (conversational rewording) | 0.9045 |
| terse (markers stripped) | 0.9096 |
| synonyms substituted, meaning preserved | 0.8806 |

Meaning-preserving paraphrase costs 0.029. The harsher styles in that harness
also delete a quarter to two-fifths of the customer's words, which is
information loss rather than rephrasing, and costs considerably more.

### Overfitting

`ANNA_dump/holdout.py` tunes on one stratified half and tests on the other
across a 54-config grid, including the evidence gate's own parameters.

    shipped config, tuned on A   0.8923      held out B   0.9260
    tuning optimism              A +0.0043   B +0.0121
    shipped config rank          10/54 (A)   12/54 (B)
    grid spread                  0.0498 (A)  0.0402 (B)

The shipped configuration is near-optimal on both halves without being the
argmax of either. Honest private-set estimate: **0.88–0.92**.

---

## 4. What did not work

Four attempts to add signal, each measured, each neutral or negative.

| change | result |
|---|---|
| Dense embeddings (six weights swept) | 0.9092 → 0.8281, ~40× latency |
| Router slots into phrase evidence | 0.909154 → 0.909154 (bit-identical) |
| Router slots into the BM25 query | 0.909154 → 0.905067 |
| LLM reranking, live | 0.9268 → 0.8668 on 40 sessions |
| Field reweighting (5 configs) | ±0.000 |
| Proactive clarification gate | 0.909154 → 0.904354 |

**They share one cause.** The evaluator builds each intent card from the
product's own `features` and `details` fields (`local_evaluator.py:57`).
Removing a field from the index confirms the dependency:

    features      -0.4414      (primary constraint source)
    details       -0.0221      (secondary source)
    description   -0.0077      (not a source)
    title         -0.0018      (fallback label only)
    store         -0.0026      (not a source)

Because the customer quotes the product's own text, lexical overlap with those
fields is very nearly ground truth rather than a proxy for it. Every semantic
layer reasons about *meaning* and in doing so dilutes an *identity* match.

**On reranking specifically.** A reranker only reorders inside the retriever's
existing top-10, so it cannot change hit rate or MTTC — only MRR. With 75 % of
targets already at rank 1, a *perfect* oracle is worth +0.045 while the floor is
−0.251. The measured layer delivered −0.060. It ships disabled.

---

## 5. Limitations

1. **The score rests on `features` overlap.** Performance would degrade under
   paraphrasing more aggressive than the four styles tested.
2. **Confidence is not separable from score.** top-1/top-2 separation is 1.04×;
   a score threshold answers at 24 % precision against a 15.5 % base rate. Only
   evidence-word count and pool size separate, which is what the gate uses.
3. **The `user_profile` carries no measurable signal.** Every subgroup lands on
   the same target rating. It can serve explanations; it cannot move the score.
4. **Customer-facing prose is templated.** `ask_attribute` is chosen from
   conversation state, but the accompanying `message` on an answering turn is
   fixed text. The contract is met; the prose is not generative.
5. **Budget handling is deliberately absent.** Budget appears in 0 of 200 public
   sessions, and a naive price parser misfires on 174 of 1,030 turns (fabric
   percentages, model numbers). Judged not worth the false-positive risk.
6. **The reranker's candidate projection is incomplete.** It requests a
   `details` field the enrichment step does not supply. Since the layer ships
   disabled this does not affect the submitted score, but any future evaluation
   of it must fix this first.

---

## 6. Reproduction

    python --version                 # 3.10+
    pip install -r requirements.txt  # numpy only

Regenerate the catalog from the repo root (gitignored, must be 50,000 lines):

    python -c "import gzip,shutil,os; os.makedirs('data',exist_ok=True); shutil.copyfileobj(gzip.open('catalog.jsonl.gz','rb'), open('data/catalog.jsonl','wb'))"

One command for the official harness:

    python -m evaluator.local_evaluator

Supporting harnesses:

    python -m unittest discover -s . -p "test_*.py"   # 21 tests
    python ANNA_dump/demo_session.py                  # one multi-turn session
    python ANNA_dump/run_integrated.py                # score + what actually ran
    python ANNA_dump/perturb_eval.py --integrated     # robustness, 5 phrasings
    python ANNA_dump/holdout.py                       # overfitting check

No environment variables are required. `ANTHROPIC_API_KEY` is read only by the
disabled reranking layer and is never needed to reproduce the submitted score.

---

## 7. Team contributions

| person | area | contribution |
|---|---|---|
| A — *name* | Conversational logic and state tracking | Intent routing, slot accumulation, intent-override detection, clarification policy |
| B — Aadhavan Anna | Retrieval and in-memory search engine | Catalog parsing, category scoping, numpy BM25, RRF fusion, phrase evidence, evidence gate, robustness and holdout harnesses |
| C — *name* | Semantic reranking | LLM reranking layer, prompt construction, contract validation and response assembly |
| D — *name* | Integration, memory and deliverables | Pipeline orchestration, session memory and context distillation, submission packaging |

*Fill in names before submission.*
