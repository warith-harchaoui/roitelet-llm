# Evaluation and ablation roadmap

Roitelet routes, fans out and fuses LLM answers. None of these design
choices is automatically better than a single well-chosen model.
Whether they help depends on the prompt class, the candidate
diversity, the judge model, and the K value. The only honest way to
defend or refute the design is to **measure**.

This document is the standing ablation plan. It describes the matrix
of comparisons that the project *should* run, not a record of what
has already been run. Where actual results exist, they are cited
explicitly with a date and a link to the run artefact; everything
else is marked as **planned**.

> ⚠️ Anything in this document that does not cite a dated run is a
> hypothesis to be tested, not a claim about how the system behaves.

---

## 1. What we're comparing

Roitelet's value claim is comparison + fusion. The right baseline
set therefore spans both ends of the spectrum:

| Configuration | What it tests |
|---|---|
| **Single strongest model** | Strong-baseline ceiling. If Roitelet's fused answer can't beat this for the prompt class, fusion isn't helping. |
| **Cheapest acceptable model** | Cost floor. If Roitelet only matches this, the extra latency + tokens were wasted. |
| **Routed top-1** | Router-only signal. Does the router pick well *without* fusion adding noise? |
| **Routed top-K, no fusion** | Top-K average / max baseline. Does picking K candidates already help without a fusion pass? |
| **Routed top-K + local fusion** | The full Roitelet path. |
| **Routed top-K + different judge models** | Judge-bias surface. Does the rolling-Elo loop and the fused answer change shape when the judge is swapped? |
| **Local-only mode** | OSS-only ceiling. How much quality do you give up when you forbid remote candidates? |
| **Remote-only candidates with local judge** | Inverse of local-only — does a local judge on remote answers add value over the strongest single remote? |

For each configuration, vary:

- **K**: 1, 2, 3, 5.
- **Task class**: coding, reasoning, writing, multilingual, factual
  QA, long-context summarisation, multimodal (if attachments are
  supported on that turn).
- **Judge model**: Qwen 3 8B (default), Llama 3.2 3B, Gemma 3 4B,
  Mistral Small. Picking three diverse judges is enough to surface
  judge-conditioned drift.

---

## 2. Metrics

| Metric | What it measures | How to compute |
|---|---|---|
| **Correctness** | The fused answer is factually right. | DeepEval `GEval` against a curated reference (`tests/eval/dataset.json`). |
| **Faithfulness** | Every claim in the fused answer is supported by at least one candidate. | DeepEval `FaithfulnessMetric` with the candidates as retrieval context. |
| **Answer relevancy** | The fused answer addresses the actual question. | DeepEval `AnswerRelevancyMetric`. |
| **Win rate vs best single candidate** | Fusion's gross uplift. | Pairwise A/B with an external judge (LLM or human) over the dataset. |
| **Per-turn cost (USD)** | Sum of paid-token spend. | Already in telemetry. |
| **Wall-clock latency** | User-perceived time. | Already in telemetry (`ChatResponse.responses[*].latency_s` + judge latency). |
| **Failure rate** | Per-turn frequency of "all candidates errored" or empty judge output. | Already in telemetry. |
| **Judge agreement** | Stability of winner under judge-model swap. | Run the same prompts through K candidates, vary only the judge model, measure how often the winner set changes. |
| **Privacy exposure level** | How much of the prompt leaves the local box. | Per-turn classification: `local-only`, `local-judge-remote-candidates`, `fully-remote`. Derived from the router decision. |
| **User preference** | Where it's available. | Per-conversation thumbs-up/down (not yet implemented; tracked in the deferred list). |

---

## 3. The runner

The existing `tests/eval/bench_pareto.py` runner is the seam. It
replays the eval dataset, scores per-candidate and fused answers
separately, and writes a JSON report. The full ablation matrix above
is just running that runner under different env vars:

```bash
# Baseline: heuristic router, default judge, K=3
make eval

# Learned-router branch
ROITELET_ROUTER=mf make eval

# Different judge (set via runtime settings, not env, today)
# — see scripts/set_judge.py (planned)
ROITELET_JUDGE=llama3.2:3b make eval

# Local-only mode (independence preferences override applied per-run)
ROITELET_PREFS_INDEPENDENCE=1 make eval
```

Each invocation writes its report to a gitignored directory and tags
the report with the configuration. Cross-run comparison is a small
follow-up script (planned) that reads multiple reports and emits a
Pareto plot.

### Current dataset coverage

`tests/eval/dataset.json` ships with 25 hand-curated prompts across:
coding (5), math (5), reasoning (5), writing (2), analysis (2),
multilingual (3), long_context (1), summarisation (2). Three of the
prompts are intentionally fusion-hostile (candidates legitimately
disagree). This is enough to **find** regressions, not to **claim
victory**. A larger labelled corpus is in the deferred list.

---

## 4. What has actually been run

This is the honest record. Each entry has a date and a link. **Do
not add entries here unless you have the artefact to point at.**

### 4.1 — Single-prompt learned-router validation (2026-05-26)

After bootstrapping 36 real telemetry records, the learned MF router
(`ROITELET_ROUTER=mf`) was compared against the heuristic on
`coding-reverse-string`. DeepEval GEval correctness: heuristic = 1.00,
learned = 0.80; both pass the 0.6 threshold. One prompt — informative
but not conclusive. Log archived in the ignored `eval_runs/` working
directory.

### 4.2 — K-sweep on 25 prompts (2026-05-26)

**Configuration**

- Dataset: `tests/eval/dataset.json`, 25 prompts across 8 categories
  (coding, math, reasoning, writing, analysis, multilingual,
  long_context, summarisation), including 3 fusion-hostile prompts.
- Router: heuristic (default), `independence=True`.
- Candidate pool: 3 small local OSS models — `llama3.2:3b`,
  `qwen2.5:3b`, `gemma3:4b` — pinned via the Ollama live-discovery
  cache. Pool capped to keep the experiment self-contained on a
  laptop; the same machinery scales to larger pools.
- Synthesis judge: `qwen3:8b` (Roitelet's default).
- DeepEval `GEval(correctness)` with the local Ollama judge, threshold
  0.6. Same judge model as the synthesis judge — a circularity the
  ablation in §5 #8 is designed to break.
- Total wall time: **74.6 min** (45.3 min inference + 29.3 min grading).
- Artefact: `eval_runs/ksweep-20260526T045340Z.json` in the ignored
  working directory.

**Headline results**

| K | mean correctness | median | pass (≥0.6) | n | total latency | candidate max | judge |
|---|---|---|---|---|---|---|---|
| 1 | **0.78** | 1.00 | 21 | 25 | 27.7 s | 2.9 s | 24.8 s |
| 2 | **0.90** | 1.00 | 23 | 25 | 37.8 s | 6.9 s | 30.9 s |
| 3 | **0.87** | 1.00 | 23 | 25 | 43.2 s | 7.7 s | 35.5 s |

**Per-category mean** (rows = K, columns = category):

| K | analysis | coding | long_ctx | math | multilingual | reasoning | summarisation | writing |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.90 | 0.84 | 0.50 | 0.76 | 0.50 | 0.88 | 0.60 | 1.00 |
| 2 | 0.90 | 0.90 | 0.70 | **1.00** | 0.67 | **0.96** | 0.80 | 1.00 |
| 3 | 0.90 | 0.82 | 0.50 | 1.00 | **1.00** | 0.84 | 0.60 | 1.00 |

**Findings**

1. **K=1 → K=2 is a real, broad uplift.** +12 pp mean correctness
   (0.78 → 0.90), +2 prompts cleared the 0.6 threshold (21 → 23),
   at a +10 s wall-clock cost. The biggest per-prompt gains were
   `fusion-hostile-units` (+1.00), `reasoning-tradeoff-cache` (+0.40),
   `coding-async-vs-thread` (+0.40), `multilingual-spanish-greeting`
   (+0.40), `summarization-photosynthesis` (+0.40). One regression:
   `coding-dict-merge` (−0.10).
2. **The K=3 row does not actually test K=3 — and the cause is an
   experimental-design mistake, not a routing bug.** Every K=3 turn
   collapsed to `fan_out=2` because one of the three "small local"
   models I picked for the pool, `gemma3:4b`, is registered as a
   VLM in `data/bootstrap/model_priors.json` (`vlm: 1`, with a
   `vision` capability prior). The router's VLM-protection filter
   correctly drops VLM specs on non-vision prompts when
   `allow_vlms=False`: `if not preferences.allow_vlms and spec.vlm
   and 'vision' not in categories: continue`. Across all 25 prompts
   in the dataset, none triggered the `vision` capability detector
   (no `image`, `photo`, `diagram`, `screenshot`, `chart` keywords),
   so `gemma3:4b` was filtered every turn and the router was left
   with two text candidates regardless of `top_k`. The K=3 row is
   therefore K=2 with the same candidate set as the K=2 row, not a
   real K=3 measurement. To get real K ≥ 3 the pool needs a third
   *text* candidate (e.g. `qwen2.5-coder:latest`, `llama3.2:1b`),
   or `allow_vlms=True`. Tracked as §5 #0 below.
3. **The judge dominates wall-clock.** Across all K, the synthesis
   pass took 89–82 % of the total turn time. Candidate fan-out is
   tiny on local small models (≤ 8 s at K=2). User-perceived
   latency improvements should target the judge first — a smaller
   judge, or a budget-capped one, is the next natural ablation.
4. **Multilingual prompts are the weakest category.** K=1 mean 0.50;
   even K=2 only reaches 0.67. The failures share a pattern: the
   model produces a *correct alternative idiom* that doesn't match
   the curated reference (the judge marks 0.00 because the literal
   phrase differs). This is a reference-set quality issue, not a
   model quality issue — listed as data-curation work below.
5. **Persistent failures** (failed at ≥ 2 K values):
   - `reasoning-tradeoff-cache` — K=1 0.40 FAIL, K=2 0.80 PASS, K=3
     0.20 FAIL. The judge wants a specific "low hit-rate raises
     mean latency" framing; the fused answer keeps drifting toward
     "cache invalidation overhead" framings instead.
   - `multilingual-french-idiom` — K=1 0.00, K=2 0.00 (FAIL FAIL);
     K=3-effective passed (1.00) by producing the canonical "L'avenir
     appartient" form. Likely a reference-set artefact; see #4 above.
   - `long-context-document-types` — K=1 0.50 FAIL, K=2 0.70 PASS,
     K=3 0.50 FAIL. The fusion drops "book", "RFC" or "transcript"
     unpredictably across runs.

**Honest caveats**

- N=25 prompts × 3 K values is small. The 0.78 → 0.90 uplift is
  large enough to survive a few percentage points of noise, but the
  per-category breakdowns (some have only 1 or 2 prompts) are not
  statistically resolvable. Treat them as direction, not magnitude.
- Same model is used for both synthesis judging and DeepEval
  grading. That circularity is documented as §5 #8 and is the next
  thing to fix.
- The candidate pool is intentionally small and OSS-only. A
  remote-augmented pool (frontier paid candidates) would change the
  K=2 vs K=1 delta — most likely in K=2's favour, but possibly
  pushing K=1 closer to ceiling and reducing the gap.

### 4.3 — K-sweep rerun with real K=3 (2026-05-26, evening)

The §4.2 sweep couldn't measure real K=3 because `gemma3:4b` was
VLM-filtered out on every text prompt. This rerun fixes that by
flipping `allow_vlms=True` on the runner, with the same three-model
pool (`llama3.2:3b`, `qwen2.5:3b`, `gemma3:4b`). Same dataset, same
judge, same grader. Every K=3 turn now reports `fan_out=3` — Gemma is
genuinely in the fusion.

| K | mean correctness | median | pass (≥0.6) | n | total latency | candidate max | judge |
|---|---|---|---|---|---|---|---|
| 1 | 0.87 † | 1.00 | 23 / 25 | 25 | 32.1 s | 9.6 s | 22.5 s |
| 2 | **0.95** | 1.00 | **25 / 25** | 25 | 55.9 s | 14.8 s | 41.1 s |
| 3 | **0.96** | 1.00 | **25 / 25** | 25 | 112.1 s | 30.2 s | 81.9 s |

† Two of the K=1 "failures" are DeepEval grader errors, not model
failures (`multilingual-french-idiom` and `multilingual-japanese-thanks`
both surfaced as `score=0.00` with `reason: grader error:`). The
respective K=2 scores for the same prompts are 1.00 and 0.90. The
true K=1 mean is therefore slightly higher than 0.87; a re-grade
pass would resolve this, but the K-sweep conclusions below survive
the correction.

**Per-category mean:**

| K | analysis | coding | long_ctx | math | multilingual | reasoning | summarisation | writing |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 0.92 | 0.80 | 0.96 | 0.33 † | 1.00 | 0.80 | 1.00 |
| 2 | 1.00 | 0.90 | 0.90 | 1.00 | **0.93** | 1.00 | 0.80 | 1.00 |
| 3 | 1.00 | 0.92 | 0.60 | 1.00 | **1.00** | 1.00 | 0.85 | 1.00 |

**Headline findings**

1. **K=1 → K=2 is the biggest delta worth paying for.** +8 pp mean
   correctness (0.87 → 0.95), and the failure list drops from 4
   prompts to 0 (every K=2 case clears the 0.6 threshold). The cost
   is +24 s wall-clock per turn.
2. **K=2 → K=3 is at quality ceiling on this dataset.** +1 pp mean
   (0.95 → 0.96), 25/25 pass at both, but wall-clock **doubles**
   (56 s → 112 s) because the judge has to fuse three candidates
   instead of two and `qwen3:8b` engages much longer chain-of-thought
   on three drafts than on two. **K=3 is not worth its cost on this
   pool / judge combination.**
3. **Multilingual is where K-fusion shines.** K=1 mean for
   multilingual = 0.33 (one good answer, two grader errors); K=2
   recovers to 0.93; K=3 hits 1.00. Anecdotally, having Gemma 3 4B
   in the K=3 fan-out helped on Spanish/Japanese/French.
4. **Long-context regresses at K=3.** 0.80 → 0.90 → 0.60. The
   `long-context-document-types` prompt asks for three concrete
   examples; the K=3 fused answer kept dropping one or substituting
   "annual report" for "book/RFC/transcript". This is the only
   category where K=3 is actively worse than K=2. Plausible
   explanation: judge chain-of-thought on three drafts produces an
   over-curated answer that strips concrete examples.
5. **The judge dominates wall-clock at every K.** K=1 70 % of total,
   K=2 74 %, K=3 73 %. Optimising the synthesis judge — a smaller
   judge, a budget cap, or a different judge model entirely — is
   the next natural ablation (§5 #2 below).

**Caveats (unchanged from §4.2 unless noted)**

- N=25 prompts × 3 K values is still small. The K=1 → K=2 delta is
  robust; the K=2 → K=3 delta (+1 pp) is **inside the noise band**
  for this N and should not be over-read.
- DeepEval emitted two grader errors at K=1. They consistently
  resolved at higher K, so the K=2/K=3 means are clean numbers.
- Same judge for synthesis and grading — circularity intact. §5 #8.
- Local-only pool. A remote-augmented pool would shift the K=1
  baseline up and probably narrow the K=1 → K=2 gap.
- **The K-sweep test pool intentionally enabled `allow_vlms=True`** to
  include `gemma3:4b`. This is a documented runner choice, not a
  default. The production VLM filter still applies on every other
  code path.

**Practical recommendation**: if you're running Roitelet on a laptop
with a `qwen3:8b` judge and an OSS-only candidate pool, **default to
K=2**. K=3 buys you almost nothing on this judge for double the
wall-clock. K=1 leaves measurable quality on the table on multilingual
and coding prompts.

That's the load-bearing run. Everything in §5 below is **planned**
until proven otherwise. Both raw JSON reports
(`ksweep-20260526T045340Z.json` and `ksweep-20260526T083344Z.json`)
live in the ignored `eval_runs/` working directory.

---

## 5. Planned ablations (priority order)

0. ~~**Re-run the K-sweep with a real K=3 candidate pool.**~~ Done in
   §4.3 (2026-05-26 evening). Real K=3 measured with
   `allow_vlms=True` and `gemma3:4b` in the fan-out. K=2 confirmed
   as the sweet spot.
1. **K-sweep with heuristic router** (`K ∈ {1, 2, 3, 5}`) on the full
   25-prompt dataset, with the default Qwen 3 8B judge. Goal: pin
   down where K stops paying off.
2. **Judge-swap** at fixed K=3: Qwen 3 8B vs Llama 3.2 3B vs Gemma 3
   4B as the synthesis judge, same dataset. Goal: surface
   judge-conditioned drift in the winner set.
3. **Local-only vs full-fleet** at fixed K=3 with the default judge.
   Goal: quantify the quality cost of `independence` mode — the
   single most important number for the local-first value prop.
4. **Heuristic vs learned-MF router** at fixed K=3, default judge.
   Goal: confirm or refute the working hypothesis that the learned
   router moves rankings in a useful direction once telemetry has
   accumulated.
5. **Cost-budget regime** at K=3 with `max_cost_usd ∈ {None, 0.005,
   0.001, 0}`. Goal: trace the cost-quality Pareto front explicitly.
6. **Embedding vs keyword capability detector** at fixed K=3. Goal:
   test whether the embedding classifier routes paraphrased prompts
   more correctly than the keyword scan.
7. **Long-context expansion**. Today the dataset has one
   `long_context` prompt; this is not enough. Add 10 prompts in the
   2k-8k character range and re-run #1 on the expanded set.
8. **External judge for the ablations themselves**. The eval suite
   currently uses the same local Ollama judge that the production
   pipeline uses; this creates a circularity (the production judge
   blesses the production judge). Switch to a stronger external
   judge for the eval phase only (cost: real API spend).

---

## 6. Deferred — but worth keeping on the list

- **User-preference signal.** Add a thumbs-up/down per assistant
  message in the web UI, tag the conversation, surface it in
  telemetry. Replaces the synthetic LLM judge with a human one for
  the prompts where the user actually disagrees with the system.
- **Giskard adversarial scan.** One-shot discovery run to seed the
  DeepEval dataset with failure cases. Currently deferred per
  maintainer instruction.
- **MT-Bench / MMLU / GSM8K runners.** Standard benchmarks, but they
  reward single-best-answer routing, not fusion; add them with
  awareness that the framing is biased against Roitelet's design.

---

## 7. How to interpret a result

A passing ablation is not a victory; a failing one is not a defeat.
Both are information.

- If **fusion under-performs** the strongest single candidate on a
  prompt class, that's evidence the judge is not adding value for
  that class. The right response is to inspect the judge transcript
  for that prompt and to tighten or replace the judge prompt — not to
  hide the result.
- If **independence mode** matches full-fleet within noise, that's a
  strong signal that the OSS bundle is enough for that workload —
  use the result to argue for local-first deployment.
- If **the learned router** doesn't beat the heuristic, the right
  conclusion is "not enough telemetry yet" or "the heuristic is
  already calibrated well" — not "learned routing is bad".

The goal is to make the design's strengths and weaknesses legible,
not to publish numbers that look good.

---

## See also

- [`docs/SLASH_COMMANDS.md`](SLASH_COMMANDS.md) — runtime overrides
  the ablation runner can flip per prompt.
- [`docs/PRIVACY.md`](PRIVACY.md) — definitions for the "privacy
  exposure level" metric above.
- [`MECHANISM.md`](../MECHANISM.md) — the routing + fusion pipeline
  the ablations are measuring.
