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

### 4.4 — Judge-swap at K=2 (2026-05-26)

The §4.2 / §4.3 runs hold the judge fixed. This run holds *everything
else* fixed and rotates the synthesis judge across three sizes, so any
shift in the fused answer is attributable to the judge itself, not the
candidates.

**Configuration**

- Dataset: identical to §4.2 (25 prompts, 8 categories).
- Router: heuristic, `independence=True`, `allow_vlms=True`.
- Candidate pool: `llama3.2:3b`, `qwen2.5:3b`, `gemma3:4b`.
- K: **2** (the §4.3 sweet spot — locks the fan-out so the judge is
  the only moving part).
- Synthesis judges (rotated): `qwen3:8b` (8B, the default), `llama3.2:3b` (3B),
  `gemma3:4b` (4B).
- DeepEval `GEval(correctness)` grader: **`qwen3:8b`** held constant
  across all three judge runs. Treat absolute scores for the `qwen3:8b`
  judge with caution (same-family grader/judge); the comparison
  between judges is still informative because the grader is the same
  for all three.
- Total wall time: **137.9 min** (62.9 min inference + 75.0 min grading).
- Artefact: `eval_runs/judgeswap-20260526T123130Z.json` in the
  ignored working directory.

**Headline results**

| Judge | mean correctness | median | pass (≥0.6) | n | mean turn latency | mean judge latency |
|---|---|---|---|---|---|---|
| **qwen3:8b** (8B)   | **0.93** | 1.00 | 24 / 25 | 25 | 53.6 s | 38.9 s |
| **gemma3:4b** (4B)  | 0.88     | 1.00 | 23 / 25 | 25 | 56.4 s | 18.4 s |
| **llama3.2:3b** (3B) | 0.72    | 1.00 | 19 / 25 | 25 | 40.8 s | 20.3 s |

**Per-category mean** (rows = judge, columns = category):

| Judge | analysis | coding | long_ctx | math | multilingual | reasoning | summarisation | writing |
|---|---|---|---|---|---|---|---|---|
| qwen3:8b   | 0.90 | 0.90 | 0.60 | **1.00** | 0.97 | **1.00** | 0.80 | 0.95 |
| gemma3:4b  | 0.95 | **1.00** | 0.60 | 0.94 | 0.47 | 0.96 | 0.80 | **1.00** |
| llama3.2:3b | 0.85 | 0.96 | 0.60 | 0.70 | 0.67 | 0.66 | 0.60 | 0.40 |

**Winner attribution** — which candidate the judge chose as the
fused-answer winner, summed over the 25 prompts (K=2, so 1–2 winners
per turn; a "winning candidate" is selected at least once per turn):

| Judge | gemma3:4b chosen | qwen2.5:3b chosen | llama3.2:3b chosen |
|---|---|---|---|
| qwen3:8b    | 23 | **4**  | 19 |
| gemma3:4b   | 21 | 20     | **1** |
| llama3.2:3b | 20 | 16     | 6 |

**Findings**

1. **Judge size matters and the gap is big.** The 8B judge scores
   **+22 pp** over the 3B judge on the same prompts with the same
   candidates (0.93 vs 0.72). The 4B judge sits in between at 0.88.
   This is the load-bearing result: most of Roitelet's wall-clock is
   the judge, and downsizing the judge gives back substantial
   correctness — not just speed. Budget the judge, don't starve it.
2. **Latency cost of the larger judge is ~2×, not free.** Mean judge
   wall-clock: qwen3:8b 38.9 s, gemma3:4b 18.4 s, llama3.2:3b 20.3 s.
   So the 8B judge buys +22 pp correctness at ~+19 s per turn versus
   the 3B judge. On a laptop that trade is worth making; in a
   latency-budgeted regime (chat-completion, real-time UI) a 4B judge
   is the better Pareto point — only −5 pp correctness for half the
   judge wall-clock.
3. **Self-preference is not the dominant judge bias on this dataset.**
   The naive worry ("a judge prefers its own family") is only partly
   borne out: `gemma3:4b` judge does pick `gemma3:4b` candidate most
   often (21/25), but the more striking pattern is **anti-llama3.2:3b
   bias on the smaller judges** — `gemma3:4b` picks `llama3.2:3b`
   only **1/25** times, and `llama3.2:3b` itself picks the
   `llama3.2:3b` candidate only 6/25. The likely mechanism: the
   smaller judges are weaker at parsing terse responses, and
   `llama3.2:3b` candidate answers were the most terse in this pool.
4. **Strongest disagreement is on writing and multilingual.** Writing:
   qwen3:8b 0.95 vs gemma3:4b 1.00 vs llama3.2:3b 0.40 — the small
   judge marks creative-tone outputs as failures that the bigger
   judges accept. Multilingual: qwen3:8b 0.97 vs gemma3:4b 0.47 vs
   llama3.2:3b 0.67 — `gemma3:4b` judge is the worst on multilingual
   despite being family-neutral on those candidates, suggesting it
   evaluates non-English idioms less reliably than even the 3B
   llama judge.
5. **Per-prompt disagreement is universal.** All 25/25 prompts have at
   least one judge picking a different winner set; 8/25 prompts have
   judges disagreeing on PASS/FAIL outright. Notable PASS→FAIL splits
   under `llama3.2:3b`: `math-quadratic` (0.00 vs 1.00 from qwen3:8b
   and gemma3:4b), `reasoning-birthday-paradox` (0.00 vs 1.00 / 1.00),
   `writing-tone-rewrite` (0.00 vs 0.90 / 1.00). The smaller judge
   is not just noisier — it is **systematically harsher** on prompts
   where the gold answer requires nuanced verification (math
   identities, paradox framing, register shifts).
6. **Long-context is the universal weak spot, judge-independent.**
   All three judges score 0.60 on `long_context` (the single-prompt
   category, `long-context-document-types`). This is consistent with
   §4.2/§4.3 and confirms the regression is in the candidate
   answers or the reference set, not in the judge.

**Honest caveats**

- The grader (`qwen3:8b`) is the same family as one of the judges
  (`qwen3:8b`). This *could* bias the qwen3:8b judge's absolute
  score up; the comparison **between** judges remains fair because
  the grader is constant. To break the circularity fully, swap to a
  stronger external grader (§5 #8).
- N=25 prompts × 3 judges = 75 grades. Headline gaps (22 pp between
  8B and 3B; 5 pp between 8B and 4B) are large enough to survive
  noise, but per-category cells with 1–2 prompts (`long_context`,
  `summarisation`) are not statistically resolvable. Direction, not
  magnitude.
- The candidate pool is intentionally small and OSS-only. A
  remote-augmented pool would likely narrow the small-vs-large judge
  gap because frontier candidates produce clearer, less ambiguous
  text that even the 3B judge can grade.
- K=2 is the §4.3 sweet spot. The judge-swap effect at K=3 is likely
  larger (more drafts to fuse → judge reasoning quality matters
  more), but expensive to measure on the same compute budget.

**Practical recommendation**: keep `qwen3:8b` as the default judge on
machines with the headroom; **fall back to `gemma3:4b` (not `llama3.2:3b`)
when judge latency is the binding constraint** — 4B costs you 5 pp
of correctness for half the wall-clock; 3B costs you 22 pp and
introduces strong anti-terse-candidate bias.

That's the load-bearing run set. Everything in §5 below is **planned**
until proven otherwise. Raw JSON reports
(`ksweep-20260526T045340Z.json`, `ksweep-20260526T083344Z.json`,
`judgeswap-20260526T123130Z.json`) live in the ignored `eval_runs/`
working directory.

---

## 5. Planned ablations (priority order)

0. ~~**Re-run the K-sweep with a real K=3 candidate pool.**~~ Done in
   §4.3 (2026-05-26 evening). Real K=3 measured with
   `allow_vlms=True` and `gemma3:4b` in the fan-out. K=2 confirmed
   as the sweet spot.
1. **K-sweep with heuristic router** (`K ∈ {1, 2, 3, 5}`) on the full
   25-prompt dataset, with the default Qwen 3 8B judge. Goal: pin
   down where K stops paying off.
2. ~~**Judge-swap** at fixed K: Qwen 3 8B vs Llama 3.2 3B vs Gemma 3
   4B as the synthesis judge, same dataset.~~ Done in §4.4
   (2026-05-26) at K=2. 8B judge wins by +22 pp over 3B; 4B is the
   Pareto sweet spot if judge latency is binding.
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
