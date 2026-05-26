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

- **2026-05-26** — single-turn validation of the learned MF router
  (`ROITELET_ROUTER=mf` after bootstrapping 36 real telemetry
  records). DeepEval GEval correctness on `coding-reverse-string`:
  heuristic = 1.00, learned = 0.80. Both pass the 0.6 threshold. One
  prompt; informative but not conclusive. Log archived under the
  `eval_runs/` working directory.

That's the only end-to-end DeepEval-graded run on record. Everything
below is **planned** until proven otherwise.

---

## 5. Planned ablations (priority order)

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
