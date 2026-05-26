# Answer-quality eval suite

Grades the **synthesis** layer of Roitelet end-to-end. The default
`pytest tests/` run skips every test in this folder — these are slow,
network-dependent, and require a running Ollama instance.

## Why this exists

The 131 unit tests in `tests/test_core.py`, `tests/test_api.py`,
`tests/test_pipeline.py`, `tests/test_commands.py`,
`tests/test_router_mf.py`, `tests/test_image_pipeline.py`, and
`tests/test_capability_classifier.py` verify *behaviour shape*:
routing decisions, storage atomicity, schema validation, fallback
paths, slash-command parsing, learned-router fitting, image-gen
pipeline. None of them verify *answer quality*.

The Elo loop in `core/registry.py` is the only quality signal in
production, and it's self-referential — the judge crowns winners, those
winners gain Elo, and the next turn's judge sees them again. If judging
quality degrades, Elo can't tell you.

These tests fix that.

## What's in here

| File | What it does |
|---|---|
| `test_judge_quality.py` | Per-prompt grading: correctness, faithfulness, answer relevancy. Marker-gated as `eval`. |
| `bench_pareto.py` | Cost-quality Pareto: runs each prompt through the fusion pipeline *and* scores every candidate in isolation, then writes a JSON summary to a gitignored working directory. Marker-gated. |
| `dataset.json` | 25 hand-curated `(id, category, prompt, expected_output)` cases spread across the 8 capability buckets, including 3 fusion-hostile prompts. |
| `conftest.py` | The `OllamaJudge` (`DeepEvalBaseLLM`) adapter that routes grading calls through the project's own `OllamaClient`, plus the session-scoped Ollama-reachable check that skips the suite cleanly when the server is down. |

## The metrics

Each prompt in `dataset.json` is run through `run_roitelet_chat`; the
resulting synthesis is graded against:

1. **Correctness** (`GEval`) — the synthesised answer matches the curated
   reference (`expected_output`).
2. **Faithfulness** — every claim is supported by at least one of the
   candidate responses (no hallucinated facts via fusion). **Runs twice
   per prompt**: once in default mode, once with
   `RouterPreferences(independence=True)` so the OSS-only vs full-fleet
   quality delta is visible — the most important number for the
   local-first value prop.
3. **Answer relevancy** — the synthesised answer actually addresses what
   the user prompt asked (catches topic drift the correctness rubric
   misses when the drift happens to land on something *also* factually
   correct).

All metrics are computed by
[DeepEval](https://github.com/confident-ai/deepeval), pinned to the
`>=3.0,<4.0` family in `pyproject.toml`. The metric and test-case
import paths we rely on have held across all 3.x patches, but DeepEval
has shipped silent metric changes in major bumps before — only widen
the pin after running the full `pytest -m eval` pass on the candidate
release.

## What the grader is

The grader is a `DeepEvalBaseLLM` subclass (`conftest.py::OllamaJudge`)
that routes DeepEval's grading calls through the same `OllamaClient` the
rest of Roitelet uses. No paid keys are needed to grade; the grader's
notion of "good" matches the production judge by construction.

If you want a stricter grader (e.g. GPT-4o), swap the `ollama_judge`
fixture for a DeepEval-built-in. Beware: the eval suite then incurs paid
API cost on every run.

## Running

```bash
# 1. install the eval extras (quote the brackets for zsh)
pip install -e '.[eval]'

# 2. make sure Ollama is up and the configured local model is pulled
ollama serve &
ollama pull qwen3:8b      # or whatever LOCAL_LLM_MODEL points at

# 3. run only the eval tests
pytest -m eval -q

# 3a. or run a single case
pytest tests/eval/test_judge_quality.py -m eval -k coding-fizzbuzz -v

# 3b. or just the Pareto runner
pytest tests/eval/bench_pareto.py -m eval -q -s
```

A full pass touches:

- 25 prompts × 3 candidate models × 1 fusion call ≈ 100 Roitelet turns
  (Pareto runner re-uses the same fan-out, doesn't duplicate),
- 25 × correctness + 25 × relevancy + 50 × faithfulness (default + independence)
  = ~100 grader calls,
- each grader call is one local-Ollama generation.

Plan for **30–60 minutes wall time** on a laptop with local-only models
(qwen3:8b synth judge is the slow part), and real API spend if your
routing picks paid candidates.

For a faster sanity check, run a single `-k` subset or just the Pareto
runner on the first few prompts.

## Validating the learned MF router

The eval suite is also the right path to verify `ROITELET_ROUTER=mf`
end-to-end. Once you have **≥ 32 telemetry records** on disk (the
learned classifier's minimum fit threshold), run with the env var set:

```bash
ROITELET_ROUTER=mf pytest -m eval tests/eval/test_judge_quality.py \
    -k coding-reverse-string -v -s
```

The reasoning trail in the routing decision (visible if you bump the
log level) calls out `LearnedMFRouter active` so you know the router
actually engaged rather than silently falling back to the heuristic.

## Extending the dataset

`dataset.json` is intentionally compact. Add new entries with:

```json
{
  "id": "kebab-case-id",
  "category": "coding | math | reasoning | writing | analysis | multilingual | long_context | summarization",
  "prompt": "The user-visible prompt.",
  "expected_output": "A concise reference answer — keep it factual, not stylistic."
}
```

Avoid stylistic references ("must be written in a friendly tone") — the
`GEval` correctness rubric is about *facts*, and stylistic noise causes
false negatives.

The `category` field doubles as the training label for the optional
embedding-based capability detector
(`core/capability_classifier.py`). Adding labelled prompts here
strengthens both the regression suite and the classifier.

## Skipping conditions

The suite skips gracefully when:

- `deepeval` is not installed (`pytest.importorskip` at the top of
  `conftest.py`),
- Ollama is unreachable at `LOCAL_LLM_BASE_URL` (a session-scoped
  `autouse` fixture probes `/api/tags` and skips if it errors).

If you want to fail loudly on these conditions instead of skipping,
remove the `pytest.skip(...)` call in `_require_ollama_reachable`.
