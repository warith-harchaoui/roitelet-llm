# Answer-quality eval suite

Grades the **synthesis** layer of Roitelet end-to-end. The default
`pytest tests/` run skips every test in this folder — these are slow,
network-dependent, and require a running Ollama instance.

## Why this exists

The 61 unit tests in `tests/test_core.py`, `tests/test_api.py`, and
`tests/test_pipeline.py` verify *behaviour shape*: routing decisions,
storage atomicity, schema validation, fallback paths. None of them
verify *answer quality*.

The Elo loop in `core/registry.py` is the only quality signal in
production, and it's self-referential — the judge crowns winners, those
winners gain Elo, and the next turn's judge sees them again. If judging
quality degrades, Elo can't tell you.

These tests fix that. Each prompt in `dataset.json` is run through
`run_roitelet_chat`; the resulting synthesis is graded against:

1. **Faithfulness** — every claim must be supported by at least one of
   the candidate responses (no hallucinated facts via fusion).
2. **Correctness** — the synthesised answer must match a curated
   reference (`expected_output`).
3. **Answer relevancy** — the synthesised answer must actually address
   what the user prompt asked, not drift onto a related-but-different
   topic.

The first two metrics run in the default-preferences (full-fleet) mode.
Faithfulness *additionally* runs in independence mode
(`RouterPreferences(independence=True)`) so the OSS-only-vs-full-fleet
fusion quality delta is visible in the report — the most important
number for the local-first value prop.

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
# 1. install the eval extras
pip install -e .[eval]

# 2. make sure Ollama is up and the configured local model is pulled
ollama serve &
ollama pull qwen3:8b      # or whatever LOCAL_LLM_MODEL points at

# 3. run only the eval tests
pytest -m eval -q

# 3a. or run a single case
pytest tests/eval/test_judge_quality.py -m eval -k coding-fizzbuzz -v
```

The full pass costs roughly:
- 5 prompts × 3 candidate models × 1 fusion call = 20 Roitelet turns,
- 5 × 2 metrics × ~3 grader calls each = 30 grader calls.

Plan for ~5 minutes wall time on a laptop with local-only models, and
real API spend if your routing picks paid candidates.

## Extending the dataset

`dataset.json` is intentionally small. Add new entries with:

```json
{
  "id": "kebab-case-id",
  "category": "coding | math | reasoning | multilingual | summarization | ...",
  "prompt": "The user-visible prompt.",
  "expected_output": "A concise reference answer — keep it factual, not stylistic."
}
```

Avoid stylistic references ("must be written in a friendly tone") — the
`GEval` correctness rubric is about *facts*, and stylistic noise causes
false negatives.

## Skipping conditions

The suite skips gracefully when:

- `deepeval` is not installed (`pytest.importorskip` at the top of
  `conftest.py`),
- Ollama is unreachable at `LOCAL_LLM_BASE_URL` (a session-scoped
  `autouse` fixture probes `/api/tags` and skips if it errors).

If you want to fail loudly on these conditions instead of skipping,
remove the `pytest.skip(...)` call in `_require_ollama_reachable`.
