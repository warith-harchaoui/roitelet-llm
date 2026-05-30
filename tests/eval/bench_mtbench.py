"""MT-Bench runner for Roitelet — the external benchmark Roitelet's router
suffers the comparison against.

Why MT-Bench
------------
The ablation in ``test_judge_quality`` and ``bench_pareto`` runs on a
hand-curated 25-prompt dataset. That's useful for fast iteration on the
project's own behaviour, but it's not a benchmark anyone else has heard
of. To stand alongside RouteLLM (which publishes on MT-Bench, MMLU and
GSM8K) the Roitelet eval needs at least one canonical benchmark.

MT-Bench is the natural pick:

* It's 80 multi-turn prompts across eight categories
  (writing / roleplay / reasoning / math / coding / extraction / stem /
  humanities). The shape matches Roitelet's design (open-ended answers
  fused by a local judge) far better than MMLU's multiple-choice format.
* It overlaps with RouteLLM's reported numbers — useful for honest
  comparison even though we use a *local* judge (qwen3:8b) instead of
  GPT-4.
* It's distributed under permissive licenses through Hugging Face
  ``lmsys/mt_bench_human_judgments``, no scraping required.

This runner is opt-in (``pytest -m eval``) for two reasons: it needs
``[eval]`` extras (DeepEval + ``datasets``) and it issues real Ollama
calls per prompt. Expect a wall-clock of several minutes on CPU.

What it does
------------
For each MT-Bench prompt (first turn only — we don't yet implement
multi-turn for the runner):

1. Run the prompt through ``run_roitelet_chat`` end-to-end.
2. Score the fused answer with the local-Ollama-backed GEval
   correctness rubric (same grader the judge eval uses).
3. Record (prompt, category, score, total wall-clock,
   router decision, candidates) in a JSON report.

The report lands under ``.private/eval_runs/`` so a regression diff
between commits is straightforward (the ``make eval`` wrapper handles
the file-naming).

What it does NOT do (yet)
-------------------------
* **Multi-turn.** MT-Bench prompts are two-turn dialogues. We only
  run the first turn; the second-turn ablation is a follow-up.
* **GPT-4 grading.** RouteLLM's headline numbers are graded with
  GPT-4. We grade with a local model — honest but not directly
  comparable. The README and PSEUDO.md say so.
* **The full 80 prompts at once.** The default run takes a
  configurable ``MTBENCH_LIMIT`` slice (env var, default 16) so a
  developer iteration is bounded. Set ``MTBENCH_LIMIT=0`` to run
  the full set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# datasets is in [eval]; conftest already importorskips deepeval, but
# datasets needs its own importorskip in case someone installs only
# part of the extra by hand.
datasets = pytest.importorskip(
    'datasets',
    reason='datasets not installed — install with `pip install -e .[eval]`',
)

from deepeval.metrics import GEval  # noqa: E402  (importorskip gate above)
from deepeval.test_case import LLMTestCase, SingleTurnParams  # noqa: E402

from core.pipeline import run_roitelet_chat  # noqa: E402
from core.schemas import ChatRequest, RouterPreferences  # noqa: E402

# How many prompts to run by default. The full MT-Bench is 80; on
# qwen3:8b CPU that's ~3 hours. ``MTBENCH_LIMIT=0`` runs everything.
_DEFAULT_LIMIT = int(os.environ.get('MTBENCH_LIMIT', '16'))


@pytest.fixture(scope='module')
def mtbench_prompts():
    """Load the first-turn MT-Bench prompts via Hugging Face datasets.

    The dataset has one row per (model, question, judgment), so we
    deduplicate on question_id to get one row per question. We keep
    the first turn only.
    """
    try:
        ds = datasets.load_dataset(
            'lmsys/mt_bench_human_judgments',
            split='human',
        )
    except Exception as exc:  # pragma: no cover - HF network dependent
        pytest.skip(f'MT-Bench dataset not loadable: {exc}')

    by_id: dict[int, dict] = {}
    for row in ds:
        qid = row.get('question_id')
        if qid is None or qid in by_id:
            continue
        first_turn = (row.get('conversation_a') or [{}])[0]
        prompt = first_turn.get('content') or ''
        category = row.get('category') or 'misc'
        if prompt:
            by_id[qid] = {'id': qid, 'prompt': prompt, 'category': category}

    selected = sorted(by_id.values(), key=lambda r: r['id'])
    if _DEFAULT_LIMIT > 0:
        selected = selected[:_DEFAULT_LIMIT]
    if not selected:
        pytest.skip('MT-Bench dataset returned no usable prompts.')
    return selected


@pytest.fixture
def mtbench_correctness(ollama_judge):
    """The same GEval rubric the judge-quality eval uses.

    Sharing the rubric across eval files is what makes the numbers
    comparable. A future MT-Bench-specific rubric (one per category,
    say) would justify its own metric — for now, one rubric.
    """
    return GEval(
        name='Correctness',
        criteria=(
            'Determine whether the `actual_output` is factually correct, '
            'helpful, and complete given the `input`. Penalise hallucinations '
            'and missing key facts.'
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        model=ollama_judge,
        threshold=0.6,
    )


def _report_dir() -> Path:
    """Where to write JSON reports — gitignored so commits stay clean."""
    target = Path(__file__).resolve().parent.parent.parent / '.private' / 'eval_runs'
    target.mkdir(parents=True, exist_ok=True)
    return target


def test_mtbench_first_turn(mtbench_prompts, mtbench_correctness):
    """Run Roitelet on a slice of MT-Bench's first turns and write a JSON report.

    Asserts only that every selected prompt produces a non-empty
    answer — the *quality numbers* are what the JSON report carries.
    Failing on score threshold across categories would make the test
    bimodal in a way that hides real signal; the report is the
    artefact, the assertion is just liveness.
    """
    import asyncio

    rows: list[dict] = []
    started = time.perf_counter()
    for case in mtbench_prompts:
        prompt = case['prompt']
        try:
            response = asyncio.get_event_loop().run_until_complete(
                run_roitelet_chat(
                    ChatRequest(prompt=prompt, preferences=RouterPreferences()),
                ),
            )
            content = response.synthesis.content or ''
        except Exception as exc:
            rows.append({
                'id': case['id'],
                'category': case['category'],
                'error': str(exc),
            })
            continue

        score: float | None = None
        try:
            tc = LLMTestCase(input=prompt, actual_output=content)
            mtbench_correctness.measure(tc)
            score = float(mtbench_correctness.score or 0.0)
        except Exception as exc:
            # Grading failed; still record the row.
            score = None
            grader_error = str(exc)
        else:
            grader_error = None

        rows.append({
            'id': case['id'],
            'category': case['category'],
            'prompt_chars': len(prompt),
            'answer_chars': len(content),
            'score': score,
            'grader_error': grader_error,
            'total_latency_s': float(response.total_latency_s),
            'selected': list(response.router.selected_model_ids),
        })
        assert content, f'MT-Bench prompt {case["id"]} produced an empty answer.'

    elapsed = time.perf_counter() - started
    summary = {
        'rows': rows,
        'meta': {
            'limit': _DEFAULT_LIMIT,
            'wall_clock_s': elapsed,
            'mean_score': (
                sum(r['score'] for r in rows if r.get('score') is not None)
                / max(1, sum(1 for r in rows if r.get('score') is not None))
            ),
        },
    }
    out = _report_dir() / f'mtbench-{int(time.time())}.json'
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n[bench_mtbench] wrote {out.relative_to(out.parent.parent.parent)} '
          f'({len(rows)} rows, mean={summary["meta"]["mean_score"]:.2f})')
