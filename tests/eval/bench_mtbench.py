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

    Designed to run unattended for hours:

    * **Incremental JSON writes** — the same report file is rewritten
      after every prompt, so a process killed at hour 4 still leaves a
      valid partial report on disk (atomically written so a SIGKILL
      mid-write cannot corrupt it).
    * **Soft failures** — pipeline / grader errors are recorded as
      fields on the row and the run continues. The test only fails
      hard if **every** prompt errored (which means Ollama is dead, a
      real signal to act on).
    * **Progress lines** — one ``[mtbench i/n …]`` line per prompt to
      stdout/log so the operator can see liveness without parsing
      JSON.
    """
    import asyncio

    rows: list[dict] = []
    started = time.perf_counter()
    out = _report_dir() / f'mtbench-{int(started)}.json'
    total = len(mtbench_prompts)

    def flush() -> None:
        """Rewrite the report file with whatever has accumulated so far.

        Atomic via tempfile-then-replace so a process killed mid-write
        leaves either the previous valid file or the new one — never a
        truncated one. Cheap (~ms) compared to a turn (~minutes), so
        running it every iteration is fine.
        """
        scored = [r for r in rows if r.get('score') is not None]
        mean = sum(r['score'] for r in scored) / max(1, len(scored))
        elapsed = time.perf_counter() - started
        summary = {
            'rows': rows,
            'meta': {
                'limit': _DEFAULT_LIMIT,
                'completed': len(rows),
                'expected_total': total,
                'wall_clock_s': elapsed,
                'mean_score': mean,
                'status': 'partial' if len(rows) < total else 'complete',
            },
        }
        tmp = out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        tmp.replace(out)

    for i, case in enumerate(mtbench_prompts, start=1):
        prompt = case['prompt']
        prompt_start = time.perf_counter()
        try:
            response = asyncio.get_event_loop().run_until_complete(
                run_roitelet_chat(
                    ChatRequest(prompt=prompt, preferences=RouterPreferences()),
                ),
            )
            content = response.synthesis.content or ''
            pipeline_error: str | None = None
            selected = list(response.router.selected_model_ids)
            turn_latency = float(response.total_latency_s)
        except Exception as exc:
            content = ''
            pipeline_error = str(exc)
            selected = []
            turn_latency = time.perf_counter() - prompt_start

        score: float | None = None
        grader_error: str | None = None
        if content and not pipeline_error:
            try:
                mtbench_correctness.measure(
                    LLMTestCase(input=prompt, actual_output=content),
                )
                score = float(mtbench_correctness.score or 0.0)
            except Exception as exc:
                grader_error = str(exc)

        rows.append({
            'id': case['id'],
            'category': case['category'],
            'prompt_chars': len(prompt),
            'answer_chars': len(content),
            'score': score,
            'pipeline_error': pipeline_error,
            'grader_error': grader_error,
            'total_latency_s': turn_latency,
            'selected': selected,
        })

        # Incremental checkpoint after every prompt so an aborted run
        # still has every completed turn on disk.
        flush()

        # Liveness line. Unbuffered so ``tail -f`` shows it immediately
        # under ``nohup``.
        elapsed_min = (time.perf_counter() - started) / 60.0
        score_str = f'{score:.2f}' if score is not None else 'n/a '
        status = 'OK' if (content and not pipeline_error) else 'FAIL'
        print(
            f'[mtbench {i:>3}/{total}] {status} score={score_str} '
            f'cat={case["category"]:<14s} '
            f'lat={turn_latency:5.1f}s elapsed={elapsed_min:5.1f}m',
            flush=True,
        )

    # Final write happens inside ``flush`` already, but call it once
    # more in case the loop body skipped a flush on some edge case.
    flush()
    rel = out.relative_to(out.parent.parent.parent)
    print(f'\n[bench_mtbench] wrote {rel} ({len(rows)} rows)', flush=True)

    # Hard failure only if *every* prompt failed — that's a real
    # operator signal (Ollama down, broken creds), not noise from
    # one weird prompt.
    failures = [r for r in rows if r['pipeline_error']]
    if len(failures) == len(rows) and rows:
        raise RuntimeError(
            f'Every MT-Bench prompt failed in the pipeline ({len(rows)} rows). '
            f'First error: {failures[0]["pipeline_error"]}'
        )
