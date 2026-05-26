"""Cost-quality Pareto benchmark for Roitelet's fusion vs. single-best.

Why this exists
---------------
The DeepEval suite in :mod:`tests.eval.test_judge_quality` answers the
question "is the fused answer good in absolute terms?". It does **not**
answer the comparative question "is the fused answer worth the cost of
fan-out + judge over just calling the strongest single candidate?".

That comparative question is exactly what RouteLLM's published Pareto
curves answer for their two-model routing setup. The runner here is
Roitelet's equivalent: for each prompt in ``dataset.json`` it captures

* the fused-answer correctness (the Roitelet output),
* each candidate model's standalone correctness on the same prompt,
* the per-call cost in USD for each path,
* the wall-clock latency.

The output is a JSON report (gitignored) that lets us look at the
trade-off directly:

    average correctness per dollar  — fusion vs. best-single vs. random-single

What this is **not**
--------------------
It is not a RouteLLM head-to-head. RouteLLM trains a router between
*one* strong and *one* weak model; Roitelet picks K from a pool and
fuses. The fair comparison is "Roitelet's fusion vs. the strongest
single candidate the router would have picked", which is what this
runner measures.

Running
-------
    pip install -e .[eval]
    pytest -m eval tests/eval/bench_pareto.py -q -s

The runner is marker-gated like the rest of ``tests/eval/`` — it skips
in the default ``pytest`` run. ``-s`` is helpful because the runner
prints a one-line summary per prompt so a long run shows liveness.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

import pytest

# DeepEval is required — conftest.py importorskips early if absent.
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from core.pipeline import run_roitelet_chat
from core.registry import get_registry
from core.schemas import ChatRequest, RouterPreferences

_DATASET = json.loads((Path(__file__).parent / 'dataset.json').read_text())

# Persist the report under an ignored working directory so it doesn't
# pollute tracked state but is still inspectable + diffable across runs.
# The default location is honoured by ``.gitignore`` so reports stay
# out of git without any extra setup.
_REPORT_DIR = Path(__file__).resolve().parent.parent.parent / '.private' / 'eval_runs'


def _now_stamp() -> str:
    """ISO-8601 UTC stamp safe for a filename."""
    return datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%SZ')


@pytest.fixture(scope='module')
def correctness_metric(ollama_judge):
    """Single shared GEval rubric — building one per case costs a model load."""
    return GEval(
        name='Correctness',
        criteria=(
            'Determine whether the `actual_output` is factually correct '
            'and complete given the `expected_output`. Minor phrasing '
            'differences are acceptable; missing key facts or stating '
            'false ones is not.'
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=ollama_judge,
        threshold=0.6,
    )


def _score(metric, prompt: str, answer: str, reference: str) -> float:
    """Run the metric once and return the float score (0..1).

    Wraps the assertion-free measurement path: we record the number even
    when it doesn't clear ``metric.threshold`` because the Pareto report
    is comparative, not pass/fail.
    """
    if not answer:
        return 0.0
    case = LLMTestCase(input=prompt, actual_output=answer, expected_output=reference)
    metric.measure(case)
    return float(metric.score or 0.0)


def _candidate_cost(response, spec) -> float:
    """Estimate per-call cost from the provider's usage telemetry + pricing prior."""
    usage = response.usage or {}
    prompt_tokens = usage.get('prompt_tokens', usage.get('prompt_eval_count', 0.0))
    completion_tokens = usage.get('completion_tokens', usage.get('eval_count', 0.0))
    return (
        (float(prompt_tokens) / 1000.0) * spec.pricing['input_per_1k']
        + (float(completion_tokens) / 1000.0) * spec.pricing['output_per_1k']
    )


@pytest.mark.eval
def test_emit_pareto_report(correctness_metric):
    """One test, full run. Emits a JSON report to an ignored directory.

    Asserts only that *something* was produced for each case. Per-prompt
    pass/fail is in the regular DeepEval suite; the Pareto report is a
    diagnostic, not a regression gate.
    """
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _REPORT_DIR / f'pareto-{_now_stamp()}.json'

    report: dict[str, Any] = {
        'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'dataset_size': len(_DATASET),
        'cases': [],
    }

    registry = get_registry()

    for case in _DATASET:
        prompt = case['prompt']
        reference = case['expected_output']

        # Run the full fusion pipeline.
        fused = asyncio.get_event_loop().run_until_complete(
            run_roitelet_chat(ChatRequest(prompt=prompt, preferences=RouterPreferences()))
        )
        fused_score = _score(correctness_metric, prompt, fused.synthesis.content, reference)
        fused_cost = sum(r.cost_usd or 0.0 for r in fused.responses)
        fused_latency = max((r.latency_s for r in fused.responses), default=0.0)

        # Score each candidate the pipeline actually called, in isolation.
        # Comparing those isolated scores against the fused score is the
        # measurement: did fusion beat its own best single candidate?
        per_candidate: list[dict[str, Any]] = []
        for response in fused.responses:
            spec = registry.get(response.model_id)
            single_score = _score(correctness_metric, prompt, response.content, reference)
            per_candidate.append({
                'model_id': response.model_id,
                'score': single_score,
                'cost_usd': _candidate_cost(response, spec),
                'latency_s': response.latency_s,
                'errored': bool(response.error),
            })

        best_single = max(per_candidate, key=lambda c: c['score'], default=None)
        delta_vs_best = fused_score - (best_single['score'] if best_single else 0.0)

        case_report = {
            'id': case['id'],
            'category': case['category'],
            'fused': {
                'score': fused_score,
                'cost_usd': fused_cost,
                'latency_s': fused_latency,
                'winning_model_ids': fused.synthesis.winning_model_ids,
            },
            'candidates': per_candidate,
            'best_single_model_id': best_single['model_id'] if best_single else None,
            'best_single_score': best_single['score'] if best_single else 0.0,
            'fused_minus_best_single': delta_vs_best,
        }
        report['cases'].append(case_report)
        print(
            f'[{case["id"]}] fused={fused_score:.2f} '
            f'best_single={(best_single["score"] if best_single else 0):.2f} '
            f'delta={delta_vs_best:+.2f}',
            flush=True,
        )

        assert per_candidate, f'No candidates returned for {case["id"]}'

    # Aggregate roll-up — the Pareto signal.
    fused_scores = [c['fused']['score'] for c in report['cases']]
    best_single_scores = [c['best_single_score'] for c in report['cases']]
    fused_costs = [c['fused']['cost_usd'] for c in report['cases']]
    report['summary'] = {
        'mean_fused_score': sum(fused_scores) / max(len(fused_scores), 1),
        'mean_best_single_score': sum(best_single_scores) / max(len(best_single_scores), 1),
        'mean_fused_minus_best_single': (
            sum(c['fused_minus_best_single'] for c in report['cases'])
            / max(len(report['cases']), 1)
        ),
        'total_fused_cost_usd': sum(fused_costs),
    }
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'\nPareto report written: {out_path}')
    print(json.dumps(report['summary'], indent=2))

    # Sanity: at least one case must have produced a non-zero fused score.
    # If everything is zero the local judge / candidate pool is broken;
    # better to fail loudly than emit a meaningless report.
    assert any(s > 0 for s in fused_scores), (
        'All fused scores were zero — likely a broken Ollama / candidate pool. '
        f'Inspect {out_path} for per-case details.'
    )


# Re-exporting nothing — the test alone drives the run. Pytest discovers
# it via the `eval` marker (conftest.py auto-applies it to this folder).
__all__: list[str] = []
