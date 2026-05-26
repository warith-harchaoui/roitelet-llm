"""Answer-quality regression tests for the Roitelet synthesis layer.

Each prompt in ``dataset.json`` is run end-to-end through
``run_roitelet_chat``. The candidates the local judge sees become the
*retrieval context* for DeepEval's faithfulness check, and the curated
reference answer drives a `GEval` correctness rubric.

What this catches that the existing unit tests don't:

* the judge silently introducing facts no candidate produced (hallucination
  via fusion),
* a regression in the system prompt that makes synthesised answers less
  correct than the candidates,
* a model swap that degrades a whole capability category at once,
* an OSS-only (independence) configuration that drops fusion quality
  enough to matter — Roitelet's local-first value prop only holds if the
  delta against full-fleet is small.

These tests are slow (one full Roitelet turn per prompt) and require both
Ollama and the API keys for paid candidates. Run them on cadence:

    pip install -e .[eval]
    pytest -m eval -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# DeepEval is required — conftest.py importorskips early if it isn't installed.
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from core.pipeline import run_roitelet_chat
from core.schemas import ChatRequest, RouterPreferences

_DATASET = json.loads((Path(__file__).parent / 'dataset.json').read_text())


def _ids(case):
    return case['id']


@pytest.fixture
def correctness_metric(ollama_judge):
    """A GEval rubric grading factual correctness vs the reference answer.

    GEval lets us define what 'correct' means in natural language and
    delegates the grading itself to the local model. The rubric is short
    on purpose — long rubrics drift across runs.
    """
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


@pytest.fixture
def faithfulness_metric(ollama_judge):
    """Faithfulness checks that every claim in the synthesis is supported by
    at least one candidate response — i.e. the judge isn't inventing facts.

    The retrieval_context here is the list of candidate answers the judge
    actually saw; that is the only ground truth available at fusion time.
    """
    return FaithfulnessMetric(model=ollama_judge, threshold=0.7)


@pytest.fixture
def answer_relevancy_metric(ollama_judge):
    """Answer relevancy checks the synthesis actually addresses the prompt.

    Different failure mode from faithfulness: a synthesis can be perfectly
    grounded in the candidates and still drift onto an adjacent topic
    (e.g. answering "how do I reverse a string in Python?" with a long
    treatise on Python sequences). This metric catches that drift.
    """
    return AnswerRelevancyMetric(model=ollama_judge, threshold=0.7)


# Independence mode == OSS-only candidates (no paid APIs). Running the full
# dataset in this regime is the only way to know whether Roitelet's
# local-first claim survives contact with real prompts.
_PREFERENCE_MODES = [
    pytest.param(
        RouterPreferences(),
        id='default',
    ),
    pytest.param(
        RouterPreferences(independence=True),
        id='independence',
    ),
]


@pytest.mark.parametrize('case', _DATASET, ids=_ids)
async def test_synthesis_is_correct(case, correctness_metric):
    """The synthesised answer must be factually consistent with the reference."""
    response = await run_roitelet_chat(
        ChatRequest(prompt=case['prompt'], preferences=RouterPreferences()),
    )
    test_case = LLMTestCase(
        input=case['prompt'],
        actual_output=response.synthesis.content,
        expected_output=case['expected_output'],
    )
    correctness_metric.measure(test_case)
    assert correctness_metric.is_successful(), (
        f"Correctness failed for {case['id']} "
        f"(score={correctness_metric.score:.2f}): {correctness_metric.reason}"
    )


@pytest.mark.parametrize('preferences', _PREFERENCE_MODES)
@pytest.mark.parametrize('case', _DATASET, ids=_ids)
async def test_synthesis_is_faithful_to_candidates(case, preferences, faithfulness_metric):
    """Every claim in the synthesis must be traceable to a candidate response.

    Parametrised over default-vs-independence preferences so the report
    surfaces the local-first quality delta directly.
    """
    response = await run_roitelet_chat(
        ChatRequest(prompt=case['prompt'], preferences=preferences),
    )
    candidate_texts: list[str] = [
        r.content for r in response.responses if r.content and not r.error
    ]
    if not candidate_texts:
        pytest.skip('All candidates errored — nothing to compare faithfulness against.')

    test_case = LLMTestCase(
        input=case['prompt'],
        actual_output=response.synthesis.content,
        retrieval_context=candidate_texts,
    )
    faithfulness_metric.measure(test_case)
    assert faithfulness_metric.is_successful(), (
        f"Faithfulness failed for {case['id']} "
        f"(score={faithfulness_metric.score:.2f}): {faithfulness_metric.reason}"
    )


@pytest.mark.parametrize('case', _DATASET, ids=_ids)
async def test_synthesis_is_relevant(case, answer_relevancy_metric):
    """The synthesised answer must actually address what the prompt asked.

    No reference answer needed: the metric grades the output against the
    input alone. Catches topic drift the correctness rubric misses when
    the drift happens to land on something *also* factually correct.
    """
    response = await run_roitelet_chat(
        ChatRequest(prompt=case['prompt'], preferences=RouterPreferences()),
    )
    test_case = LLMTestCase(
        input=case['prompt'],
        actual_output=response.synthesis.content,
    )
    answer_relevancy_metric.measure(test_case)
    assert answer_relevancy_metric.is_successful(), (
        f"Relevancy failed for {case['id']} "
        f"(score={answer_relevancy_metric.score:.2f}): "
        f"{answer_relevancy_metric.reason}"
    )
