"""End-to-end pipeline tests for Roitelet LLM.

Run with:
    pytest tests/test_pipeline.py -q

These tests exercise the full ``run_roitelet_chat`` orchestration with the
two heavy seams (provider clients and the local judge) replaced by
deterministic doubles. They verify routing, parallel inference, judging,
conversation persistence, telemetry round-tripping, and Elo updates.
"""

from __future__ import annotations

import copy

import pytest

from core.schemas import (
    ChatRequest,
    ModelResponse,
    RouterPreferences,
    SynthesisResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect storage + registry to a per-test ``tmp_path``.

    Post-D1 the pipeline reads the storage and registry singletons through
    ``core.storage.get_storage()`` and ``core.registry.get_registry()`` —
    so a single monkey-patch at the factory propagates to every importer
    instead of having to patch each module one-by-one.
    """
    import core.config as config_mod
    import core.registry as registry_mod
    import core.storage as storage_mod

    monkeypatch.setattr(storage_mod, 'get_settings', config_mod.get_settings)

    # --- Fresh storage instance scoped to tmp_path. ---
    fresh_storage = storage_mod.StorageManager()
    fresh_storage.root = tmp_path
    fresh_storage.conversations_dir = tmp_path / 'conversations'
    fresh_storage.telemetry_dir = tmp_path / 'telemetry'
    fresh_storage.runtime_dir = tmp_path / 'runtime'
    fresh_storage.cache_dir = tmp_path / 'cache'
    for directory in (
        fresh_storage.conversations_dir,
        fresh_storage.telemetry_dir,
        fresh_storage.runtime_dir,
        fresh_storage.cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_mod, 'get_storage', lambda: fresh_storage)

    # --- Registry Elo: reuse the real registry singleton but redirect its
    # on-disk path. Resetting elo_state to {} guarantees test independence. ---
    registry_singleton = registry_mod.get_registry()
    original_elo_path = registry_singleton.elo_path
    original_elo_state = copy.deepcopy(registry_singleton.elo_state)
    registry_singleton.elo_path = tmp_path / 'runtime' / 'elo_state.json'
    registry_singleton.elo_state = {}

    try:
        yield {
            'tmp_path': tmp_path,
            'storage': fresh_storage,
            'registry': registry_singleton,
        }
    finally:
        registry_singleton.elo_path = original_elo_path
        registry_singleton.elo_state = original_elo_state


def _make_fake_provider(failures: set[str] | None = None):
    """Build a fake provider-client factory.

    Parameters
    ----------
    failures:
        Set of model ids that should return a response with ``error`` set
        and empty content. Used to test the partial-failure path.
    """
    failures = failures or set()

    class _FakeClient:
        async def generate(self, *, model_id, messages):
            if model_id in failures:
                return ModelResponse(
                    model_id=model_id,
                    provider=_provider_of(model_id),
                    content='',
                    latency_s=0.0,
                    usage={'prompt_tokens': 0.0, 'completion_tokens': 0.0},
                    error='simulated provider failure',
                )
            return ModelResponse(
                model_id=model_id,
                provider=_provider_of(model_id),
                content=f'Mock answer from {model_id}.',
                latency_s=0.05,
                usage={'prompt_tokens': 12.0, 'completion_tokens': 24.0},
            )

    def factory(_provider: str, model_id: str | None = None):
        return _FakeClient()

    return factory


def _provider_of(model_id: str) -> str:
    """Infer provider key from the model id prefix."""
    return model_id.split('/', 1)[0] if '/' in model_id else 'ollama'


def _make_fake_judge(winner_index: int = 0):
    """Build a fake judge that crowns one candidate.

    Parameters
    ----------
    winner_index:
        0-based index of the response to mark as winner. Falls back to 0
        if the list is shorter than expected.
    """

    async def _judge(prompt, responses):
        idx = min(winner_index, max(0, len(responses) - 1))
        winner = responses[idx]
        return SynthesisResult(
            model_id='ollama/qwen3:8b',
            provider='ollama',
            content=f'Synthesized: {winner.content}',
            judge_summary=f'Candidate {idx + 1} wins.\nWINNERS: {idx + 1}',
            winning_model_ids=[winner.model_id],
        )

    return _judge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunRoiteletChat:
    """End-to-end pipeline behaviour with mocked provider and judge."""

    async def test_golden_path_persists_conversation_and_telemetry(
        self, isolated_state, monkeypatch
    ):
        """A single coding prompt round-trips through router, providers,
        judge, conversation log, and telemetry log."""
        monkeypatch.setattr(
            'core.pipeline.get_provider_client', _make_fake_provider()
        )
        monkeypatch.setattr(
            'core.pipeline.judge_and_synthesize', _make_fake_judge()
        )

        from core.pipeline import run_roitelet_chat

        prompt = 'Write a Python function to compute Fibonacci numbers.'
        request = ChatRequest(prompt=prompt, preferences=RouterPreferences())
        response = await run_roitelet_chat(request)

        # --- ChatResponse shape ---
        assert response.conversation_id
        assert response.telemetry_id
        assert len(response.responses) == 2, 'Default top_k=2 selects two models'
        assert response.synthesis.content.startswith('Synthesized:')

        # --- Routing reflects the prompt ---
        dominant = max(response.router.categories.items(), key=lambda kv: kv[1])[0]
        assert dominant == 'coding', f'Coding prompt should dominate, got {dominant}'
        assert len(response.router.selected_model_ids) == 2

        # --- Conversation persisted with user + assistant messages ---
        storage = isolated_state['storage']
        convo = storage.get_conversation(response.conversation_id)
        assert convo is not None
        assert [m.role for m in convo.messages] == ['user', 'assistant']
        assert convo.messages[0].content == prompt
        assert convo.messages[1].content == response.synthesis.content

        # The assistant message preserves full router + responses +
        # synthesis payload, plus the end-to-end pipeline wall-clock so
        # the GUI can render the user-perceived latency.
        meta = convo.messages[1].metadata
        assert {'router', 'responses', 'synthesis', 'total_latency_s'} <= set(meta.keys())
        assert len(meta['responses']) == 2
        assert isinstance(meta['total_latency_s'], (int, float))
        assert meta['total_latency_s'] >= 0.0

        # --- Telemetry persisted ---
        records = storage.list_telemetry()
        assert len(records) == 1
        record = records[0]
        assert record.record_id == response.telemetry_id
        assert record.conversation_id == response.conversation_id
        assert record.prompt == prompt
        assert len(record.model_responses) == 2
        # Shadow pool is at least max(top_k, 5) — pipeline records the
        # wider pool so post-hoc analysis can replay the road not taken.
        assert len(record.shadow_reference_model_ids) >= 2

    async def test_elo_winner_global_score_increases(
        self, isolated_state, monkeypatch
    ):
        """The winning model's rolling Elo adjustment must grow after a turn."""
        monkeypatch.setattr(
            'core.pipeline.get_provider_client', _make_fake_provider()
        )
        monkeypatch.setattr(
            'core.pipeline.judge_and_synthesize', _make_fake_judge(winner_index=0)
        )

        from core.pipeline import run_roitelet_chat

        registry = isolated_state['registry']
        # Sanity: snapshot fixture starts with an empty Elo state.
        assert registry.elo_state == {}

        request = ChatRequest(
            prompt='Write a Python function to compute Fibonacci numbers.',
            preferences=RouterPreferences(),
        )
        response = await run_roitelet_chat(request)

        winner_id = response.synthesis.winning_model_ids[0]
        loser_ids = [
            r.model_id for r in response.responses if r.model_id != winner_id
        ]

        assert registry.elo_state.get(winner_id, {}).get('global', 0.0) > 0.0
        for loser in loser_ids:
            assert registry.elo_state.get(loser, {}).get('global', 0.0) < 0.0

        # Capability-specific deltas land on the dominant capability of the prompt.
        # Coding-dominant prompt → winner should gain 'coding' Elo.
        assert registry.elo_state[winner_id].get('coding', 0.0) > 0.0

    async def test_conversation_continuation_reuses_same_id(
        self, isolated_state, monkeypatch
    ):
        """Passing ``conversation_id`` appends to the existing flight."""
        monkeypatch.setattr(
            'core.pipeline.get_provider_client', _make_fake_provider()
        )
        monkeypatch.setattr(
            'core.pipeline.judge_and_synthesize', _make_fake_judge()
        )

        from core.pipeline import run_roitelet_chat

        first = await run_roitelet_chat(
            ChatRequest(prompt='First question.', preferences=RouterPreferences())
        )
        second = await run_roitelet_chat(
            ChatRequest(
                prompt='Follow-up question.',
                conversation_id=first.conversation_id,
                preferences=RouterPreferences(),
            )
        )

        assert second.conversation_id == first.conversation_id

        storage = isolated_state['storage']
        convo = storage.get_conversation(first.conversation_id)
        assert convo is not None
        # Two turns → 4 messages (user, assistant, user, assistant).
        assert [m.role for m in convo.messages] == [
            'user', 'assistant', 'user', 'assistant'
        ]
        assert convo.messages[0].content == 'First question.'
        assert convo.messages[2].content == 'Follow-up question.'

        # Telemetry rows are independent — two records, two ids.
        records = storage.list_telemetry()
        assert len(records) == 2
        assert {r.record_id for r in records} == {
            first.telemetry_id,
            second.telemetry_id,
        }

    async def test_independence_mode_selects_only_local_models(
        self, isolated_state, monkeypatch
    ):
        """``preferences.independence=True`` filters out remote candidates."""
        monkeypatch.setattr(
            'core.pipeline.get_provider_client', _make_fake_provider()
        )
        monkeypatch.setattr(
            'core.pipeline.judge_and_synthesize', _make_fake_judge()
        )

        from core.pipeline import run_roitelet_chat

        request = ChatRequest(
            prompt='Help me refactor this Python module.',
            preferences=RouterPreferences(independence=True),
        )
        response = await run_roitelet_chat(request)

        for model_id in response.router.selected_model_ids:
            assert model_id.startswith('ollama/'), (
                f'Independence mode must keep only local models, got {model_id}'
            )

    async def test_partial_provider_failure_still_synthesizes(
        self, isolated_state, monkeypatch
    ):
        """If one provider errors, the judge still runs on the remaining
        valid responses and the pipeline returns a synthesis."""
        # Decide failure target after a dry-run route so we hit a real selected id.
        from core.router import RoiteletRouter

        prompt = 'Write a Python function to compute Fibonacci numbers.'
        dry_run = RoiteletRouter().route(prompt, RouterPreferences(), top_k=3)
        failing_id = dry_run.selected_model_ids[0]

        monkeypatch.setattr(
            'core.pipeline.get_provider_client',
            _make_fake_provider(failures={failing_id}),
        )

        captured = {}

        async def _spy_judge(prompt_arg, responses):
            captured['responses'] = list(responses)
            # Always crown index 0 of whatever the judge actually receives.
            return SynthesisResult(
                model_id='ollama/qwen3:8b',
                provider='ollama',
                content='Synthesized from survivors.',
                judge_summary='WINNERS: 1',
                winning_model_ids=[responses[0].model_id],
            )

        monkeypatch.setattr('core.pipeline.judge_and_synthesize', _spy_judge)

        from core.pipeline import run_roitelet_chat

        # Pin top_k=3 so the test still exercises a 3-way fan-out
        # (one failure + two survivors); the default is K=2 for
        # production but a wider fan-out keeps the assertion below
        # meaningful.
        response = await run_roitelet_chat(
            ChatRequest(prompt=prompt, preferences=RouterPreferences(), top_k=3)
        )

        # All three model responses are recorded in the final payload
        # (including the failed one — telemetry must not hide failures).
        assert len(response.responses) == 3
        failed = [r for r in response.responses if r.model_id == failing_id][0]
        assert failed.error == 'simulated provider failure'
        assert failed.content == ''

        # The judge only saw the two valid responses.
        assert len(captured['responses']) == 2
        assert failing_id not in {r.model_id for r in captured['responses']}
        assert all(r.content for r in captured['responses'])

        # Synthesis still came back.
        assert response.synthesis.content == 'Synthesized from survivors.'

    async def test_all_candidates_failed_raises_no_judge_call(
        self, isolated_state, monkeypatch
    ):
        """When every model fails, the pipeline raises rather than fabricating
        a fake synthesis — and crucially, the judge is NOT invoked."""
        from core.router import RoiteletRouter

        prompt = 'Anything.'
        dry_run = RoiteletRouter().route(prompt, RouterPreferences(), top_k=3)
        # Fail all selected ids.
        monkeypatch.setattr(
            'core.pipeline.get_provider_client',
            _make_fake_provider(failures=set(dry_run.selected_model_ids)),
        )

        judge_calls = {'count': 0}

        async def _exploding_judge(*_a, **_kw):
            judge_calls['count'] += 1
            raise AssertionError('Judge must not run when every candidate failed.')

        monkeypatch.setattr('core.pipeline.judge_and_synthesize', _exploding_judge)

        from core.pipeline import AllCandidatesFailedError, run_roitelet_chat

        with pytest.raises(AllCandidatesFailedError) as info:
            await run_roitelet_chat(
                ChatRequest(prompt=prompt, preferences=RouterPreferences(), top_k=3)
            )

        assert judge_calls['count'] == 0
        # The error carries the failed responses so the API layer can detail them.
        assert len(info.value.responses) == 3
        assert all(r.error for r in info.value.responses)


class TestJudgeFallback:
    """When the local judge returns empty content, the synthesis result
    must surface the top candidate verbatim and admit the judge was
    unreachable — never echo the meta-message back to the user as content."""

    async def test_empty_judge_returns_top_candidate_verbatim(self, isolated_state, monkeypatch):
        class _EmptyJudgeClient:
            async def generate(self, *, model_id, messages):
                return ModelResponse(
                    model_id=model_id,
                    provider='ollama',
                    content='',  # judge unreachable / blank completion
                    latency_s=0.0,
                    usage={},
                )

        # Real judge call path; fake only the local-judge provider.
        monkeypatch.setattr(
            'core.judge.get_provider_client', lambda _key: _EmptyJudgeClient(),
        )

        from core.judge import judge_and_synthesize

        top = ModelResponse(
            model_id='openrouter/test/model-a',
            provider='openrouter',
            content='THE-TOP-CANDIDATE-ANSWER',
            latency_s=0.0,
            usage={},
        )
        runner_up = ModelResponse(
            model_id='openrouter/test/model-b',
            provider='openrouter',
            content='runner-up answer',
            latency_s=0.0,
            usage={},
        )

        result = await judge_and_synthesize('any prompt', [top, runner_up])

        # The user must get the real top candidate, not a meta-message.
        assert result.content == 'THE-TOP-CANDIDATE-ANSWER'
        # The judge_summary must admit the judge was unreachable.
        assert 'unavailable' in result.judge_summary.lower()
        # No winners recorded: a silent judge MUST NOT feed Elo with a
        # fabricated reward. The fallback content is best-effort, the
        # reward signal is honest.
        assert result.winning_model_ids == []


class TestEstimateCost:
    """Lock in the Pydantic-enforced contract for ModelResponse.usage.

    ``usage`` is typed ``Dict[str, float]`` (see ``core/schemas.py``). Pydantic
    coerces numeric strings to float on construction, which is the only reason
    ``_estimate_cost`` can safely arithmetic on the dict values. If a future
    schema change loosens the type, these tests fail loudly.
    """

    def test_numeric_string_usage_is_coerced_to_float(self):
        from core.pipeline import _estimate_cost

        response = ModelResponse(
            model_id='ollama/qwen3:8b',
            provider='ollama',
            content='ok',
            latency_s=0.0,
            usage={'prompt_tokens': '12', 'completion_tokens': '24'},
        )
        # Pydantic must have coerced the strings — otherwise the arithmetic
        # inside _estimate_cost would raise TypeError.
        assert isinstance(response.usage['prompt_tokens'], float)
        assert _estimate_cost(response.model_id, response) == 0.0  # local pricing is 0
