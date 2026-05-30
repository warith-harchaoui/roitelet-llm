"""End-to-end pipeline tests.

The full ``run_roitelet_chat`` orchestration with the two heavy seams
(provider clients and the local judge) replaced by deterministic doubles.

Four story-level tests:

1. Golden path — a chat turn end-to-end, persists conversation +
   telemetry + Elo with the right deltas, then a follow-up reuses
   the same conversation id.
2. Independence mode filters remote candidates out before fan-out.
3. Partial failure — one candidate errors, the judge still runs on
   the survivors and the failed response is still in telemetry.
4. All-fail — when *every* candidate errors the pipeline raises
   ``AllCandidatesFailedError`` and the judge is NOT invoked.

The judge-fallback contract (silent judge → top candidate verbatim,
no winners recorded) is the fifth test — it lives next to the others
because it concerns the same orchestration surface.
"""

from __future__ import annotations

import copy

import pytest

from core.schemas import ChatRequest, ModelResponse, RouterPreferences, SynthesisResult


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect storage + registry to a per-test ``tmp_path``."""
    import core.config as config_mod
    import core.registry as registry_mod
    import core.storage as storage_mod

    monkeypatch.setattr(storage_mod, 'get_settings', config_mod.get_settings)

    fresh_storage = storage_mod.StorageManager()
    fresh_storage.root = tmp_path
    fresh_storage.conversations_dir = tmp_path / 'conversations'
    fresh_storage.telemetry_dir = tmp_path / 'telemetry'
    fresh_storage.runtime_dir = tmp_path / 'runtime'
    fresh_storage.cache_dir = tmp_path / 'cache'
    for d in (
        fresh_storage.conversations_dir, fresh_storage.telemetry_dir,
        fresh_storage.runtime_dir, fresh_storage.cache_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(storage_mod, 'get_storage', lambda: fresh_storage)

    registry_singleton = registry_mod.get_registry()
    original_path = registry_singleton.elo_path
    original_state = copy.deepcopy(registry_singleton.elo_state)
    registry_singleton.elo_path = tmp_path / 'runtime' / 'elo_state.json'
    registry_singleton.elo_state = {}

    try:
        yield {'storage': fresh_storage, 'registry': registry_singleton}
    finally:
        registry_singleton.elo_path = original_path
        registry_singleton.elo_state = original_state


def _provider_of(model_id: str) -> str:
    return model_id.split('/', 1)[0] if '/' in model_id else 'ollama'


def _fake_provider(failures: set[str] | None = None):
    """Build a fake provider client; optionally mark some model ids as failing."""
    failures = failures or set()

    class _Client:
        async def generate(self, *, model_id, messages):
            if model_id in failures:
                return ModelResponse(
                    model_id=model_id, provider=_provider_of(model_id),
                    content='', latency_s=0.0, usage={},
                    error='simulated provider failure',
                )
            return ModelResponse(
                model_id=model_id, provider=_provider_of(model_id),
                content=f'Mock answer from {model_id}.', latency_s=0.05,
                usage={'prompt_tokens': 12.0, 'completion_tokens': 24.0},
            )

    return lambda *_args, **_kw: _Client()


def _fake_judge(winner_index: int = 0):
    """Crown a specific candidate index."""
    async def _judge(prompt, responses):
        idx = min(winner_index, max(0, len(responses) - 1))
        winner = responses[idx]
        return SynthesisResult(
            model_id='ollama/qwen3:8b', provider='ollama',
            content=f'Synthesized: {winner.content}',
            judge_summary=f'WINNERS: {idx + 1}',
            winning_model_ids=[winner.model_id],
        )
    return _judge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_golden_path_persists_state_and_continues_a_conversation(
    isolated_state, monkeypatch,
):
    """One turn end-to-end + a follow-up turn that re-uses the same id.

    Pins:
    * the ChatResponse shape (top_k=2 → two responses);
    * routing reflects prompt content (coding prompt → coding dominant);
    * conversation + telemetry records on disk match the returned ids;
    * winner's Elo grows on the dominant capability, losers lose Elo;
    * follow-up turns extend the same conversation document.
    """
    monkeypatch.setattr('core.pipeline.get_provider_client', _fake_provider())
    monkeypatch.setattr('core.pipeline.judge_and_synthesize', _fake_judge())

    from core.pipeline import run_roitelet_chat

    storage = isolated_state['storage']
    registry = isolated_state['registry']
    assert registry.elo_state == {}

    prompt = 'Write a Python function to compute Fibonacci numbers.'
    first = await run_roitelet_chat(
        ChatRequest(prompt=prompt, preferences=RouterPreferences()),
    )

    # ChatResponse shape.
    assert len(first.responses) == 2  # default top_k=2
    assert first.synthesis.content.startswith('Synthesized:')

    # Routing reflects prompt content.
    dominant = max(first.router.categories.items(), key=lambda kv: kv[1])[0]
    assert dominant == 'coding'

    # Conversation persisted with both messages + metadata.
    convo = storage.get_conversation(first.conversation_id)
    assert [m.role for m in convo.messages] == ['user', 'assistant']
    meta = convo.messages[1].metadata
    assert {'router', 'responses', 'synthesis', 'total_latency_s'} <= set(meta.keys())
    assert meta['total_latency_s'] >= 0.0

    # Telemetry persisted with the right id + shadow pool.
    records = storage.list_telemetry()
    assert len(records) == 1 and records[0].record_id == first.telemetry_id
    assert len(records[0].shadow_reference_model_ids) >= 2

    # Elo: winner gained on coding; every loser lost globally.
    winner_id = first.synthesis.winning_model_ids[0]
    assert registry.elo_state[winner_id]['coding'] > 0
    for r in first.responses:
        if r.model_id != winner_id:
            assert registry.elo_state.get(r.model_id, {}).get('global', 0.0) < 0.0

    # Follow-up turn re-uses the conversation id.
    second = await run_roitelet_chat(
        ChatRequest(
            prompt='Follow-up question.',
            conversation_id=first.conversation_id,
            preferences=RouterPreferences(),
        ),
    )
    assert second.conversation_id == first.conversation_id
    convo = storage.get_conversation(first.conversation_id)
    assert [m.role for m in convo.messages] == ['user', 'assistant', 'user', 'assistant']
    assert len(storage.list_telemetry()) == 2


async def test_independence_mode_keeps_only_local_candidates(
    isolated_state, monkeypatch,
):
    """The privacy gate that matters most: ``independence=True`` must
    strip every remote candidate out before fan-out."""
    monkeypatch.setattr('core.pipeline.get_provider_client', _fake_provider())
    monkeypatch.setattr('core.pipeline.judge_and_synthesize', _fake_judge())

    from core.pipeline import run_roitelet_chat
    response = await run_roitelet_chat(
        ChatRequest(
            prompt='Help me refactor this Python module.',
            preferences=RouterPreferences(independence=True),
        ),
    )
    assert all(mid.startswith('ollama/') for mid in response.router.selected_model_ids)


async def test_partial_failure_synthesises_from_survivors_and_records_the_failure(
    isolated_state, monkeypatch,
):
    """If one candidate errors, the judge sees only the survivors but the
    failed response is still in telemetry (we never silently hide failures)."""
    from core.router import RoiteletRouter

    prompt = 'Write a Python function to compute Fibonacci numbers.'
    failing_id = RoiteletRouter().route(
        prompt, RouterPreferences(), top_k=3,
    ).selected_model_ids[0]

    monkeypatch.setattr(
        'core.pipeline.get_provider_client',
        _fake_provider(failures={failing_id}),
    )

    judged: dict = {}

    async def _spy(prompt_arg, responses):
        judged['responses'] = list(responses)
        return SynthesisResult(
            model_id='ollama/qwen3:8b', provider='ollama',
            content='Synthesized from survivors.', judge_summary='WINNERS: 1',
            winning_model_ids=[responses[0].model_id],
        )

    monkeypatch.setattr('core.pipeline.judge_and_synthesize', _spy)

    from core.pipeline import run_roitelet_chat
    response = await run_roitelet_chat(
        ChatRequest(prompt=prompt, preferences=RouterPreferences(), top_k=3),
    )

    # All three candidates show up on the response (including the failed one).
    assert len(response.responses) == 3
    failed = next(r for r in response.responses if r.model_id == failing_id)
    assert failed.error == 'simulated provider failure'
    assert failed.content == ''

    # The judge only saw the two surviving candidates.
    assert len(judged['responses']) == 2
    assert failing_id not in {r.model_id for r in judged['responses']}
    assert response.synthesis.content == 'Synthesized from survivors.'


async def test_all_candidates_failed_raises_without_calling_judge(
    isolated_state, monkeypatch,
):
    """Honesty contract: with every candidate failing, the pipeline must
    raise rather than synthesise from nothing, and the judge must not
    be invoked (so the Elo loop receives no spurious signal)."""
    from core.router import RoiteletRouter

    selected = RoiteletRouter().route(
        'Anything.', RouterPreferences(), top_k=3,
    ).selected_model_ids
    monkeypatch.setattr(
        'core.pipeline.get_provider_client', _fake_provider(failures=set(selected)),
    )

    judge_calls = {'n': 0}

    async def _exploding(*_a, **_kw):
        judge_calls['n'] += 1
        raise AssertionError('Judge must not run when every candidate failed.')

    monkeypatch.setattr('core.pipeline.judge_and_synthesize', _exploding)

    from core.pipeline import AllCandidatesFailedError, run_roitelet_chat
    with pytest.raises(AllCandidatesFailedError) as info:
        await run_roitelet_chat(
            ChatRequest(prompt='Anything.', preferences=RouterPreferences(), top_k=3),
        )
    assert judge_calls['n'] == 0
    assert len(info.value.responses) == 3
    assert all(r.error for r in info.value.responses)


async def test_silent_judge_surfaces_top_candidate_and_records_no_winners(
    isolated_state, monkeypatch,
):
    """If the local judge returns empty content, the user must still get
    a real answer (the top candidate verbatim), the summary must admit
    the judge was unreachable, and *no* winners must be recorded — the
    Elo loop receives no fabricated reward."""
    class _EmptyJudge:
        async def generate(self, *, model_id, messages):
            return ModelResponse(
                model_id=model_id, provider='ollama',
                content='', latency_s=0.0, usage={},
            )

    monkeypatch.setattr(
        'core.judge.get_provider_client', lambda _key: _EmptyJudge(),
    )

    from core.judge import judge_and_synthesize
    top = ModelResponse(
        model_id='openrouter/test/model-a', provider='openrouter',
        content='THE-TOP-CANDIDATE-ANSWER', latency_s=0.0, usage={},
    )
    runner_up = ModelResponse(
        model_id='openrouter/test/model-b', provider='openrouter',
        content='runner-up', latency_s=0.0, usage={},
    )
    result = await judge_and_synthesize('any prompt', [top, runner_up])
    assert result.content == 'THE-TOP-CANDIDATE-ANSWER'
    assert 'unavailable' in result.judge_summary.lower()
    assert result.winning_model_ids == []
