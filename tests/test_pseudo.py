"""Offline tests for :mod:`core.pseudo`.

Every visible behaviour worth pinning, in five carefully-written tests:

1. forward pass round-trips a typical PII payload;
2. forward pass fails closed on each of the five validation invariants —
   exhaustively, so a regression breaking any of them shows up;
3. restore pass handles both the trivial overlap-ordering case and
   the inflection case that triggers the LLM repair pass;
4. pipeline-level integration confirms the original prompt is what
   the conversation log persists (substitutes live in metadata).

The Ollama client is monkeypatched at the provider-factory seam so the
whole suite stays network-free.
"""

from __future__ import annotations

import json

import pytest

from core.pseudo import (
    PseudonymizationError,
    _has_orphan_substitutes,
    literal_restore,
    pseudonymize_prompt,
    restore_text,
)
from core.schemas import ChatMessage, ModelResponse, PIIMapping

# ---------------------------------------------------------------------------
# Test fixture: a stub Ollama client that returns canned bodies in order.
# ---------------------------------------------------------------------------


class _StubClient:
    """Stand-in for the Ollama client; yields canned reply bodies FIFO."""

    def __init__(self, body: str | list[str]) -> None:
        self._bodies = [body] if isinstance(body, str) else list(body)
        self.calls: list[dict] = []

    async def generate(self, model_id: str, messages: list[ChatMessage]) -> ModelResponse:
        self.calls.append({'model_id': model_id, 'messages': [m.model_dump() for m in messages]})
        content = self._bodies.pop(0) if self._bodies else ''
        return ModelResponse(
            model_id=model_id,
            provider=model_id.split('/', 1)[0],
            content=content,
            latency_s=0.01,
        )


@pytest.fixture
def patch_provider(monkeypatch):
    """Install a stub provider client and pin the resolved model id."""
    from core import pseudo as pseudo_mod

    def install(body: str | list[str]) -> _StubClient:
        client = _StubClient(body)
        monkeypatch.setattr(pseudo_mod, 'get_provider_client', lambda _p: client)

        class _Stub:
            local_synthesis_model = 'qwen3:8b'
            pseudo_model_id = ''

        class _Storage:
            def load_app_settings(self):
                return _Stub()

        monkeypatch.setattr(pseudo_mod._storage_mod, 'get_storage', lambda: _Storage())
        return client

    return install


def _forward_body(rewritten: str, mappings: list[dict]) -> str:
    """Serialise the JSON envelope the local model is expected to return."""
    return json.dumps({'pseudonymized_prompt': rewritten, 'mappings': mappings})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_forward_pass_round_trips_a_typical_pii_payload(patch_provider):
    """The wrapper validates whatever the model returns and exposes it
    on the audit. One representative case (name + place) suffices —
    every other PII category goes through the same code path.

    The empty-mapping case is also a legitimate outcome (a coding
    question shouldn't be pseudonymised), so we assert that too.
    """
    patch_provider(_forward_body(
        'Email Camille Lefèvre about the Toulouse meeting.',
        [
            {'original': 'Marie Dupont', 'substitute': 'Camille Lefèvre', 'kind': 'person_name'},
            {'original': 'Lyon', 'substitute': 'Toulouse', 'kind': 'place_name'},
        ],
    ))
    audit = await pseudonymize_prompt('Email Marie Dupont about the Lyon meeting.')
    assert audit.pseudonymized_prompt == 'Email Camille Lefèvre about the Toulouse meeting.'
    assert {m.kind for m in audit.mappings} == {'person_name', 'place_name'}
    assert audit.model_id.endswith('qwen3:8b')

    # Empty mapping is legitimate, not an error.
    patch_provider(_forward_body('How do I reverse a Python list in place?', []))
    audit = await pseudonymize_prompt('How do I reverse a Python list in place?')
    assert audit.mappings == []


async def test_forward_pass_fails_closed_on_every_validation_breach(patch_provider):
    """The safety contract: the unredacted prompt is never sent.

    Any of these breaches must raise :class:`PseudonymizationError`:

    * model returns non-JSON,
    * mapping refers to a substring not in the input,
    * substitute is missing from the rewritten prompt,
    * the original leaks through into the rewritten prompt,
    * mapping is a no-op (``original == substitute``).

    A fenced JSON envelope is the one *recoverable* breach — local
    models often wrap output in ```` ```json ```` despite the prompt
    saying not to. We tolerate that.
    """
    # Invalid JSON.
    patch_provider('definitely not json')
    with pytest.raises(PseudonymizationError, match='valid JSON'):
        await pseudonymize_prompt('hello')

    # Original not actually in the prompt.
    patch_provider(_forward_body(
        'hello stranger',
        [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
    ))
    with pytest.raises(PseudonymizationError, match='was in the prompt'):
        await pseudonymize_prompt('hello')

    # Substitute missing from the rewritten output.
    patch_provider(_forward_body(
        'hello stranger',
        [{'original': 'hello', 'substitute': 'salut', 'kind': 'person_name'}],
    ))
    with pytest.raises(PseudonymizationError, match='did not actually use'):
        await pseudonymize_prompt('hello stranger')

    # Original leaks into the rewritten prompt.
    patch_provider(_forward_body(
        'Marie also greets Camille.',
        [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
    ))
    with pytest.raises(PseudonymizationError, match='left the original'):
        await pseudonymize_prompt('Marie says hi.')

    # No-op mapping.
    patch_provider(_forward_body(
        'Marie says hi.',
        [{'original': 'Marie', 'substitute': 'Marie', 'kind': 'person_name'}],
    ))
    with pytest.raises(PseudonymizationError, match='no-op'):
        await pseudonymize_prompt('Marie says hi.')

    # ```json fences are tolerated — recoverable, not a breach.
    patch_provider(
        '```json\n'
        + _forward_body(
            'I am Camille.',
            [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
        )
        + '\n```'
    )
    audit = await pseudonymize_prompt('I am Marie.')
    assert audit.pseudonymized_prompt == 'I am Camille.'


async def test_restore_handles_overlap_then_inflection(patch_provider):
    """Restore is two stages and both have one subtle case worth pinning.

    * Stage 1 (literal): if substitute A is contained in substitute B,
      longer-first ordering is what stops the inner replace from
      corrupting the outer one ("Toulouse" inside "New Toulouse").
    * Stage 2 (LLM repair): when the synthesis judge inflects a
      multi-token substitute ("Camille Lefèvre" → "Mme Lefèvre")
      the literal pass leaves "Lefèvre" surviving; the repair pass
      fires exactly here, not on a single-token substitute that the
      literal pass would already have handled.
    """
    # Stage 1: overlap ordering.
    mappings = [
        PIIMapping(original='New York', substitute='New Toulouse', kind='place_name'),
        PIIMapping(original='Lyon', substitute='Toulouse', kind='place_name'),
    ]
    assert (
        literal_restore('I left New Toulouse and arrived at Toulouse.', mappings)
        == 'I left New York and arrived at Lyon.'
    )

    # Orphan-detection invariants: fires on a surviving distinctive token,
    # not when the token was in the original too.
    inflection_case = [PIIMapping(
        original='Marie Dupont', substitute='Camille Lefèvre', kind='person_name',
    )]
    assert _has_orphan_substitutes('Mme Lefèvre said hi.', inflection_case) is True

    shared_token_case = [PIIMapping(
        original='Marie Curie', substitute='Marie Sklodowska', kind='person_name',
    )]
    assert _has_orphan_substitutes('Marie was happy.', shared_token_case) is False

    # Stage 2: repair pass fires on the inflection case and the audit
    # records that it ran.
    client = patch_provider('Mme Dupont wrote back the next day.')
    out, repair_used = await restore_text(
        'Mme Lefèvre wrote back the next day.', inflection_case,
    )
    assert repair_used is True
    assert out == 'Mme Dupont wrote back the next day.'
    assert 'Camille Lefèvre → Marie Dupont' in client.calls[0]['messages'][1]['content']


async def test_pipeline_persists_original_prompt_and_returns_restored_answer(
    monkeypatch, tmp_path,
):
    """The persistence contract: the conversation log stores what the
    user typed, never the substituted version. Substitutes live only
    on ``metadata.pseudonymization`` so a future audit / reload reads
    naturally to the user while still proving what left the box.

    Every model-side seam is stubbed so the test stays hermetic; the
    real storage manager writes to ``tmp_path``."""
    from core import config as config_mod
    from core import pipeline as pipeline_mod
    from core import pseudo as pseudo_mod
    from core import storage as storage_mod
    from core.schemas import (
        ChatRequest,
        ModelResponse,
        RouterDecision,
        RouterPreferences,
        SynthesisResult,
    )

    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    from core.storage import get_storage as _get_storage
    config_mod.get_settings.cache_clear()
    _get_storage.cache_clear()

    # Pseudonymizer stub.
    client = _StubClient(_forward_body(
        'Email Camille about Toulouse.',
        [
            {'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'},
            {'original': 'Lyon', 'substitute': 'Toulouse', 'kind': 'place_name'},
        ],
    ))
    monkeypatch.setattr(pseudo_mod, 'get_provider_client', lambda _p: client)

    seen: list[str] = []

    async def fake_query_one(model_id: str, messages):
        seen.append(messages[0].content)
        return ModelResponse(model_id=model_id, provider='ollama',
                             content=f'Reply for {messages[0].content!r}', latency_s=0.01)

    async def fake_judge(prompt: str, responses):
        # The candidate side and the judge both see the substituted prompt.
        assert prompt == 'Email Camille about Toulouse.'
        return SynthesisResult(
            model_id='ollama/qwen3:8b', provider='ollama',
            content='Camille will receive an email about Toulouse.',
            judge_summary='', winning_model_ids=[], latency_s=0.01,
        )

    class _RouterStub:
        def route(self, prompt, preferences, top_k):
            seen.append(f'router:{prompt}')
            return RouterDecision(
                prompt=prompt, categories={'reasoning': 1.0}, candidates=[],
                selected_model_ids=['ollama/qwen3:8b'], reasoning=[],
            )

    class _RegistryStub:
        elo_state: dict = {}

        def update_elo(self, *_args, **_kw):
            return None

    monkeypatch.setattr(pipeline_mod, '_query_one', fake_query_one)
    monkeypatch.setattr(pipeline_mod, 'judge_and_synthesize', fake_judge)
    monkeypatch.setattr(pipeline_mod, 'get_router', lambda: _RouterStub())
    monkeypatch.setattr(pipeline_mod._registry_mod, 'get_registry', lambda: _RegistryStub())

    response = await pipeline_mod.run_roitelet_chat(
        ChatRequest(
            prompt='Email Marie about Lyon.',
            preferences=RouterPreferences(pseudonymize=True),
        ),
    )

    # Router and candidate both ran on the substituted prompt.
    assert 'router:Email Camille about Toulouse.' in seen
    # The user-visible answer has the originals back.
    assert response.synthesis.content == 'Marie will receive an email about Lyon.'
    # The audit is on the response.
    assert response.pseudonymization is not None
    assert {m.original for m in response.pseudonymization.mappings} == {'Marie', 'Lyon'}

    # Persistence contract: original prompt on the user message,
    # substitutes only in metadata.
    conv = storage_mod.get_storage().get_conversation(response.conversation_id)
    assert conv.messages[0].content == 'Email Marie about Lyon.'
    assert 'pseudonymization' in conv.messages[0].metadata
