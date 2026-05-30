"""Offline unit tests for :mod:`core.pseudo`.

The Ollama client is monkeypatched at the provider-factory seam so the
whole suite stays network-free. Each test drives the public
:func:`pseudonymize_prompt` / :func:`restore_text` API end-to-end with
a hand-written stub reply, asserting the fail-closed contract and the
PII taxonomy coverage.

Why this layer instead of an end-to-end DeepEval pass:

* the eval pass (``tests/eval/test_pseudo_quality.py``) measures
  *quality*; this pass measures *contract*. They catch different
  classes of regression.
* a unit test that asserts e.g. "an IP address actually round-trips"
  is meaningful only at the validation seam — the eval suite would
  attribute any miss to the model rather than the wrapper.
"""

from __future__ import annotations

import json
from typing import Any

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
# Helpers
# ---------------------------------------------------------------------------


class _StubClient:
    """Pluggable Ollama-shape client whose .generate(...) returns a fixed body.

    The body can be a string (one-shot) or a list of strings consumed
    FIFO (for two-stage forward + repair scenarios).
    """

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
    """Return a closure that installs a stub provider client for one test."""
    from core import pseudo as pseudo_mod

    def install(body: str | list[str]) -> _StubClient:
        client = _StubClient(body)
        monkeypatch.setattr(
            pseudo_mod, 'get_provider_client', lambda _provider: client
        )
        return client

    return install


@pytest.fixture
def patch_settings(monkeypatch):
    """Pin the settings-resolved model id so tests don't depend on disk state."""
    from core import pseudo as pseudo_mod

    class _Stub:
        local_synthesis_model = 'qwen3:8b'
        pseudo_model_id = ''

    class _Storage:
        def load_app_settings(self):
            return _Stub()

    monkeypatch.setattr(pseudo_mod._storage_mod, 'get_storage', lambda: _Storage())


def _forward_body(rewritten: str, mappings: list[dict]) -> str:
    return json.dumps({'pseudonymized_prompt': rewritten, 'mappings': mappings})


# ---------------------------------------------------------------------------
# Forward-pass tests
# ---------------------------------------------------------------------------


class TestForwardHappyPath:
    """Each PII category, end-to-end through the stub."""

    @pytest.mark.parametrize(
        ('prompt', 'rewritten', 'mappings'),
        [
            (
                'Email Marie Dupont about the Lyon meeting.',
                'Email Camille Lefèvre about the Toulouse meeting.',
                [
                    {'original': 'Marie Dupont', 'substitute': 'Camille Lefèvre', 'kind': 'person_name'},
                    {'original': 'Lyon', 'substitute': 'Toulouse', 'kind': 'place_name'},
                ],
            ),
            (
                'My number is +33 6 12 34 56 78 and my email is marie@orange.fr.',
                'My number is +33 6 98 76 54 32 10 and my email is camille@orange.fr.',
                [
                    {'original': '+33 6 12 34 56 78', 'substitute': '+33 6 98 76 54 32 10', 'kind': 'phone'},
                    {'original': 'marie@orange.fr', 'substitute': 'camille@orange.fr', 'kind': 'email'},
                ],
            ),
            (
                'Card 4242 4242 4242 4242 expires 12/29.',
                'Card 5500 0000 0000 0004 expires 12/29.',
                [
                    {'original': '4242 4242 4242 4242', 'substitute': '5500 0000 0000 0004', 'kind': 'financial_id'},
                ],
            ),
            (
                'My SSN is 123-45-6789.',
                'My SSN is 987-65-4321.',
                [
                    {'original': '123-45-6789', 'substitute': '987-65-4321', 'kind': 'national_id'},
                ],
            ),
            (
                'Server 192.168.1.42 is down.',
                'Server 10.0.5.117 is down.',
                [
                    {'original': '192.168.1.42', 'substitute': '10.0.5.117', 'kind': 'ip_address'},
                ],
            ),
            (
                'I work at Acme Corp and report to Jane Doe, the CEO.',
                'I work at Globex Industries and report to Liam Carter, the CEO.',
                [
                    {'original': 'Acme Corp', 'substitute': 'Globex Industries', 'kind': 'organization'},
                    {'original': 'Jane Doe', 'substitute': 'Liam Carter', 'kind': 'person_name'},
                ],
            ),
        ],
        ids=[
            'name+place',
            'phone+email',
            'credit-card',
            'ssn',
            'ip-address',
            'org+name',
        ],
    )
    async def test_forward_round_trips(self, patch_provider, patch_settings, prompt, rewritten, mappings):
        patch_provider(_forward_body(rewritten, mappings))
        audit = await pseudonymize_prompt(prompt)
        assert audit.pseudonymized_prompt == rewritten
        assert len(audit.mappings) == len(mappings)
        assert {m.kind for m in audit.mappings} == {m['kind'] for m in mappings}
        assert audit.model_id.endswith('qwen3:8b')

    async def test_no_pii_means_empty_mappings(self, patch_provider, patch_settings):
        """The model is allowed to find nothing; that is a legitimate outcome."""
        prompt = 'How do I reverse a Python list in place?'
        patch_provider(_forward_body(prompt, []))
        audit = await pseudonymize_prompt(prompt)
        assert audit.mappings == []
        assert audit.pseudonymized_prompt == prompt


class TestForwardFailClosed:
    """Every validation must abort the turn, never silently let the prompt through."""

    async def test_invalid_json_raises(self, patch_provider, patch_settings):
        patch_provider('definitely not json')
        with pytest.raises(PseudonymizationError, match='valid JSON'):
            await pseudonymize_prompt('hello')

    async def test_original_not_in_prompt_raises(self, patch_provider, patch_settings):
        body = _forward_body(
            'hello stranger',
            [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
        )
        patch_provider(body)
        with pytest.raises(PseudonymizationError, match='was in the prompt'):
            await pseudonymize_prompt('hello')

    async def test_substitute_not_in_rewritten_raises(self, patch_provider, patch_settings):
        body = _forward_body(
            'hello stranger',
            [{'original': 'hello', 'substitute': 'salut', 'kind': 'person_name'}],
        )
        patch_provider(body)
        with pytest.raises(PseudonymizationError, match='did not actually use'):
            await pseudonymize_prompt('hello stranger')

    async def test_original_leaks_into_rewritten_raises(self, patch_provider, patch_settings):
        body = _forward_body(
            'Marie also greets Camille.',
            [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
        )
        patch_provider(body)
        with pytest.raises(PseudonymizationError, match='left the original'):
            await pseudonymize_prompt('Marie says hi.')

    async def test_no_op_mapping_raises(self, patch_provider, patch_settings):
        body = _forward_body(
            'Marie says hi.',
            [{'original': 'Marie', 'substitute': 'Marie', 'kind': 'person_name'}],
        )
        patch_provider(body)
        with pytest.raises(PseudonymizationError, match='no-op'):
            await pseudonymize_prompt('Marie says hi.')

    async def test_empty_response_raises(self, patch_provider, patch_settings):
        patch_provider('')
        with pytest.raises(PseudonymizationError, match='empty'):
            await pseudonymize_prompt('hello')

    async def test_invalid_kind_raises(self, patch_provider, patch_settings):
        body = _forward_body(
            'hello stranger',
            [{'original': 'hello', 'substitute': 'salut', 'kind': 'not_a_real_kind'}],
        )
        patch_provider(body)
        # Pydantic Literal validation surfaces inside our PII parse step.
        with pytest.raises(PseudonymizationError, match='invalid mapping entry'):
            await pseudonymize_prompt('hello stranger')

    async def test_fenced_json_is_tolerated(self, patch_provider, patch_settings):
        """Models that wrap output in ```json fences despite the prompt must still work."""
        prompt = 'I am Marie.'
        body = '```json\n' + _forward_body(
            'I am Camille.',
            [{'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'}],
        ) + '\n```'
        patch_provider(body)
        audit = await pseudonymize_prompt(prompt)
        assert audit.pseudonymized_prompt == 'I am Camille.'


# ---------------------------------------------------------------------------
# Reverse-pass tests
# ---------------------------------------------------------------------------


class TestLiteralRestore:
    def test_simple_swap(self):
        mappings = [
            PIIMapping(original='Marie Dupont', substitute='Camille Lefèvre', kind='person_name'),
            PIIMapping(original='Lyon', substitute='Toulouse', kind='place_name'),
        ]
        text = 'Camille Lefèvre will travel to Toulouse on Monday.'
        assert literal_restore(text, mappings) == 'Marie Dupont will travel to Lyon on Monday.'

    def test_longer_substitute_wins(self):
        """If one substitute contains another, the longer match must replace first."""
        mappings = [
            PIIMapping(original='New York', substitute='New Toulouse', kind='place_name'),
            PIIMapping(original='Lyon', substitute='Toulouse', kind='place_name'),
        ]
        text = 'I left New Toulouse and arrived at Toulouse.'
        # Without length-ordering, the inner "Toulouse" would consume the
        # outer "New Toulouse" first and the answer would be wrong.
        assert literal_restore(text, mappings) == 'I left New York and arrived at Lyon.'

    def test_empty_mappings_passthrough(self):
        assert literal_restore('hello', []) == 'hello'


class TestOrphanDetection:
    def test_orphan_detected_when_substitute_remains(self):
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        # Substitute still in text post-literal-pass: literal pass would
        # have removed it, so this state implies the judge paraphrased
        # around it.
        assert _has_orphan_substitutes('Mme Camille said hi.', mappings) is True

    def test_no_orphan_when_literal_pass_covered_everything(self):
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        assert _has_orphan_substitutes('Mme Marie said hi.', mappings) is False


class TestRestoreText:
    async def test_literal_only_skips_llm_call(self, patch_provider, patch_settings):
        client = patch_provider('SHOULD NOT BE CALLED')
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        out, repair_used = await restore_text('Hi Camille!', mappings)
        assert out == 'Hi Marie!'
        assert repair_used is False
        assert client.calls == []  # the repair pass never ran

    async def test_repair_pass_fires_on_orphans(self, patch_provider, patch_settings):
        # The literal pass would replace ``Camille`` only. The judge
        # used ``Mme Camille``, so the substitute survives and triggers
        # the repair call.
        client = patch_provider('Mme Marie wrote back the next day.')
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        text = 'Mme Camille wrote back the next day.'
        out, repair_used = await restore_text(text, mappings)
        assert repair_used is True
        assert out == 'Mme Marie wrote back the next day.'
        assert len(client.calls) == 1
        # The repair prompt carries both the table and the literally-
        # restored text — verify the table arrived.
        repair_user_msg = client.calls[0]['messages'][1]['content']
        assert 'Camille → Marie' in repair_user_msg

    async def test_repair_pass_failure_falls_back_to_literal(
        self, patch_provider, patch_settings,
    ):
        client = patch_provider('')  # empty repair response
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        out, repair_used = await restore_text('Mme Camille said hi.', mappings)
        # Literal pass replaced ``Camille`` -> ``Marie``; ``Mme Camille``
        # → ``Mme Marie``. Repair returned empty, so we keep that.
        assert out == 'Mme Marie said hi.'
        assert repair_used is False
        assert len(client.calls) == 1  # we tried, then fell back

    async def test_allow_llm_repair_false_disables_second_pass(
        self, patch_provider, patch_settings,
    ):
        client = patch_provider('SHOULD NOT BE CALLED')
        mappings = [PIIMapping(original='Marie', substitute='Camille', kind='person_name')]
        out, repair_used = await restore_text(
            'Mme Camille said hi.', mappings, allow_llm_repair=False,
        )
        assert repair_used is False
        assert client.calls == []
        # Literal-pass output still correct on the embedded substitute,
        # even without repair.
        assert out == 'Mme Marie said hi.'


# ---------------------------------------------------------------------------
# Pipeline-integration smoke test (pipeline + storage layer, no Ollama)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """End-to-end pipeline turn with pseudonymization on, every model stubbed."""

    async def test_pipeline_uses_pseudonymized_prompt_and_restores(
        self, monkeypatch, tmp_path, patch_provider, patch_settings,
    ):
        from core import pipeline as pipeline_mod
        from core import storage as storage_mod
        from core.schemas import (
            ChatRequest,
            ModelResponse,
            RouterDecision,
            RouterPreferences,
            SynthesisResult,
        )

        # Point storage at a clean temp dir so the test doesn't touch
        # the user's real conversation log.
        monkeypatch.setattr(storage_mod, '_storage_mod', storage_mod)
        storage_mod.get_storage.cache_clear()
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))

        from core import config as config_mod
        config_mod.get_settings.cache_clear()
        # The settings stub overrides the in-process resolution, but the
        # pipeline calls into storage which respects ROITELET_DATA_DIR.

        # Stub the pseudonymizer's local model.
        rewritten = 'Email Camille about Toulouse.'
        forward_body = _forward_body(
            rewritten,
            [
                {'original': 'Marie', 'substitute': 'Camille', 'kind': 'person_name'},
                {'original': 'Lyon', 'substitute': 'Toulouse', 'kind': 'place_name'},
            ],
        )
        patch_provider(forward_body)

        seen_prompts: list[str] = []

        async def fake_query_one(model_id: str, messages: Any) -> ModelResponse:
            seen_prompts.append(messages[0].content)
            return ModelResponse(
                model_id=model_id,
                provider='ollama',
                content=f'Reply for {messages[0].content!r}',
                latency_s=0.01,
            )

        async def fake_judge(prompt: str, responses: list[ModelResponse]) -> SynthesisResult:
            # The judge sees the pseudonymized prompt and echoes a
            # paraphrase that uses the substitute.
            assert prompt == rewritten
            return SynthesisResult(
                model_id='ollama/qwen3:8b',
                provider='ollama',
                content='Camille will receive an email about Toulouse.',
                judge_summary='',
                winning_model_ids=[],
                latency_s=0.01,
            )

        monkeypatch.setattr(pipeline_mod, '_query_one', fake_query_one)
        monkeypatch.setattr(pipeline_mod, 'judge_and_synthesize', fake_judge)

        # Bypass the router heuristic: install a tiny router stub that
        # returns one selected model id.
        class _RouterStub:
            def route(self, prompt, preferences, top_k):
                seen_prompts.append(f'router:{prompt}')
                return RouterDecision(
                    prompt=prompt,
                    categories={'reasoning': 1.0},
                    candidates=[],
                    selected_model_ids=['ollama/qwen3:8b'],
                    reasoning=[],
                )

        monkeypatch.setattr(pipeline_mod, 'get_router', lambda: _RouterStub())

        response = await pipeline_mod.run_roitelet_chat(
            ChatRequest(
                prompt='Email Marie about Lyon.',
                preferences=RouterPreferences(pseudonymize=True),
            )
        )

        # The router and the candidate both saw the pseudonymized text.
        assert seen_prompts[0] == 'router:Email Camille about Toulouse.'
        assert any(p == 'Email Camille about Toulouse.' for p in seen_prompts)
        # The user sees the originals back.
        assert response.synthesis.content == 'Marie will receive an email about Lyon.'
        # The audit is attached to the response.
        assert response.pseudonymization is not None
        assert response.pseudonymization.pseudonymized_prompt == rewritten
        assert {m.original for m in response.pseudonymization.mappings} == {'Marie', 'Lyon'}

        # Persistence contract: the conversation log stores the ORIGINAL
        # prompt; the substitute lives only in metadata.
        storage = storage_mod.get_storage()
        conv = storage.get_conversation(response.conversation_id)
        assert conv is not None
        user_message = conv.messages[0]
        assert user_message.content == 'Email Marie about Lyon.'
        assert 'pseudonymization' in user_message.metadata
