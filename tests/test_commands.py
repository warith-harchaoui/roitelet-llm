"""Tests for the slash-command parser and chat-endpoint integration.

The parser is pure, so most of the surface is unit-tested directly.
The API-layer integration is verified through ``test_api.py`` patterns
— start a FastAPI test client, stub the pipeline, hit ``/api/chat``
with a slash-command prefix, assert the right behaviour.
"""

from __future__ import annotations

import pytest


class TestParseCommand:
    """Pure parser — every branch of the if-ladder has one test."""

    def test_no_command_is_chat(self):
        from core.commands import parse_command

        parsed = parse_command('Hello there.')
        assert parsed.route_to == 'chat'
        assert parsed.stripped_prompt == 'Hello there.'
        assert parsed.personal_override is False
        assert parsed.matched_commands == []

    def test_image_routes_to_image(self):
        from core.commands import parse_command

        parsed = parse_command('/image a wren in oil paint')
        assert parsed.route_to == 'image'
        assert parsed.stripped_prompt == 'a wren in oil paint'
        assert parsed.matched_commands == ['/image']

    def test_image_aliases(self):
        from core.commands import parse_command

        for alias in ('/image', '/image-gen', '/img', '/Image', '/IMG'):
            parsed = parse_command(f'{alias} sunset')
            assert parsed.route_to == 'image', f'{alias} failed'

    def test_speech_routes_to_speech(self):
        from core.commands import parse_command

        parsed = parse_command('/speech')
        assert parsed.route_to == 'speech'
        assert parsed.stripped_prompt == ''

    def test_help_routes_to_help(self):
        from core.commands import parse_command

        parsed = parse_command('/help')
        assert parsed.route_to == 'help'
        assert parsed.stripped_prompt == ''

    def test_personal_routes_to_chat_with_override(self):
        """``/personal Q`` is a per-turn route that injects the wiki context."""
        from core.commands import parse_command

        parsed = parse_command('/personal what did I write about Q3?')
        assert parsed.route_to == 'chat'
        assert parsed.personal_override is True
        assert parsed.stripped_prompt == 'what did I write about Q3?'
        assert parsed.matched_commands == ['/personal']

    def test_unknown_command_passes_through(self):
        """Typos must not silently change behaviour."""
        from core.commands import parse_command

        parsed = parse_command('/imagine a sunset')
        assert parsed.route_to == 'chat'
        assert parsed.stripped_prompt == '/imagine a sunset'
        assert parsed.matched_commands == []

    def test_removed_preference_slashes_pass_through_as_text(self):
        """``/local``, ``/cheap``, ``/k``, ``/pseudo``, ``/nopseudo`` were removed.

        After the 2026-05-30 UX simplification, per-turn preferences are
        controlled via visible affordances (composer sliders, CLI ``--``
        flags, API booleans). The old slash forms must NOT silently
        toggle behaviour — they must pass through as plain text so a
        muscle-memory typist sees their prompt unchanged.
        """
        from core.commands import parse_command

        for legacy in ('/local hello', '/cheap 0.005 hello', '/k 5 hello',
                       '/pseudo hello', '/nopseudo hello'):
            parsed = parse_command(legacy)
            assert parsed.route_to == 'chat'
            assert parsed.stripped_prompt == legacy
            assert parsed.matched_commands == []


class TestRenderHelp:
    def test_help_is_non_empty_markdown(self):
        from core.commands import render_help

        body = render_help()
        assert 'slash-command' in body.lower()
        assert '/image' in body
        assert '/speech' in body
        assert '/help' in body


class TestApiIntegration:
    """``/api/chat`` must honour the parser's verdicts."""

    @pytest.fixture
    def api_client(self, monkeypatch):
        from fastapi.testclient import TestClient

        # Stub the pipeline so the test doesn't need Ollama.
        from core import pipeline as pipeline_mod
        from core.schemas import ChatResponse, ModelResponse, RouterDecision, SynthesisResult

        async def stub_run(payload, router=None):
            return ChatResponse(
                conversation_id='conv',
                router=RouterDecision(
                    prompt=payload.prompt,
                    categories={'reasoning': 1.0},
                    candidates=[],
                    selected_model_ids=[],
                    reasoning=[f'preferences={payload.preferences.model_dump()}', f'top_k={payload.top_k}'],
                ),
                responses=[
                    ModelResponse(model_id='stub', provider='stub', content='ok',
                                  latency_s=0.1),
                ],
                synthesis=SynthesisResult(
                    model_id='stub-judge', provider='stub', content='stubbed',
                    judge_summary='', winning_model_ids=[],
                ),
                telemetry_id='tel',
            )

        monkeypatch.setattr(pipeline_mod, 'run_roitelet_chat', stub_run)
        # Make sure the api module sees the stubbed name even though
        # it imported the original symbol at module load.
        from api import main as api_main
        monkeypatch.setattr(api_main, 'run_roitelet_chat', stub_run)

        return TestClient(api_main.app)

    def test_help_short_circuits(self, api_client):
        response = api_client.post('/api/chat', json={'prompt': '/help'})
        assert response.status_code == 200
        data = response.json()
        assert 'slash-command' in data['synthesis']['content'].lower()
        # Help is static — no pipeline run, no telemetry id.
        assert data['telemetry_id'] == ''

    def test_image_rejected_with_pointer(self, api_client):
        response = api_client.post('/api/chat', json={'prompt': '/image a sunset'})
        assert response.status_code == 400
        detail = response.json()['detail']
        assert detail['route_to'] == 'image'
        assert '/api/images' in detail['message']

    def test_speech_rejected_with_pointer(self, api_client):
        response = api_client.post('/api/chat', json={'prompt': '/speech'})
        assert response.status_code == 400
        detail = response.json()['detail']
        assert detail['route_to'] == 'speech'
        assert 'multimodal' in detail['message']

    def test_preferences_flow_through_request_payload(self, api_client):
        """Per-turn preferences ride in the JSON ``preferences`` field — no slash needed."""
        response = api_client.post(
            '/api/chat',
            json={
                'prompt': 'refactor',
                'preferences': {'independence': True, 'pseudonymize': True},
                'top_k': 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        joined = ' '.join(data['router']['reasoning'])
        assert "'independence': True" in joined
        assert "'pseudonymize': True" in joined
        assert 'top_k=5' in joined
