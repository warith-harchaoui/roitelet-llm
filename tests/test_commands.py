"""Slash-command parser + API integration.

Two tests. The parser is pure so we exercise every recognised route
plus the soft-fail-on-unknown contract in one comprehensive test. The
API integration test confirms /help short-circuits without a pipeline
call, /image and /speech redirect to the right endpoint, and per-turn
preferences flow through the JSON body (now that the per-turn slashes
``/local`` / ``/cheap`` / ``/k`` / ``/pseudo`` were retired).
"""

from __future__ import annotations

import pytest


def test_parser_recognises_every_route_and_fails_soft_on_unknown():
    """The parser is leading-only, route-only, and fail-soft. Every
    recognised route shows up here:

    * ``/image`` and its aliases route to image-gen,
    * ``/speech`` to STT,
    * ``/help`` to the static catalogue,
    * ``/personal`` is a per-turn override that prepends the wiki.

    Unknown commands (typos) pass through as plain text — this is the
    contract that lets a user *talk about* slash commands in a prompt.
    The retired per-turn slashes (``/local``, ``/cheap``, ``/k``,
    ``/pseudo``, ``/nopseudo``) also pass through verbatim so muscle
    memory doesn't trigger silent behaviour changes.
    """
    from core.commands import parse_command, render_help

    # No command → chat with empty overrides.
    plain = parse_command('Hello there.')
    assert plain.route_to == 'chat' and plain.matched_commands == []

    # /image and its aliases.
    for alias in ('/image', '/image-gen', '/img', '/IMG'):
        assert parse_command(f'{alias} sunset').route_to == 'image'

    # /speech, /help.
    assert parse_command('/speech').route_to == 'speech'
    assert parse_command('/help').route_to == 'help'

    # /personal sets the per-turn override and strips the prefix.
    personal = parse_command('/personal what did I write about Q3?')
    assert personal.route_to == 'chat'
    assert personal.personal_override is True
    assert personal.stripped_prompt == 'what did I write about Q3?'

    # Unknown command and every retired preference slash → pass through.
    for raw in ('/imagine a sunset', '/local hello', '/cheap 0.005 hello',
                '/k 5 hello', '/pseudo hello', '/nopseudo hello'):
        parsed = parse_command(raw)
        assert parsed.route_to == 'chat'
        assert parsed.stripped_prompt == raw
        assert parsed.matched_commands == []

    # render_help is the user-facing static catalogue.
    body = render_help()
    assert '/image' in body and '/speech' in body and '/help' in body


@pytest.fixture
def api_client(monkeypatch):
    """A TestClient with the pipeline stubbed."""
    from fastapi.testclient import TestClient

    from core.schemas import (
        ChatResponse,
        ModelResponse,
        RouterDecision,
        SynthesisResult,
    )

    async def stub(payload, router=None):
        return ChatResponse(
            conversation_id='conv',
            router=RouterDecision(
                prompt=payload.prompt,
                categories={'reasoning': 1.0},
                candidates=[],
                selected_model_ids=[],
                reasoning=[
                    f'preferences={payload.preferences.model_dump()}',
                    f'top_k={payload.top_k}',
                ],
            ),
            responses=[ModelResponse(model_id='stub', provider='stub',
                                     content='ok', latency_s=0.1)],
            synthesis=SynthesisResult(model_id='stub-judge', provider='stub',
                                      content='stubbed', judge_summary='',
                                      winning_model_ids=[]),
            telemetry_id='tel',
        )

    from api import main as api_main
    monkeypatch.setattr(api_main, 'run_roitelet_chat', stub)
    return TestClient(api_main.app)


def test_api_routes_help_redirects_image_and_speech_and_passes_preferences(api_client):
    """One test covers the four meaningful API behaviours:

    * ``/help`` short-circuits — the static catalogue comes back without
      a pipeline call (no telemetry id);
    * ``/image`` returns 400 with a pointer to ``/api/images``;
    * ``/speech`` returns 400 with a pointer to multimodal;
    * non-slash chat requests pass ``preferences`` + ``top_k`` straight
      through to the pipeline.
    """
    # /help — no pipeline call, no telemetry id.
    help_body = api_client.post('/api/chat', json={'prompt': '/help'}).json()
    assert 'slash-command' in help_body['synthesis']['content'].lower()
    assert help_body['telemetry_id'] == ''

    # /image redirect.
    img = api_client.post('/api/chat', json={'prompt': '/image a sunset'})
    assert img.status_code == 400
    assert img.json()['detail']['route_to'] == 'image'
    assert '/api/images' in img.json()['detail']['message']

    # /speech redirect.
    speech = api_client.post('/api/chat', json={'prompt': '/speech'})
    assert speech.status_code == 400
    assert speech.json()['detail']['route_to'] == 'speech'
    assert 'multimodal' in speech.json()['detail']['message']

    # Per-turn preferences ride in the JSON body.
    body = api_client.post(
        '/api/chat',
        json={
            'prompt': 'refactor',
            'preferences': {'independence': True, 'pseudonymize': True},
            'top_k': 5,
        },
    ).json()
    joined = ' '.join(body['router']['reasoning'])
    assert "'independence': True" in joined
    assert "'pseudonymize': True" in joined
    assert 'top_k=5' in joined
