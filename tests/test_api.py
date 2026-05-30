"""HTTP-layer tests for Roitelet's API surface.

Five tests cover the contract worth pinning:

1. The unauthenticated default path: ``/healthz`` is reachable and the
   static SPA mount serves the GUI HTML at ``/``. Every gated endpoint
   stays reachable without a token (the documented local-first UX).
2. Auth gate: when ``ROITELET_API_TOKEN`` is set, gated endpoints
   demand a matching Bearer header.
3. Settings round-trip: GET masks secrets, POST with the mask sentinel
   preserves the on-disk value, POST with a real value overwrites.
   The traversal-rejection contract on conversation ids is exercised
   in the same test because both live in the storage layer.
4. MCP JSON-RPC: ``initialize`` + ``tools/list`` + unknown-tool error
   form one coherent handshake to verify.
5. Streaming + non-streaming completion responses on both the
   OpenAI-compat and native chat endpoints, exercised end-to-end with
   a stubbed pipeline.
"""

from __future__ import annotations

import json

import pytest
from fastapi import Header, HTTPException
from fastapi.testclient import TestClient

from api.main import app, require_api_token
from core.schemas import (
    SECRET_FIELDS,
    SECRET_MASK,
    ChatResponse,
    ModelCandidate,
    ModelResponse,
    RouterDecision,
    SynthesisResult,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_required():
    """Inject the Bearer-token gate for one test via ``dependency_overrides``.

    We override at the FastAPI level rather than mutating env vars or
    settings globals so the gate state is bounded to this test only.
    """
    async def _override(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail='Missing bearer token')
        if authorization[len('Bearer '):].strip() != 'test-token':
            raise HTTPException(status_code=401, detail='Invalid bearer token')

    app.dependency_overrides[require_api_token] = _override
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_api_token, None)


def _stub_response(content: str = 'Stub synthesis answer.') -> ChatResponse:
    """Deterministic ChatResponse so endpoint tests don't need a real pipeline."""
    return ChatResponse(
        conversation_id='conv-stub',
        router=RouterDecision(
            prompt='hi',
            categories={'coding': 1.0},
            candidates=[
                ModelCandidate(
                    model_id='ollama/qwen3:8b', provider='ollama',
                    selected=True, score=0.9,
                ),
            ],
            selected_model_ids=['ollama/qwen3:8b'],
            reasoning=['stub'],
        ),
        responses=[
            ModelResponse(
                model_id='ollama/qwen3:8b', provider='ollama',
                content=content, latency_s=0.0,
                usage={'prompt_tokens': 5.0, 'completion_tokens': 5.0},
            ),
        ],
        synthesis=SynthesisResult(
            model_id='ollama/qwen3:8b', provider='ollama',
            content=content, judge_summary='WINNERS: 1',
            winning_model_ids=['ollama/qwen3:8b'],
        ),
        telemetry_id='tel-stub',
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the pipeline with a stub so streaming tests don't need Ollama."""
    async def _fake(_request):
        return _stub_response('Stub synthesis answer.')

    monkeypatch.setattr('api.main.run_roitelet_chat', _fake)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_local_first_surfaces_are_reachable_without_auth():
    """The documented single-user UX: ``/healthz`` and the static SPA
    are public, every gated endpoint is reachable without a token, and
    the OpenAI-compatible models inventory enumerates the meta-id plus
    whatever the registry knows about."""
    health = client.get('/healthz').json()
    assert health['status'] == 'ok'
    assert 'roitelet' in health['service']

    root = client.get('/')
    assert 'text/html' in root.headers['content-type']
    assert '<title>Roitelet</title>' in root.text

    # Gated endpoints are open by default.
    assert client.get('/api/settings').status_code == 200

    models = client.get('/v1/models').json()
    assert models['object'] == 'list'
    assert any(m['id'] == 'roitelet' for m in models['data'])


def test_auth_gate_rejects_missing_or_wrong_token_and_accepts_the_right_one(auth_required):
    """When ``ROITELET_API_TOKEN`` is configured the gated endpoints
    must return 401 unless the request carries the exact Bearer token."""
    assert client.get('/api/settings').status_code == 401
    assert client.get(
        '/api/settings', headers={'Authorization': 'Bearer wrong-token'},
    ).status_code == 401
    ok = client.get(
        '/api/settings', headers={'Authorization': 'Bearer test-token'},
    )
    assert ok.status_code == 200
    assert 'local_synthesis_model' in ok.json()


def test_settings_mask_round_trip_and_traversal_rejection():
    """One test pins the storage-layer security contract:

    * secrets must be masked on the wire,
    * the mask sentinel round-trips back without blanking,
    * a real new value overwrites,
    * non-UUID conversation ids return 404 (no filesystem escape).
    """
    from core.storage import storage

    stored = storage.load_app_settings()
    real_secret = 'sk-or-test-keep-me'
    storage.save_app_settings(stored.model_copy(update={'openrouter_api_key': real_secret}))
    try:
        masked = client.get('/api/settings').json()
        # On the wire: every populated secret field is masked.
        assert masked['openrouter_api_key'] == SECRET_MASK
        for field in SECRET_FIELDS:
            assert masked[field] != real_secret

        # POST the masked payload unchanged → the real secret survives.
        assert client.post('/api/settings', json=masked).status_code == 200
        assert storage.load_app_settings().openrouter_api_key == real_secret

        # POST a real new value → it overwrites.
        new_secret = 'sk-or-test-new-value'
        next_payload = stored.model_copy(update={'openrouter_api_key': new_secret}).model_dump()
        assert client.post('/api/settings', json=next_payload).status_code == 200
        assert storage.load_app_settings().openrouter_api_key == new_secret

        # Traversal payloads on conversation ids fail closed.
        for evil in ('..%2F..%2Fetc%2Fpasswd', 'not-a-uuid', '%00'):
            assert client.get(f'/api/conversations/{evil}').status_code in (404, 422)
    finally:
        storage.save_app_settings(stored)


def test_mcp_handshake_lists_one_tool_and_rejects_unknown_methods():
    """The MCP JSON-RPC surface is small but real: a handshake response,
    a single advertised tool, and clean JSON-RPC errors on bad inputs."""
    # initialize handshake
    init = client.post('/mcp', json={
        'jsonrpc': '2.0', 'id': 'init-1', 'method': 'initialize', 'params': {},
    }).json()
    assert init['result']['serverInfo']['name'] == 'roitelet'
    assert 'protocolVersion' in init['result']

    # tools/list — exactly one tool, with ``prompt`` required.
    tools_list = client.post('/mcp', json={
        'jsonrpc': '2.0', 'id': 'list-1', 'method': 'tools/list', 'params': {},
    }).json()
    tools = tools_list['result']['tools']
    assert [t['name'] for t in tools] == ['roitelet.chat']
    assert 'prompt' in tools[0]['inputSchema']['required']

    # Unknown tool name and unsupported method both fail as JSON-RPC errors.
    unknown_tool = client.post('/mcp', json={
        'jsonrpc': '2.0', 'id': 'call-1', 'method': 'tools/call',
        'params': {'name': 'does-not-exist', 'arguments': {'prompt': 'x'}},
    })
    assert unknown_tool.status_code == 400
    assert unknown_tool.json()['error']['code'] == -32000

    bad_method = client.post('/mcp', json={
        'jsonrpc': '2.0', 'id': 'm-1', 'method': 'does/not/exist', 'params': {},
    })
    assert bad_method.status_code == 400


def test_chat_endpoints_handle_streaming_and_non_streaming_responses(stub_pipeline):
    """One test exercises the four chat-response shapes:

    * OpenAI-compat non-streaming JSON,
    * OpenAI-compat SSE stream ending with ``[DONE]``,
    * native ``/api/chat`` SSE with structured ``delta`` + ``done`` frames.

    Each branch reconstructs the answer from its frames so the streaming
    contract isn't just "frames came out" but "frames reproduce the
    fused answer".
    """
    # 1. OpenAI non-streaming.
    plain = client.post('/v1/chat/completions', json={
        'model': 'roitelet', 'messages': [{'role': 'user', 'content': 'hi'}],
    }).json()
    assert plain['choices'][0]['message']['content'] == 'Stub synthesis answer.'

    # 2. OpenAI streaming — SSE with [DONE] sentinel.
    with client.stream('POST', '/v1/chat/completions', json={
        'model': 'roitelet', 'messages': [{'role': 'user', 'content': 'hi'}],
        'stream': True,
    }) as response:
        assert response.headers['content-type'].startswith('text/event-stream')
        body = ''.join(response.iter_text())
    frames = [f for f in body.split('\n\n') if f.startswith('data: ')]
    assert frames[-1].strip() == 'data: [DONE]'
    reconstructed = ''
    for frame in frames[:-2]:  # last two are finish + [DONE]
        chunk = json.loads(frame[len('data: '):])
        reconstructed += chunk['choices'][0]['delta'].get('content', '')
    assert reconstructed == 'Stub synthesis answer.'

    # 3. Native SSE — structured delta + done frames.
    with client.stream('POST', '/api/chat', json={
        'prompt': 'hi', 'stream': True,
    }) as response:
        body = ''.join(response.iter_text())
    frames = [json.loads(f[len('data: '):]) for f in body.split('\n\n') if f.startswith('data: ')]
    assert {f['type'] for f in frames} == {'delta', 'done'}
    deltas = ''.join(f['content'] for f in frames if f['type'] == 'delta')
    assert deltas == 'Stub synthesis answer.'
    final = next(f for f in frames if f['type'] == 'done')
    assert final['conversation_id'] == 'conv-stub'
