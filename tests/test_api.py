import pytest
from fastapi.testclient import TestClient

from api.main import app, require_api_token

client = TestClient(app)


@pytest.fixture
def auth_required():
    """Force the Bearer-token gate to require ``Authorization: Bearer test-token``
    for the lifetime of one test, without touching environment variables.

    ``require_api_token`` is a no-op by default (local-first single-user
    UX). Tests that exercise the gated path override the dependency via
    FastAPI's ``dependency_overrides`` so we don't mutate settings globals
    or leak state between tests.
    """
    async def _require(authorization: str | None = None):
        from fastapi import HTTPException
        # Replicate the production check with a fixed expected token.
        if authorization is None:
            raise HTTPException(status_code=401, detail='Missing bearer token')
        if not authorization.startswith('Bearer '):
            raise HTTPException(status_code=401, detail='Missing bearer token')
        if authorization[len('Bearer '):].strip() != 'test-token':
            raise HTTPException(status_code=401, detail='Invalid bearer token')

    # The override needs the *exact* signature for FastAPI to inject Header.
    from fastapi import Header
    async def _override(authorization: str | None = Header(default=None)):
        await _require(authorization)

    app.dependency_overrides[require_api_token] = _override
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_api_token, None)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "roitelet" in response.json().get("service", "")


def test_root_serves_spa():
    """The vanilla JS client is mounted at '/' and must be served as HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Roitelet</title>" in response.text

def test_v1_models():
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert len(response.json()["data"]) > 0

def test_api_settings():
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert "local_synthesis_model" in response.json()


def test_conversation_path_traversal_rejected():
    """Malicious conversation ids must return 404, not leak files outside data/."""
    # Both raw and URL-encoded traversal payloads must fail closed.
    for evil in ("..%2F..%2Fetc%2Fpasswd", "not-a-uuid", "%00"):
        response = client.get(f"/api/conversations/{evil}")
        assert response.status_code in (404, 422), f"Unexpected status for {evil}: {response.status_code}"


def test_api_settings_masks_api_keys():
    """GET /api/settings must never echo real API keys to the client."""
    from core.schemas import SECRET_FIELDS, SECRET_MASK
    from core.storage import storage

    stored = storage.load_app_settings()
    real_secret = 'sk-or-test-real-secret-value'
    updated = stored.model_copy(update={'openrouter_api_key': real_secret})
    storage.save_app_settings(updated)
    try:
        body = client.get('/api/settings').json()
        # Any non-empty secret must be masked.
        assert body['openrouter_api_key'] == SECRET_MASK
        for field in SECRET_FIELDS:
            assert body[field] != real_secret
    finally:
        storage.save_app_settings(stored)


def test_api_settings_post_preserves_masked_secrets():
    """POSTing the mask sentinel must keep the stored key, not overwrite it."""
    from core.schemas import SECRET_MASK
    from core.storage import storage

    stored = storage.load_app_settings()
    real_secret = 'sk-or-test-keep-me'
    storage.save_app_settings(stored.model_copy(update={'openrouter_api_key': real_secret}))
    try:
        masked = client.get('/api/settings').json()
        assert masked['openrouter_api_key'] == SECRET_MASK
        # Round-trip the masked payload unchanged.
        response = client.post('/api/settings', json=masked)
        assert response.status_code == 200
        # The on-disk value must still be the real secret.
        assert storage.load_app_settings().openrouter_api_key == real_secret
    finally:
        storage.save_app_settings(stored)


def test_settings_unauthorized_when_token_required(auth_required):
    """With a token configured, GET /api/settings must reject missing creds."""
    assert client.get('/api/settings').status_code == 401
    assert client.get(
        '/api/settings', headers={'Authorization': 'Bearer wrong-token'}
    ).status_code == 401


def test_settings_accepts_correct_bearer(auth_required):
    """The configured token must unlock the gated endpoints."""
    response = client.get(
        '/api/settings', headers={'Authorization': 'Bearer test-token'}
    )
    assert response.status_code == 200
    assert 'local_synthesis_model' in response.json()


def test_settings_unauthenticated_by_default():
    """Without ROITELET_API_TOKEN set, the endpoint is reachable without auth.

    This is the documented local-first default — single-user, single
    machine. Tests must continue to pass without any token configured.
    """
    response = client.get('/api/settings')
    assert response.status_code == 200


def test_api_settings_post_accepts_new_secret():
    """A POST with a real (non-mask) secret value must actually overwrite."""
    from core.storage import storage

    stored = storage.load_app_settings()
    try:
        new_secret = 'sk-or-test-new-value'
        next_payload = stored.model_copy(update={'openrouter_api_key': new_secret}).model_dump()
        response = client.post('/api/settings', json=next_payload)
        assert response.status_code == 200
        assert storage.load_app_settings().openrouter_api_key == new_secret
    finally:
        storage.save_app_settings(stored)


# ---------------------------------------------------------------------------
# /mcp JSON-RPC endpoint
# ---------------------------------------------------------------------------


def test_mcp_initialize():
    """The MCP handshake must advertise the protocol version and server info."""
    body = {'jsonrpc': '2.0', 'id': 'init-1', 'method': 'initialize', 'params': {}}
    payload = client.post('/mcp', json=body).json()
    assert payload['id'] == 'init-1'
    assert payload['result']['serverInfo']['name'] == 'roitelet-llm'
    assert 'protocolVersion' in payload['result']


def test_mcp_tools_list_contains_roitelet_chat():
    """tools/list must expose the single roitelet.chat tool."""
    body = {'jsonrpc': '2.0', 'id': 'list-1', 'method': 'tools/list', 'params': {}}
    payload = client.post('/mcp', json=body).json()
    tools = payload['result']['tools']
    assert [t['name'] for t in tools] == ['roitelet.chat']
    assert 'prompt' in tools[0]['inputSchema']['required']


def test_mcp_tools_call_unknown_tool_errors():
    """An unknown tool name must surface as a JSON-RPC error, not a crash."""
    body = {
        'jsonrpc': '2.0',
        'id': 'call-1',
        'method': 'tools/call',
        'params': {'name': 'does-not-exist', 'arguments': {'prompt': 'x'}},
    }
    response = client.post('/mcp', json=body)
    assert response.status_code == 400
    payload = response.json()
    assert 'error' in payload
    assert payload['error']['code'] == -32000


def test_mcp_unsupported_method_errors():
    body = {'jsonrpc': '2.0', 'id': 'm-1', 'method': 'does/not/exist', 'params': {}}
    response = client.post('/mcp', json=body)
    assert response.status_code == 400
    assert 'error' in response.json()


# ---------------------------------------------------------------------------
# /v1/chat/completions streaming branch
# ---------------------------------------------------------------------------


def _stub_pipeline_response(content: str = 'Stub synthesis answer.'):
    """Build a ChatResponse the streaming/non-streaming branches can render."""
    from core.schemas import (
        ChatResponse,
        ModelCandidate,
        ModelResponse,
        RouterDecision,
        SynthesisResult,
    )
    return ChatResponse(
        conversation_id='conv-stub',
        router=RouterDecision(
            prompt='hi',
            categories={'coding': 1.0},
            candidates=[
                ModelCandidate(
                    model_id='ollama/qwen3:8b',
                    provider='ollama',
                    selected=True,
                    score=0.9,
                    capability_scores=[],
                ),
            ],
            selected_model_ids=['ollama/qwen3:8b'],
            reasoning=['stub'],
        ),
        responses=[
            ModelResponse(
                model_id='ollama/qwen3:8b',
                provider='ollama',
                content=content,
                latency_s=0.0,
                usage={'prompt_tokens': 5.0, 'completion_tokens': 5.0},
            ),
        ],
        synthesis=SynthesisResult(
            model_id='ollama/qwen3:8b',
            provider='ollama',
            content=content,
            judge_summary='WINNERS: 1',
            winning_model_ids=['ollama/qwen3:8b'],
        ),
        telemetry_id='tel-stub',
    )


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace the pipeline with a deterministic stub so endpoint tests
    don't touch real providers, registries, or storage."""
    async def _fake(_request):
        return _stub_pipeline_response('Stub synthesis answer.')

    monkeypatch.setattr('api.main.run_roitelet_chat', _fake)


def test_openai_streaming_returns_sse(stub_pipeline):
    """stream=True must emit an SSE stream with delta chunks and a [DONE] sentinel."""
    body = {
        'model': 'roitelet-llm',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'stream': True,
    }
    with client.stream('POST', '/v1/chat/completions', json=body) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        body_text = ''.join(response.iter_text())

    # Frames must each start with "data: " per the SSE spec.
    frames = [f for f in body_text.split('\n\n') if f.startswith('data: ')]
    assert len(frames) >= 2, f'expected several delta frames + final [DONE], got {frames!r}'
    assert frames[-1].strip() == 'data: [DONE]'

    # Recombining the delta payloads must reproduce the stubbed content.
    import json as _json
    reconstructed = ''
    for frame in frames[:-2]:  # last two are the finish chunk + [DONE]
        chunk = _json.loads(frame[len('data: '):])
        reconstructed += chunk['choices'][0]['delta'].get('content', '')
    assert reconstructed == 'Stub synthesis answer.'


def test_openai_non_streaming_returns_json(stub_pipeline):
    """stream=False (default) must keep returning a plain JSON completion."""
    body = {'model': 'roitelet-llm', 'messages': [{'role': 'user', 'content': 'hi'}]}
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload['choices'][0]['message']['content'] == 'Stub synthesis answer.'


def test_native_chat_streaming_returns_sse(stub_pipeline):
    """/api/chat with stream=True must emit delta frames + a structured done frame."""
    body = {'prompt': 'hi', 'stream': True}
    with client.stream('POST', '/api/chat', json=body) as response:
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        body_text = ''.join(response.iter_text())

    frames = [f for f in body_text.split('\n\n') if f.startswith('data: ')]
    assert frames, 'no frames emitted'

    import json as _json
    parsed = [_json.loads(f[len('data: '):]) for f in frames]
    # Every frame must carry a recognised type.
    types = {f['type'] for f in parsed}
    assert types == {'delta', 'done'}, f'unexpected frame types: {types}'

    deltas = [f['content'] for f in parsed if f['type'] == 'delta']
    assert ''.join(deltas) == 'Stub synthesis answer.'

    final = [f for f in parsed if f['type'] == 'done'][0]
    assert final['conversation_id'] == 'conv-stub'
    assert final['telemetry_id'] == 'tel-stub'
