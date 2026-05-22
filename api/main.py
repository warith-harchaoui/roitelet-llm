"""FastAPI application for Roitelet LLM.

The API exposes three access styles:
1. a native Roitelet JSON API,
2. an OpenAI-compatible `/v1/chat/completions` endpoint,
3. a compact MCP-compatible JSON-RPC endpoint.

Examples
--------
>>> from core.main import app
>>> app.title
'Roitelet LLM API'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.config import get_settings
from core.mcp import handle_mcp_request
from core.pipeline import run_roitelet_chat
from core.registry import warm_ollama_cache
from core.schemas import (
    AppSettingsPayload,
    ChatRequest,
    MCPRequest,
    OpenAIChatCompletionRequest,
    RouterPreferences,
)
from core.storage import StorageManager, get_storage

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan hook: warm Ollama cache at startup."""
    runtime = get_storage().load_app_settings()
    ollama_url = runtime.ollama_base_url or settings.local_llm_base_url
    warm_ollama_cache(ollama_url, force=True)
    logger.info('Roitelet LLM API ready — Ollama at %s', ollama_url)
    yield
    logger.info('Roitelet LLM API shutting down.')


async def require_api_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token gate for sensitive endpoints.

    Defaults to a no-op (local-first single-user UX is preserved). When
    ``ROITELET_API_TOKEN`` is set the request must carry a matching
    ``Authorization: Bearer <token>`` header. Wired as a FastAPI dependency
    so tests can override it via ``app.dependency_overrides`` when needed.
    """
    expected = settings.api_token
    if not expected:
        return  # Auth disabled — preserve unauthenticated localhost UX.
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing bearer token')
    if authorization[len('Bearer '):].strip() != expected:
        raise HTTPException(status_code=401, detail='Invalid bearer token')


app = FastAPI(title='Roitelet LLM API', version='0.1.0', lifespan=lifespan)

_cors_raw = settings.cors_allowed_origins.strip()
_cors_origins = ['*'] if _cors_raw == '*' else [
    origin.strip() for origin in _cors_raw.split(',') if origin.strip()
]
# `allow_credentials=True` is incompatible with wildcard origins per CORS spec;
# downgrade automatically when the operator opts into the wildcard.
_allow_credentials = _cors_origins != ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/healthz')
async def health() -> Dict[str, Any]:
    """Return a lightweight health payload to indicate the server is alive.

    The web client is mounted at ``/`` as static files, so the health probe
    moved to ``/healthz``.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the status, service name, and public base URL.
    """
    return {'status': 'ok', 'service': 'roitelet-llm', 'base_url': settings.public_base_url}


@app.get('/api/settings', dependencies=[Depends(require_api_token)])
async def get_app_settings(storage: StorageManager = Depends(get_storage)) -> Dict[str, Any]:
    """Return persisted control-room settings with API keys masked.

    Stored credentials never leave the server — any client (even on the same
    machine) sees a fixed mask in place of real keys. The web UI round-trips
    that mask back through POST, and the server reuses the stored value.
    """
    return storage.load_app_settings().masked().model_dump()


@app.post('/api/settings', dependencies=[Depends(require_api_token)])
async def save_app_settings(
    payload: AppSettingsPayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """Persist control-room settings edited from the web UI.

    Fields whose value still equals the secret mask sentinel are preserved
    from the previously stored payload — the UI may submit masked values
    untouched without wiping credentials.
    """
    stored = storage.load_app_settings()
    merged = stored.merge_unmasked(payload)
    storage.save_app_settings(merged)
    return {'status': 'saved'}


@app.get('/api/conversations')
async def list_conversations(
    storage: StorageManager = Depends(get_storage),
) -> List[Dict[str, Any]]:
    """List all stored conversations via the local storage manager."""
    return [conversation.model_dump() for conversation in storage.list_conversations()]


@app.get('/api/conversations/{conversation_id}')
async def get_conversation(
    conversation_id: str,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """Fetch one conversation by its unique identifier."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail='Conversation not found')
    return conversation.model_dump()


@app.get('/api/telemetry')
async def list_telemetry(
    storage: StorageManager = Depends(get_storage),
) -> List[Dict[str, Any]]:
    """Return all telemetry records containing performance data."""
    return [record.model_dump() for record in storage.list_telemetry()]


def _sse(event: Dict[str, Any]) -> str:
    """Format one Server-Sent Events frame from a JSON-serialisable payload."""
    return f'data: {json.dumps(event)}\n\n'


async def _stream_synthesis_content(content: str, chunk_size: int = 8) -> AsyncGenerator[str, None]:
    """Yield the synthesis content as small character-aligned chunks.

    Character (not word) splitting preserves markdown structure: code
    fences, newlines, and inline punctuation survive intact. The chunk
    size is a perceptual setting — bigger = fewer events but choppier UX.
    """
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]


@app.post('/api/chat')
async def roitelet_chat(payload: ChatRequest):
    """Run one native Roitelet chat turn.

    When ``payload.stream`` is true the response is a Server-Sent Events
    stream: ``{type: "delta", content: ...}`` frames followed by a final
    ``{type: "done", conversation_id, telemetry_id, router, responses, synthesis}``
    summary so the client gets every piece the non-streaming JSON would
    have returned, just split in time.

    Parameters
    ----------
    payload : ChatRequest
        The prompt, router preferences, and optional ``stream`` flag.

    Returns
    -------
    Dict[str, Any] | StreamingResponse
        The resulting synthesis, full conversation body, and model decisions.
    """
    response = await run_roitelet_chat(payload)
    if not payload.stream:
        return response.model_dump()

    async def event_stream() -> AsyncGenerator[str, None]:
        async for token in _stream_synthesis_content(response.synthesis.content):
            yield _sse({'type': 'delta', 'content': token})
        yield _sse({
            'type': 'done',
            'conversation_id': response.conversation_id,
            'telemetry_id': response.telemetry_id,
            'router': response.router.model_dump(),
            'responses': [r.model_dump() for r in response.responses],
            'synthesis': response.synthesis.model_dump(),
        })

    return StreamingResponse(event_stream(), media_type='text/event-stream')


@app.get('/v1/models')
async def list_models() -> Dict[str, Any]:
    """Expose the local OpenAI-compatible model inventory."""
    return {
        'object': 'list',
        'data': [
            {
                'id': 'roitelet-llm',
                'object': 'model',
                'owned_by': 'deraison.ai',
            }
        ],
    }


@app.post('/v1/chat/completions', dependencies=[Depends(require_api_token)])
async def openai_chat_completions(payload: OpenAIChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    payload : OpenAIChatCompletionRequest
        Data strictly adhering to the standard OpenAI `/v1/chat/completions` spec.

    Returns
    -------
    Union[Dict[str, Any], StreamingResponse]
        Standard static completions response dict, or a Server-Sent Events stream.
    """
    prompt = '\n'.join(message.content for message in payload.messages if message.role == 'user')
    response = await run_roitelet_chat(
        ChatRequest(
            prompt=prompt,
            preferences=RouterPreferences(),
        )
    )

    if payload.stream:
        completion_id = f'chatcmpl-{uuid.uuid4().hex}'

        async def event_stream() -> AsyncGenerator[str, None]:
            async for token in _stream_synthesis_content(response.synthesis.content):
                chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'roitelet-llm',
                    'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}],
                }
                yield _sse(chunk)
            done = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'roitelet-llm',
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
            }
            yield _sse(done)
            yield 'data: [DONE]\n\n'

        return StreamingResponse(event_stream(), media_type='text/event-stream')

    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': 'roitelet-llm',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': response.synthesis.content},
                'finish_reason': 'stop',
            }
        ],
        'usage': {
            'prompt_tokens': sum(
                int(r.usage.get('prompt_tokens', r.usage.get('prompt_eval_count', 0)))
                for r in response.responses
            ),
            'completion_tokens': sum(
                int(r.usage.get('completion_tokens', r.usage.get('eval_count', 0)))
                for r in response.responses
            ),
            'total_tokens': sum(
                int(r.usage.get('total_tokens', 0))
                + int(r.usage.get('prompt_eval_count', 0))
                + int(r.usage.get('eval_count', 0))
                for r in response.responses
            ),
        },
        'roitelet_metadata': response.model_dump(),
    }


@app.post('/mcp')
async def mcp_endpoint(payload: MCPRequest):
    """Handle MCP-like JSON-RPC requests over plain HTTP.

    Parameters
    ----------
    payload : MCPRequest
        A JSON-RPC 2.0 conformant request carrying an arbitrary method and params.

    Returns
    -------
    Union[Dict[str, Any], JSONResponse]
        The computed JSON-RPC protocol response.
    """
    try:
        return await handle_mcp_request(payload)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={'jsonrpc': '2.0', 'id': payload.id, 'error': {'code': -32000, 'message': str(exc)}},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Static SPA. Mounted at '/' so the vanilla JS client lives at the root URL.
# Must be declared after all API routes to avoid shadowing them.
# ──────────────────────────────────────────────────────────────────────────────

_WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
if _WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=_WEB_DIR, html=True), name='web')
