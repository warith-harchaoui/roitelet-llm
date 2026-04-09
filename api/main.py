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
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import get_settings
from core.core.mcp import handle_mcp_request
from core.core.pipeline import run_roitelet_chat
from core.core.registry import warm_ollama_cache
from core.schemas import (
    AppSettingsPayload,
    ChatRequest,
    MCPRequest,
    OpenAIChatCompletionRequest,
    RouterPreferences,
)
from core.storage import storage

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan hook: warm Ollama cache at startup."""
    runtime = storage.load_app_settings()
    ollama_url = runtime.ollama_base_url or settings.local_llm_base_url
    warm_ollama_cache(ollama_url, force=True)
    logger.info('Roitelet LLM API ready — Ollama at %s', ollama_url)
    yield
    logger.info('Roitelet LLM API shutting down.')


app = FastAPI(title='Roitelet LLM API', version='0.1.0', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
async def root() -> Dict[str, Any]:
    """Return a lightweight health payload to indicate the server is alive.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the status, service name, and public base URL.
    """
    return {'status': 'ok', 'service': 'roitelet-llm', 'base_url': settings.public_base_url}


@app.get('/api/settings')
async def get_app_settings() -> Dict[str, Any]:
    """Return persisted control-room settings.

    Returns
    -------
    Dict[str, Any]
        The settings serialized as a dictionary.
    """
    return storage.load_app_settings().model_dump()


@app.post('/api/settings')
async def save_app_settings(payload: AppSettingsPayload) -> Dict[str, Any]:
    """Persist control-room settings edited from Streamlit.

    Parameters
    ----------
    payload : AppSettingsPayload
        The updated settings derived from the frontend.

    Returns
    -------
    Dict[str, Any]
        A success status message.
    """
    storage.save_app_settings(payload)
    return {'status': 'saved'}


@app.get('/api/conversations')
async def list_conversations() -> List[Dict[str, Any]]:
    """List all stored conversations via the local storage manager.

    Returns
    -------
    List[Dict[str, Any]]
        A list of serialized conversation payloads.
    """
    return [conversation.model_dump() for conversation in storage.list_conversations()]


@app.get('/api/conversations/{conversation_id}')
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """Fetch one conversation by its unique identifier.

    Parameters
    ----------
    conversation_id : str
        The UUID of the conversation to load.

    Returns
    -------
    Dict[str, Any]
        The requested conversation payload.

    Raises
    ------
    HTTPException
        If the conversation could not be located on disk.
    """
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail='Conversation not found')
    return conversation.model_dump()


@app.get('/api/telemetry')
async def list_telemetry() -> List[Dict[str, Any]]:
    """Return all telemetry records containing performance data.

    Returns
    -------
    List[Dict[str, Any]]
        A chronological list of metric snapshots.
    """
    return [record.model_dump() for record in storage.list_telemetry()]


@app.post('/api/chat')
async def roitelet_chat(payload: ChatRequest) -> Dict[str, Any]:
    """Run one native Roitelet chat turn using the advanced routing logic.

    Parameters
    ----------
    payload : ChatRequest
        The prompt and router preferences.

    Returns
    -------
    Dict[str, Any]
        The resulting synthesis, full conversation body, and model decisions.
    """
    response = await run_roitelet_chat(payload)
    return response.model_dump()


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


@app.post('/v1/chat/completions')
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
        async def event_stream() -> AsyncGenerator[str, None]:
            content = response.synthesis.content
            completion_id = f'chatcmpl-{uuid.uuid4().hex}'
            # Stream in small character chunks to preserve markdown formatting.
            # Word-splitting destroys code blocks, newlines, and bullet points.
            chunk_size = 8
            for i in range(0, len(content), chunk_size):
                token = content[i:i + chunk_size]
                chunk = {
                    'id': completion_id,
                    'object': 'chat.completion.chunk',
                    'created': int(time.time()),
                    'model': 'roitelet-llm',
                    'choices': [{'index': 0, 'delta': {'content': token}, 'finish_reason': None}],
                }
                yield f'data: {json.dumps(chunk)}\n\n'
            done = {
                'id': completion_id,
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'roitelet-llm',
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
            }
            yield f'data: {json.dumps(done)}\n\n'
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
