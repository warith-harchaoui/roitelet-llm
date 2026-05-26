"""FastAPI application for Roitelet LLM.

The API exposes three access styles:
1. a native Roitelet JSON API,
2. an OpenAI-compatible `/v1/chat/completions` endpoint,
3. a compact MCP-compatible JSON-RPC endpoint.

Examples
--------
>>> from api.main import app
>>> app.title
'Roitelet LLM API'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.commands import parse_command, render_help
from core.config import get_settings
from core.image_pipeline import NoImageProviderError, run_roitelet_image_chat
from core.mcp import handle_mcp_request
from core.personal import build_personal_context, ingest_inbox, personal_status, project_chunks_2d
from core.pipeline import AllCandidatesFailedError, run_roitelet_chat
from core.registry import warm_ollama_cache
from core.schemas import (
    AppSettingsPayload,
    ChatRequest,
    ImageGenRequest,
    MCPRequest,
    OpenAIChatCompletionRequest,
    OpenAIImagesRequest,
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
async def health() -> dict[str, Any]:
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
async def get_app_settings(storage: StorageManager = Depends(get_storage)) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Persist control-room settings edited from the web UI.

    Fields whose value still equals the secret mask sentinel are preserved
    from the previously stored payload — the UI may submit masked values
    untouched without wiping credentials.
    """
    stored = storage.load_app_settings()
    merged = stored.merge_unmasked(payload)
    storage.save_app_settings(merged)
    return {'status': 'saved'}


@app.get('/api/conversations', dependencies=[Depends(require_api_token)])
async def list_conversations(
    storage: StorageManager = Depends(get_storage),
) -> list[dict[str, Any]]:
    """List all stored conversations via the local storage manager."""
    return [conversation.model_dump() for conversation in storage.list_conversations()]


@app.get('/api/conversations/{conversation_id}', dependencies=[Depends(require_api_token)])
async def get_conversation(
    conversation_id: str,
    storage: StorageManager = Depends(get_storage),
) -> dict[str, Any]:
    """Fetch one conversation by its unique identifier."""
    conversation = storage.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail='Conversation not found')
    return conversation.model_dump()


@app.get('/api/telemetry', dependencies=[Depends(require_api_token)])
async def list_telemetry(
    storage: StorageManager = Depends(get_storage),
) -> list[dict[str, Any]]:
    """Return all telemetry records containing performance data."""
    return [record.model_dump() for record in storage.list_telemetry()]


def _sse(event: dict[str, Any]) -> str:
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


async def _run_chat_or_502(payload: ChatRequest):
    """Wrap ``run_roitelet_chat`` to convert all-fail into a clear HTTP 502.

    Without this, ``AllCandidatesFailedError`` would surface as a 500 with a
    stack trace — useless to the web UI. Mapping it to 502 (Bad Gateway)
    matches the semantic: every upstream model failed.
    """
    try:
        return await run_roitelet_chat(payload)
    except AllCandidatesFailedError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                'error': 'all_candidates_failed',
                'message': str(exc),
                'failures': [
                    {'model_id': r.model_id, 'error': r.error or 'empty response'}
                    for r in exc.responses
                ],
            },
        ) from exc


def _apply_command_overrides(payload: ChatRequest) -> tuple[ChatRequest, str | None]:
    """Strip leading slash commands and route or apply per-turn overrides.

    Returns
    -------
    (rewritten_payload, short_circuit_body)
        ``short_circuit_body`` is non-None only when the command short-
        circuits the pipeline (``/help`` today, possibly more later).
        Otherwise the caller proceeds with the rewritten payload.

    Raises
    ------
    HTTPException
        For commands that can't run on the chat endpoint
        (``/image`` and ``/speech`` need their dedicated endpoints).
    """
    parsed = parse_command(payload.prompt)
    if parsed.route_to == 'help':
        return payload, render_help()
    if parsed.route_to == 'image':
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'wrong_endpoint',
                'message': 'Use POST /api/images for /image prompts.',
                'route_to': 'image',
                'stripped_prompt': parsed.stripped_prompt,
            },
        )
    if parsed.route_to == 'speech':
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'wrong_endpoint',
                'message': 'Use POST /api/chat/multimodal with an audio attachment for /speech.',
                'route_to': 'speech',
            },
        )

    overrides: dict = {}
    # If /personal fired, splice the personal-knowledge-base context
    # in front of the (already-stripped) prompt before any other
    # rewrite. The injection respects the wiki-vs-RAG strategy in
    # ``core.personal``: small corpus → all in long context, large
    # corpus → top-K retrieval.
    if parsed.personal_override:
        ctx = build_personal_context(parsed.stripped_prompt or payload.prompt)
        if ctx:
            new_prompt = f'{ctx}\n## Question\n\n{parsed.stripped_prompt or payload.prompt}'
            overrides['prompt'] = new_prompt
        elif parsed.stripped_prompt != payload.prompt:
            overrides['prompt'] = parsed.stripped_prompt
    elif parsed.stripped_prompt != payload.prompt:
        overrides['prompt'] = parsed.stripped_prompt
    pref_updates: dict = {}
    if parsed.independence_override is not None:
        pref_updates['independence'] = parsed.independence_override
    if parsed.max_cost_usd_override is not None:
        pref_updates['max_cost_usd'] = parsed.max_cost_usd_override
    if pref_updates:
        overrides['preferences'] = payload.preferences.model_copy(update=pref_updates)
    if parsed.top_k_override is not None:
        overrides['top_k'] = parsed.top_k_override
    if not overrides:
        return payload, None
    return payload.model_copy(update=overrides), None


@app.post('/api/chat', dependencies=[Depends(require_api_token)])
async def roitelet_chat(payload: ChatRequest):
    """Run one native Roitelet chat turn.

    Leading slash commands (``/local``, ``/cheap``, ``/k``, ``/help``)
    are consumed here as inline per-turn overrides; ``/image`` and
    ``/speech`` are rejected with a 400 pointing at the right endpoint.
    See :mod:`core.commands` for the full catalogue.

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
    payload, short_circuit_body = _apply_command_overrides(payload)
    if short_circuit_body is not None:
        # Static response (today: /help). No fan-out, no judge, no Elo update.
        return {
            'conversation_id': '',
            'synthesis': {
                'model_id': 'roitelet-commands',
                'provider': 'local',
                'content': short_circuit_body,
                'judge_summary': '',
                'winning_model_ids': [],
            },
            'router': None,
            'responses': [],
            'telemetry_id': '',
        }

    response = await _run_chat_or_502(payload)
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


_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac'}
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.heic', '.heif'}
_PDF_EXTS = {'.pdf'}


def _modality_of(upload: UploadFile) -> str | None:
    """Classify an uploaded file as 'audio', 'image', 'pdf', or None.

    Trust MIME type first (browser-provided), fall back to extension so
    blank/wrong MIMEs still get routed correctly.
    """
    mime = (upload.content_type or '').lower()
    if mime.startswith('audio/'):
        return 'audio'
    if mime.startswith('image/'):
        return 'image'
    if mime == 'application/pdf':
        return 'pdf'
    ext = Path(upload.filename or '').suffix.lower()
    if ext in _AUDIO_EXTS:
        return 'audio'
    if ext in _IMAGE_EXTS:
        return 'image'
    if ext in _PDF_EXTS:
        return 'pdf'
    return None


@app.post('/api/chat/multimodal', dependencies=[Depends(require_api_token)])
async def roitelet_chat_multimodal(
    prompt: str = Form(''),
    conversation_id: str | None = Form(None),
    top_k: int = Form(3),
    allow_vlms: bool = Form(False),
    files: list[UploadFile] = File(default_factory=list),
):
    """Run one chat turn with attached audio, image, or PDF files.

    Each attachment is converted to text locally before the standard
    pipeline runs — the router, candidate fan-out, and judge remain
    text-only:

    * ``audio/*`` → whisper.cpp transcription + NeMo Sortformer diarization
    * ``image/*`` → local Ollama VLM caption (gated on ``allow_vlms``)
    * ``application/pdf`` → kreuzberg text extraction (with OCR fallback)

    Unknown file types are skipped with a note in the augmented prompt so
    the user can see what was ignored.
    """
    import tempfile

    augmentations: list[str] = []
    skipped: list[str] = []

    for upload in files:
        if not upload.filename:
            continue
        modality = _modality_of(upload)
        if modality is None:
            skipped.append(upload.filename)
            continue
        if modality == 'image' and not allow_vlms:
            skipped.append(f'{upload.filename} (vision disabled)')
            continue

        suffix = Path(upload.filename).suffix or {
            'audio': '.wav', 'image': '.png', 'pdf': '.pdf',
        }[modality]
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'wb') as fh:
                fh.write(await upload.read())
            tmp = Path(tmp_path)
            if modality == 'audio':
                from core.multimodal.audio import transcribe_audio
                text = await transcribe_audio(tmp)
                label = f'[Audio: {upload.filename}]'
            elif modality == 'image':
                from core.multimodal.image import describe_image
                text = await describe_image(tmp)
                label = f'[Image: {upload.filename}]'
            else:  # pdf
                from core.multimodal.pdf import extract_pdf
                text = await extract_pdf(tmp)
                label = f'[PDF: {upload.filename}]'
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if text:
            augmentations.append(f'{label}\n{text}')
        else:
            skipped.append(f'{upload.filename} (extraction empty)')

    if skipped:
        augmentations.append('[Note] Skipped: ' + ', '.join(skipped))

    augmented = '\n\n'.join([*augmentations, prompt]).strip() if augmentations else prompt
    if not augmented:
        raise HTTPException(status_code=400, detail='Empty prompt and no usable attachments.')

    response = await _run_chat_or_502(
        ChatRequest(
            prompt=augmented,
            conversation_id=conversation_id,
            top_k=top_k,
            preferences=RouterPreferences(allow_vlms=allow_vlms),
        )
    )
    return response.model_dump()


@app.get('/v1/models')
async def list_models() -> dict[str, Any]:
    """Expose the local OpenAI-compatible model inventory."""
    return {
        'object': 'list',
        'data': [
            {
                'id': 'roitelet-llm',
                'object': 'model',
                'owned_by': 'roitelet-llm',
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


@app.get('/api/personal', dependencies=[Depends(require_api_token)])
async def personal_state() -> dict[str, Any]:
    """Return personal-mode corpus stats.

    Returns
    -------
    dict
        Counts (inbox / wiki) and the active context strategy
        (``wiki`` for inline concat, ``rag`` for retrieval, ``empty``
        when the folders are empty).
    """
    return personal_status()


@app.get('/api/personal/embeddings', dependencies=[Depends(require_api_token)])
async def personal_embeddings() -> dict[str, Any]:
    """Return 2-D PCA-projected embeddings for every wiki chunk.

    Returns
    -------
    dict
        ``{'points': [...]}`` where each point carries ``path``,
        ``chunk_index``, ``text``, ``x``, ``y``. Empty ``points``
        list when the wiki is empty or the local embedding model
        (``nomic-embed-text`` by default) is unreachable.

    Notes
    -----
    Karpathy-style scatter: similar topics cluster spatially because
    cosine-near vectors project to nearby 2-D coordinates. The
    rendering layer (web SPA) draws the points as an SVG scatter.
    """
    return {'points': project_chunks_2d()}


@app.post('/api/personal/ingest', dependencies=[Depends(require_api_token)])
async def personal_ingest(force: bool = False) -> dict[str, Any]:
    """Walk ``data/personal/inbox/`` and convert each new file to a wiki entry.

    Parameters
    ----------
    force : bool, default=False
        Re-ingest every inbox file even if the manifest already
        recorded a conversion (useful after an extractor upgrade).

    Returns
    -------
    dict
        ``{'results': [{'source', 'wiki_path', 'modality', 'error'}, ...],
        'status': personal_status()}``.
    """
    results = await ingest_inbox(force=force)
    return {
        'results': [
            {
                'source': str(r.source),
                'wiki_path': str(r.wiki_path) if r.wiki_path else None,
                'modality': r.modality,
                'error': r.error,
            }
            for r in results
        ],
        'status': personal_status(),
    }


@app.post('/api/images', dependencies=[Depends(require_api_token)])
async def roitelet_images(payload: ImageGenRequest):
    """Run one image-generation turn.

    K=1 by design — image fusion isn't a well-defined operation, so the
    pipeline picks the strongest single ``image_gen``-capable
    candidate and returns its output. Bytes land under
    ``data/images/<uuid>.png`` and the response references them by
    path.
    """
    try:
        result = await run_roitelet_image_chat(payload)
    except NoImageProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.model_dump()


@app.post('/v1/images/generations', dependencies=[Depends(require_api_token)])
async def openai_images(payload: OpenAIImagesRequest):
    """OpenAI-compatible images endpoint.

    Subset of the OpenAI ``/v1/images/generations`` shape — clients
    that already speak that protocol can hit Roitelet with their
    existing code path.
    """
    try:
        result = await run_roitelet_image_chat(
            ImageGenRequest(
                prompt=payload.prompt,
                size=payload.size,
                n=payload.n,
            )
        )
    except NoImageProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        'created': int(time.time()),
        'data': [
            {'b64_json': None, 'url': None, 'roitelet_path': img.path, 'roitelet_error': img.error}
            for img in result.images
        ],
        'roitelet_metadata': result.model_dump(),
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

_ASSETS_DIR = Path(__file__).resolve().parent.parent / 'assets'
if _ASSETS_DIR.is_dir():
    app.mount('/assets', StaticFiles(directory=_ASSETS_DIR), name='assets')

# Serve generated images straight from disk. ``data/images/`` is
# populated by the image pipeline; mounting it lets the UI display
# them with a plain ``<img src="/data/images/<uuid>.png">``.
_IMAGES_DIR = settings.data_dir / 'images'
_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount('/data/images', StaticFiles(directory=_IMAGES_DIR), name='generated_images')

_WEB_DIR = Path(__file__).resolve().parent.parent / 'web'
if _WEB_DIR.is_dir():
    app.mount('/', StaticFiles(directory=_WEB_DIR, html=True), name='web')
