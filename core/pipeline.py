"""End-to-end Roitelet orchestration pipeline.

This module wires together the router, provider clients, local synthesis,
conversation persistence, telemetry, and lightweight online Elo updates.

Examples
--------
>>> # Real network-backed inference is required for a full run.
>>> from core.pipeline import build_title
>>> build_title('How do I optimize this Python function?')
'How do I optimize this Python function?'
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache

from . import registry as _registry_mod
from . import storage as _storage_mod
from .judge import judge_and_synthesize
from .providers.factory import get_provider_client
from .router import RoiteletRouter
from .router_protocol import Router
from .schemas import ChatMessage, ChatRequest, ChatResponse, ConversationMessage, ModelResponse, TelemetryRecord


class AllCandidatesFailedError(RuntimeError):
    """Raised when every routed model failed and there is nothing for the judge to fuse.

    Carries the raw per-model errors so the API layer can surface a clear
    explanation to the user instead of fabricating a synthesized answer
    from empty content.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        details = ', '.join(
            f'{r.model_id}: {r.error or "empty response"}' for r in self.responses
        ) or 'no candidates returned'
        super().__init__(f'All routed models failed — {details}')


@lru_cache(maxsize=1)
def get_router() -> RoiteletRouter:
    """Return the process-wide :class:`RoiteletRouter` instance.

    The router is stateless (every ``route()`` call rebuilds candidates
    from the live registry), so a singleton is safe and cheap.
    """
    return RoiteletRouter()


def __getattr__(name: str):
    """Backwards-compatible lazy access for ``from core.pipeline import router``."""
    if name == 'router':
        return get_router()
    raise AttributeError(f"module 'core.pipeline' has no attribute {name!r}")


def build_title(prompt: str, max_length: int = 60) -> str:
    """Build a compact conversation title from the first prompt.

    Parameters
    ----------
    prompt : str
        The initial query sent to the LLM.
    max_length : int, default=60
        Max string length for the truncated title.

    Returns
    -------
    str
        A short alphanumeric snippet suitable for sidebar display.
    """
    cleaned = ' '.join(prompt.strip().split())
    return cleaned[:max_length] or 'New flight'


async def _query_one(model_id: str, messages: Sequence[ChatMessage]) -> ModelResponse:
    """Query one registered model through the correct provider client.

    Parameters
    ----------
    model_id : str
        The unique registered string identifier.
    messages : Sequence[ChatMessage]
        The fully baked sequence of textual context or user instructions.

    Returns
    -------
    ModelResponse
        A populated unified schema model output payload with injected cost logic.
    """
    spec = _registry_mod.get_registry().get(model_id)
    client = get_provider_client(spec.provider)
    response = await client.generate(model_id=model_id, messages=messages)
    response.cost_usd = _estimate_cost(model_id, response)
    return response


def _estimate_cost(model_id: str, response: ModelResponse) -> float:
    """Estimate request cost from stored pricing priors and token usage.

    Parameters
    ----------
    model_id : str
        The target identifier fetched from the registry.
    response : ModelResponse
        The payload returned by provider APIs, exposing token metrics.

    Returns
    -------
    float
        Total sum of inference transactions in USD.
    """
    spec = _registry_mod.get_registry().get(model_id)
    usage = response.usage
    prompt_tokens = usage.get('prompt_tokens', usage.get('prompt_eval_count', 0.0))
    completion_tokens = usage.get('completion_tokens', usage.get('eval_count', 0.0))
    return (
        (prompt_tokens / 1000.0) * spec.pricing['input_per_1k']
        + (completion_tokens / 1000.0) * spec.pricing['output_per_1k']
    )


async def run_roitelet_chat(
    request: ChatRequest,
    router: Router | None = None,
) -> ChatResponse:
    """Run the full Roitelet prompt pipeline.

    Parameters
    ----------
    request:
        Native chat request with prompt and user preferences.
    router:
        Optional override for the :class:`Router` implementation. Defaults
        to the singleton returned by :func:`get_router` — pass a custom
        one (a learned classifier, an A/B router, a test double) without
        touching globals.

    Returns
    -------
    ChatResponse
        Routed, executed, judged, and persisted turn result.
    """
    # Wall-clock start: the canonical user-perceived latency reported
    # on the returned ChatResponse. Includes router decision, candidate
    # fan-out (bounded by the slowest candidate), judge synthesis, Elo
    # update and telemetry persistence — every step the user waits on.
    turn_started = time.perf_counter()

    storage = _storage_mod.get_storage()
    registry = _registry_mod.get_registry()
    if router is None:
        router = get_router()

    conversation = storage.get_conversation(request.conversation_id) if request.conversation_id else None
    if conversation is None:
        conversation = storage.create_conversation(title=build_title(request.prompt))
    storage.append_message(
        conversation.conversation_id,
        ConversationMessage(role='user', content=request.prompt),
    )

    decision = router.route(request.prompt, request.preferences, top_k=request.top_k)
    shadow_reference = [candidate.model_id for candidate in decision.candidates[:max(request.top_k, 5)]]
    messages = [ChatMessage(role='user', content=request.prompt)]

    selected_responses = await asyncio.gather(
        *[_query_one(model_id, messages) for model_id in decision.selected_model_ids]
    )

    # Drop responses that failed entirely before sending to the judge.
    valid_responses = [r for r in selected_responses if r.content and not r.error]
    if not valid_responses:
        # Every routed model failed. Do NOT call the judge — there is
        # nothing to fuse and fabricating a "synthesis" from empty content
        # would be dishonest. Surface a real error to the API layer.
        raise AllCandidatesFailedError(selected_responses)

    synthesis = await judge_and_synthesize(request.prompt, valid_responses)

    winners = synthesis.winning_model_ids
    losers = [response.model_id for response in selected_responses if response.model_id not in winners]
    registry.update_elo(winners=winners, losers=losers, capabilities=decision.categories)

    assistant_payload = {
        'router': decision.model_dump(),
        'responses': [response.model_dump() for response in selected_responses],
        'synthesis': synthesis.model_dump(),
    }
    storage.append_message(
        conversation.conversation_id,
        ConversationMessage(role='assistant', content=synthesis.content, metadata=assistant_payload),
    )

    telemetry = TelemetryRecord(
        record_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC),
        conversation_id=conversation.conversation_id,
        prompt=request.prompt,
        router_decision=decision,
        model_responses=list(selected_responses),
        synthesis=synthesis,
        reward_model_ids=winners,
        shadow_reference_model_ids=shadow_reference,
        metadata={
            'shadow_full_pool': request.shadow_full_pool,
            'top_k': request.top_k,
        },
    )
    storage.save_telemetry(telemetry)


    total_latency_s = time.perf_counter() - turn_started
    return ChatResponse(
        conversation_id=conversation.conversation_id,
        router=decision,
        responses=list(selected_responses),
        synthesis=synthesis,
        telemetry_id=telemetry.record_id,
        total_latency_s=total_latency_s,
    )
