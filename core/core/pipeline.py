"""End-to-end Roitelet orchestration pipeline.

This module wires together the router, provider clients, local synthesis,
conversation persistence, telemetry, and lightweight online Elo updates.

Examples
--------
>>> # Real network-backed inference is required for a full run.
>>> from core.core.pipeline import build_title
>>> build_title('How do I optimize this Python function?')
'How do I optimize this Python function?'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Sequence

from ..providers.factory import get_provider_client
from ..schemas import ChatMessage, ChatRequest, ChatResponse, ConversationMessage, ModelResponse, TelemetryRecord
from ..storage import storage
from .judge import judge_and_synthesize
from .registry import registry
from .router import RoiteletRouter

router = RoiteletRouter()


def build_title(prompt: str, max_length: int = 60) -> str:
    """Build a compact conversation title from the first prompt."""
    cleaned = ' '.join(prompt.strip().split())
    return cleaned[:max_length] or 'New flight'


async def _query_one(model_id: str, messages: Sequence[ChatMessage]) -> ModelResponse:
    """Query one registered model through the correct provider client."""
    spec = registry.get(model_id)
    client = get_provider_client(spec.provider)
    response = await client.generate(model_id=model_id, messages=messages)
    response.cost_usd = _estimate_cost(model_id, response)
    return response


def _estimate_cost(model_id: str, response: ModelResponse) -> float:
    """Estimate request cost from stored pricing priors and token usage."""
    spec = registry.get(model_id)
    usage = response.usage
    prompt_tokens = usage.get('prompt_tokens', usage.get('prompt_eval_count', 0.0))
    completion_tokens = usage.get('completion_tokens', usage.get('eval_count', 0.0))
    return (prompt_tokens / 1000.0) * spec.pricing['input_per_1k'] + (completion_tokens / 1000.0) * spec.pricing['output_per_1k']


async def run_roitelet_chat(request: ChatRequest) -> ChatResponse:
    """Run the full Roitelet prompt pipeline.

    Parameters
    ----------
    request:
        Native chat request with prompt and user preferences.

    Returns
    -------
    ChatResponse
        Routed, executed, judged, and persisted turn result.
    """
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

    selected_responses = await asyncio.gather(*[_query_one(model_id, messages) for model_id in decision.selected_model_ids])

    # Filter out responses that failed entirely before sending to the judge.
    # Always keep at least one candidate so synthesis has content to work with.
    valid_responses = [r for r in selected_responses if r.content and not r.error]
    if not valid_responses:
        valid_responses = list(selected_responses)  # all failed — let judge handle gracefully

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
        created_at=datetime.now(timezone.utc),
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

    return ChatResponse(
        conversation_id=conversation.conversation_id,
        router=decision,
        responses=list(selected_responses),
        synthesis=synthesis,
        telemetry_id=telemetry.record_id,
    )
