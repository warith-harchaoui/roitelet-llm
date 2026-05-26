"""End-to-end image-generation pipeline.

Sister module to :mod:`core.pipeline`. The text pipeline does top-K
fan-out + fusion; image generation is **K=1** because there's no
meaningful "fuse three images into one" operation. We pick the
strongest single image-gen model, run it, persist the bytes, and log
telemetry.

Design fits the existing seams (the same router, the same telemetry
schema with slightly different metadata, the same storage manager) so
adding image-gen doesn't require parallel infrastructure. What is new
is:

- a small registry filter that surfaces only ``image_gen``-capable
  models when the prompt is image-y,
- a `K=1` selection that ignores ``ChatRequest.top_k``,
- skipping the judge (no fusion).

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from . import storage as _storage_mod
from .capabilities import detect_capabilities, top_capabilities
from .providers.openai_images import get_image_client
from .registry import ModelRegistry
from .schemas import (
    ConversationMessage,
    ImageGenRequest,
    ImageGenResponse,
    RouterPreferences,
)


class NoImageProviderError(RuntimeError):
    """No registered model exposes a non-zero ``image_gen`` prior.

    Raised when the user asks for an image but the registry contains
    only text models. The API layer translates this to HTTP 503 so the
    UI can surface a clean "image generation isn't configured" hint.
    """


def _image_capable(spec) -> bool:
    """A model is image-capable when its priors expose a non-zero ``image_gen`` score.

    The bootstrap entries don't include ``image_gen`` for any model yet
    (the feature scaffold). Operators add an image-gen model the same
    way they add any other paid LLM: via ``paid_openai_compatible_models``
    plus a hand-edited bootstrap entry with ``image_gen`` set.
    """
    return spec.capabilities.get('image_gen', 0.0) > 0.0


def _pick_image_model(registry: ModelRegistry, preferences: RouterPreferences):
    """Select the strongest image-gen candidate subject to preferences.

    Same filter semantics as the text router: ``independence`` drops
    remote candidates, ``max_cost_usd`` drops expensive ones. The
    scoring is intentionally simple — image-gen models don't have the
    rich capability decomposition text models do, so we sort by the
    flat ``image_gen`` prior.
    """
    eligible = []
    for spec in registry.list_models():
        if not _image_capable(spec):
            continue
        if preferences.independence and not spec.local:
            continue
        estimated_cost = spec.pricing['input_per_1k'] + spec.pricing['output_per_1k']
        if (
            preferences.max_cost_usd is not None
            and estimated_cost > preferences.max_cost_usd
        ):
            continue
        eligible.append(spec)

    if not eligible:
        raise NoImageProviderError(
            'No image-generation model registered (set ``image_gen > 0`` in a '
            'bootstrap entry, or add an OpenAI-compatible image provider).'
        )

    eligible.sort(key=lambda s: s.capabilities.get('image_gen', 0.0), reverse=True)
    return eligible[0]


async def run_roitelet_image_chat(request: ImageGenRequest) -> ImageGenResponse:
    """Run the full image-generation pipeline.

    Returns
    -------
    ImageGenResponse
        Carries the conversation id, the picked model, the path of
        each generated image, and the wall-clock latency.
    """
    storage = _storage_mod.get_storage()
    conversation = (
        storage.get_conversation(request.conversation_id)
        if request.conversation_id
        else None
    )
    if conversation is None:
        title = request.prompt[:60] or 'Image'
        conversation = storage.create_conversation(title=title)
    storage.append_message(
        conversation.conversation_id,
        ConversationMessage(role='user', content=request.prompt),
    )

    app_settings = storage.load_app_settings()
    registry = ModelRegistry(app_settings=app_settings)
    spec = _pick_image_model(registry, request.preferences)

    # Surface the routing rationale in the response for telemetry.
    categories = detect_capabilities(request.prompt)
    rationale = (
        f'image_gen pipeline · capability_top={top_capabilities(categories)} · '
        f'model={spec.model_id} · provider={spec.provider}'
    )

    client = get_image_client(spec.provider)
    response = await client.generate_image(
        model_id=spec.model_id,
        prompt=request.prompt,
        size=request.size,
        n=request.n,
        conversation_id=conversation.conversation_id,
    )

    # Record the assistant message — the image *paths* live in
    # metadata so the conversation viewer can render them. The
    # ``content`` is a human-readable summary so non-image-aware UIs
    # still show something useful.
    summary = (
        f'[image: {len(response.images)} generated by {spec.model_id}]'
        if response.images
        else '[image: generation failed]'
    )
    storage.append_message(
        conversation.conversation_id,
        ConversationMessage(
            role='assistant',
            content=summary,
            metadata={
                'image_paths': [img.path for img in response.images if img.path],
                'errors': [img.error for img in response.images if img.error],
                'model_id': spec.model_id,
                'provider': spec.provider,
                'rationale': rationale,
                'created_at': datetime.now(UTC).isoformat(),
                'turn_id': str(uuid.uuid4()),
            },
        ),
    )
    response.conversation_id = conversation.conversation_id
    return response
