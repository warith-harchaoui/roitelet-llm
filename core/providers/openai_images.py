"""Image-generation client for OpenAI-compatible ``/v1/images/generations``.

The shape mirrors :class:`core.providers.openai_compatible.OpenAICompatibleClient`
but targets the images endpoint and writes the bytes to disk under
``data/images/<uuid>.png``. Returning a filesystem path (rather than
inline base64) keeps the rest of the pipeline JSON-friendly and lets
the FastAPI static-files mount serve images straight from disk.

Compatible providers tested by shape:

- OpenAI direct (``https://api.openai.com/v1``) — ``gpt-image-1``, DALL-E 3.
- OpenRouter image relays.
- Together / Fireworks image endpoints (where exposed in OpenAI shape).
- AUTOMATIC1111 ``sd-webui-openai-compatible-api`` extension.

For Stability AI's native ``/v2beta/`` endpoint or ComfyUI's
JSON-RPC graph, a dedicated client lives next to this one (not yet
implemented — see ``.private/IMAGEGEN.md`` §3 for sequencing).

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path

import httpx

from ..config import get_settings
from ..schemas import GeneratedImage, ImageGenResponse


def _images_dir() -> Path:
    """Where generated image bytes live on disk."""
    path = get_settings().data_dir / 'images'
    path.mkdir(parents=True, exist_ok=True)
    return path


class OpenAIImagesClient:
    """Async client for OpenAI-compatible image generation endpoints."""

    def __init__(self, base_url: str, api_key: str, provider_name: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_name = provider_name

    async def generate_image(
        self,
        model_id: str,
        prompt: str,
        size: str = '1024x1024',
        n: int = 1,
        conversation_id: str | None = None,
    ) -> ImageGenResponse:
        """Send one image-generation request and persist the bytes.

        The OpenAI shape returns either ``url`` or ``b64_json`` per
        choice. We default to b64 by passing ``response_format`` so we
        can write the file regardless of whether the provider hosts the
        result (URL-based providers expire links; local providers
        return base64 directly).
        """
        started = time.perf_counter()
        endpoint = f'{self.base_url}/images/generations'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': model_id.split('/', 1)[-1],
            'prompt': prompt,
            'n': max(1, int(n)),
            'size': size,
            'response_format': 'b64_json',
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            runtime = time.perf_counter() - started
            return ImageGenResponse(
                conversation_id=conversation_id or '',
                model_id=model_id,
                provider=self.provider_name,
                images=[
                    GeneratedImage(
                        path='',
                        model_id=model_id,
                        provider=self.provider_name,
                        error=str(exc),
                    )
                ],
                latency_s=runtime,
            )

        images: list[GeneratedImage] = []
        for choice in data.get('data', []):
            generated_uuid = uuid.uuid4().hex
            file_path = _images_dir() / f'{generated_uuid}.png'
            payload_bytes = await _fetch_image_bytes(choice)
            if payload_bytes is None:
                images.append(
                    GeneratedImage(
                        path='',
                        model_id=model_id,
                        provider=self.provider_name,
                        error='provider returned no usable image payload',
                    )
                )
                continue
            file_path.write_bytes(payload_bytes)
            images.append(
                GeneratedImage(
                    path=str(file_path),
                    model_id=model_id,
                    provider=self.provider_name,
                    revised_prompt=choice.get('revised_prompt'),
                )
            )

        runtime = time.perf_counter() - started
        return ImageGenResponse(
            conversation_id=conversation_id or '',
            model_id=model_id,
            provider=self.provider_name,
            images=images,
            latency_s=runtime,
        )


async def _fetch_image_bytes(choice: dict) -> bytes | None:
    """Resolve an OpenAI-shaped image choice to raw bytes.

    Providers respond with either ``b64_json`` (inline base64, the
    common path now) or ``url`` (a hosted link, often expiring). We
    handle both so the pipeline keeps working when a provider flips
    formats.
    """
    b64 = choice.get('b64_json')
    if b64:
        try:
            return base64.b64decode(b64)
        except Exception:
            return None
    url = choice.get('url')
    if url:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.content
        except Exception:
            return None
    return None


def get_image_client(provider: str) -> OpenAIImagesClient:
    """Pick an image-generation client by provider key.

    Mirrors :func:`core.providers.factory.get_provider_client` for the
    image side. Today the same endpoint shape covers every supported
    provider; future native clients (Stability v2beta, ComfyUI) would
    branch here.
    """
    from .. import storage as _storage_mod

    settings = get_settings()
    runtime_settings = _storage_mod.get_storage().load_app_settings()

    if provider == 'openai':
        return OpenAIImagesClient(
            base_url='https://api.openai.com/v1',
            api_key=runtime_settings.openai_api_key or settings.openai_api_key,
            provider_name='openai',
        )
    if provider == 'openrouter':
        return OpenAIImagesClient(
            base_url=settings.openrouter_base_url,
            api_key=runtime_settings.openrouter_api_key or settings.openrouter_api_key,
            provider_name='openrouter',
        )
    if provider == 'openai-compatible':
        return OpenAIImagesClient(
            base_url=runtime_settings.openai_compatible_base_url or settings.openai_compatible_base_url,
            api_key=runtime_settings.openai_compatible_api_key or settings.openai_compatible_api_key,
            provider_name='openai-compatible',
        )
    raise ValueError(f'Unsupported image-generation provider: {provider}')
