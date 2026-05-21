"""Factory for provider clients.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from ..config import get_settings
from ..storage import storage
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient


def get_provider_client(provider: str):
    """Return an initialized provider client.

    Parameters
    ----------
    provider:
        Provider identifier such as `'ollama'` or `'openrouter'`.

    Returns
    -------
    object
        Provider client implementing the `generate` method.
    """
    settings = get_settings()
    runtime_settings = storage.load_app_settings()
    if provider == 'ollama':
        return OllamaClient(base_url=runtime_settings.ollama_base_url or settings.local_llm_base_url)
    if provider == 'openrouter':
        return OpenAICompatibleClient(
            base_url=settings.openrouter_base_url,
            api_key=runtime_settings.openrouter_api_key or settings.openrouter_api_key,
            provider_name='openrouter',
        )
    if provider == 'openai-compatible':
        return OpenAICompatibleClient(
            base_url=runtime_settings.openai_compatible_base_url or settings.openai_compatible_base_url,
            api_key=runtime_settings.openai_compatible_api_key or settings.openai_compatible_api_key,
            provider_name='openai-compatible',
        )
    raise ValueError(f'Unsupported provider: {provider}')
