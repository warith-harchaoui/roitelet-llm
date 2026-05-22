"""Factory for provider clients.

Each branch maps a registered provider string (the prefix of a
``model_id`` like ``"openai/gpt-4.1"``) to a concrete client. Adding a
new paid provider with an OpenAI-compatible HTTP API takes three lines
here plus a bootstrap entry — see :class:`OpenAICompatibleClient` for
the contract every remote provider must satisfy.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from .. import storage as _storage_mod
from ..config import get_settings
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient


def get_provider_client(provider: str):
    """Return an initialized provider client.

    Parameters
    ----------
    provider:
        Provider identifier such as ``"ollama"``, ``"openai"``,
        ``"openrouter"``, or ``"openai-compatible"``.

    Returns
    -------
    object
        Provider client implementing the ``generate`` method.
    """
    settings = get_settings()
    runtime_settings = _storage_mod.get_storage().load_app_settings()
    if provider == 'ollama':
        return OllamaClient(base_url=runtime_settings.ollama_base_url or settings.local_llm_base_url)
    if provider == 'openai':
        # Direct OpenAI access. Set OPENAI_API_KEY (env or web UI) and a
        # bootstrap entry like ``openai/gpt-4.1`` becomes routable.
        return OpenAICompatibleClient(
            base_url='https://api.openai.com/v1',
            api_key=runtime_settings.openai_api_key or settings.openai_api_key,
            provider_name='openai',
        )
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
