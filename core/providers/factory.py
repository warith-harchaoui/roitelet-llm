"""Factory for provider clients.

Each branch maps a registered provider string (the prefix of a
``model_id`` like ``"openai/gpt-4.1"``) to a concrete client. Adding a
new paid provider with an OpenAI-compatible HTTP API takes three lines
here plus a bootstrap entry — see :class:`OpenAICompatibleClient` for
the contract every remote provider must satisfy.
"""

from __future__ import annotations

from .. import storage as _storage_mod
from ..config import get_settings
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient


def get_provider_client(provider: str, model_id: str | None = None):
    """Return an initialized provider client.

    Parameters
    ----------
    provider : str
        Provider identifier such as ``"ollama"``, ``"openai"``,
        ``"openrouter"``, or ``"openai-compatible"``.
    model_id : str or None, optional
        Required for multi-engine dispatch on the ``openai-compatible``
        provider. When ``model_id`` starts with
        ``openai-compatible/<label>/...`` the factory looks up
        ``<label>`` in the runtime's ``custom_engines`` list and uses
        that engine's ``base_url`` + ``api_key``. When ``model_id`` is
        ``None`` or has no engine prefix (legacy
        ``openai-compatible/<model_name>``), it falls back to the
        single-endpoint settings (``openai_compatible_base_url`` +
        ``openai_compatible_api_key``).

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
        # Multi-engine dispatch: parse ``<label>`` from a model id of
        # shape ``openai-compatible/<label>/<model>`` and resolve the
        # matching ``custom_engines`` entry. Fall through to the
        # legacy single-endpoint settings if there's no prefix or no
        # matching engine.
        if model_id:
            suffix = model_id.removeprefix('openai-compatible/')
            if '/' in suffix:
                label = suffix.split('/', 1)[0]
                for engine in (runtime_settings.custom_engines or []):
                    if engine.label == label:
                        return OpenAICompatibleClient(
                            base_url=engine.base_url,
                            api_key=engine.api_key,
                            provider_name=f'openai-compatible/{label}',
                        )
        return OpenAICompatibleClient(
            base_url=runtime_settings.openai_compatible_base_url or settings.openai_compatible_base_url,
            api_key=runtime_settings.openai_compatible_api_key or settings.openai_compatible_api_key,
            provider_name='openai-compatible',
        )
    raise ValueError(f'Unsupported provider: {provider}')
