"""Model registry, rolling Elo state, and live Ollama discovery for Roitelet.

The registry merges three model sources in priority order:

1. **Bootstrap priors** — curated benchmark-inspired metadata in
   ``data/bootstrap/model_priors.json``. These are the most accurate and
   should not be overridden by dynamic sources.

2. **User-configured models** — ``selected_ollama_models`` and
   ``paid_openrouter_models`` saved from the web control room. These
   are merged at every route call (no restart required).

3. **Live Ollama discovery** — models returned by ``GET /api/tags`` on the
   local Ollama server. This is fetched once at FastAPI startup and then
   refreshed lazily every 60 seconds. New models pulled via ``ollama pull``
   appear in the router within one TTL window without any configuration.

Elo state keys are restricted to ``KNOWN_CAPABILITIES`` to prevent unbounded
growth from typos or novel capability strings.

Examples
--------
>>> from core.registry import ModelRegistry
>>> registry = ModelRegistry()
>>> len(registry.list_models()) >= 1
True
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

# Capabilities the router recognises. Elo keys are restricted to this set plus
# 'global' so that typos or novel capability strings cannot pollute the state file.
KNOWN_CAPABILITIES: set[str] = {
    'coding', 'math', 'reasoning', 'writing', 'analysis',
    'vision', 'multilingual', 'long_context',
    # ``image_gen`` gates the image-generation pipeline. A bootstrap
    # entry with ``image_gen > 0`` becomes routable by the K=1 image
    # pipeline in ``core.image_pipeline``.
    'image_gen',
}

# Default capability profile for a generic chat model not in the bootstrap file.
_DEFAULT_CAPABILITIES: dict[str, float] = {cap: 0.65 for cap in KNOWN_CAPABILITIES}

# Default pricing / latency / energy for dynamically-added models.
_OLLAMA_DEFAULTS: dict[str, Any] = {
    'provider': 'ollama',
    'local': True,
    'vlm': False,
    'pricing': {'input_per_1k': 0.0, 'output_per_1k': 0.0},
    'latency_s': 4.0,
    'energy_kwh': 0.0009,
}
_OPENROUTER_DEFAULTS: dict[str, Any] = {
    'provider': 'openrouter',
    'local': False,
    'vlm': False,
    'pricing': {'input_per_1k': 0.003, 'output_per_1k': 0.009},
    'latency_s': 4.5,
    'energy_kwh': 0.00060,
}
# Generic OpenAI-compatible endpoint defaults. Conservative pricing
# stand-in — the operator should override the prior with a real cost
# estimate via the bootstrap file once they know the provider's
# published rate. Defaults skew toward "moderately priced cloud LLM"
# so the cost-budget regime can still gate it sanely.
_OPENAI_COMPATIBLE_DEFAULTS: dict[str, Any] = {
    'provider': 'openai-compatible',
    'local': False,
    'vlm': False,
    'pricing': {'input_per_1k': 0.002, 'output_per_1k': 0.006},
    'latency_s': 4.0,
    'energy_kwh': 0.00060,
}

# TTL for the live Ollama model cache (seconds). After this delay the next
# route call will trigger a background refresh.
_OLLAMA_CACHE_TTL_S: float = 60.0


# ---------------------------------------------------------------------------
# Live Ollama model cache
# ---------------------------------------------------------------------------

class _OllamaModelCache:
    """Thread-safe-ish cache for live Ollama model names.

    The cache is populated at application startup via :func:`warm_ollama_cache`
    and then lazily refreshed after :data:`_OLLAMA_CACHE_TTL_S` seconds on
    the next read.  A synchronous ``httpx`` call is used deliberately so the
    refresh is compatible with both async (startup) and sync (import-time)
    contexts without introducing an event-loop dependency.
    """

    def __init__(self) -> None:
        self._models: list[str] = []
        self._fetched_at: float = 0.0
        self._base_url: str = ''

    def configure(self, base_url: str) -> None:
        """Set the Ollama base URL (must be called before first read)."""
        self._base_url = base_url.rstrip('/')

    def _fetch(self) -> list[str]:
        """Fetch the live model list from Ollama synchronously."""
        if not self._base_url:
            return []
        try:
            response = httpx.get(
                f'{self._base_url}/api/tags',
                timeout=5.0,
            )
            response.raise_for_status()
            names = [item['name'] for item in response.json().get('models', [])]
            logger.debug('Live Ollama discovery: %d model(s) found.', len(names))
            return names
        except Exception as exc:
            logger.debug('Ollama discovery skipped (%s).', exc)
            return []

    def refresh(self, force: bool = False) -> None:
        """Refresh the cache if TTL has expired or ``force=True``."""
        if force or (time.monotonic() - self._fetched_at) > _OLLAMA_CACHE_TTL_S:
            self._models = self._fetch()
            self._fetched_at = time.monotonic()

    @property
    def models(self) -> list[str]:
        """Return cached model names, refreshing lazily if stale."""
        self.refresh()
        return list(self._models)


# Module-level cache instance shared across all registry objects.
ollama_cache = _OllamaModelCache()


def warm_ollama_cache(base_url: str, force: bool = True) -> None:
    """Warm the live Ollama model cache eagerly.

    Intended to be called from the FastAPI ``lifespan`` startup hook so that
    the first request after server boot does not incur discovery latency.

    Parameters
    ----------
    base_url:
        Ollama server base URL (e.g. ``'http://localhost:11434'``).
    force:
        When ``True`` (default) fetch unconditionally even if the cache is
        still fresh.
    """
    ollama_cache.configure(base_url)
    ollama_cache.refresh(force=force)
    logger.info('Ollama cache warmed: %d model(s).', len(ollama_cache.models))


# ---------------------------------------------------------------------------
# Model data class
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelSpec:
    """Static and slowly-changing metadata for one candidate model."""

    model_id: str
    provider: str
    capabilities: dict[str, float]
    pricing: dict[str, float]
    latency_s: float
    energy_kwh: float
    local: bool
    vlm: bool


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Registry of candidate models plus rolling Elo overrides.

    Model sources (merged in this order, lower-priority → higher-priority):
    1. Bootstrap JSON priors (most curated — never overwritten).
    2. User-configured models from the web control room.
    3. Live-discovered Ollama models (via the module-level cache).
    """

    def __init__(self, app_settings=None) -> None:
        """Load model priors and any saved online adjustments.

        Parameters
        ----------
        app_settings:
            Optional :class:`~core.schemas.AppSettingsPayload` pre-loaded from
            disk. When ``None`` the registry loads it lazily from storage.
        """
        settings = get_settings()
        self.bootstrap_path = settings.data_dir / 'bootstrap' / 'model_priors.json'
        self.elo_path = settings.data_dir / 'runtime' / 'elo_state.json'
        if not self.bootstrap_path.exists():
            raise FileNotFoundError(f'Bootstrap priors not found: {self.bootstrap_path}')
        payload = json.loads(self.bootstrap_path.read_text(encoding='utf-8'))
        self.models: dict[str, ModelSpec] = {
            model_id: ModelSpec(
                model_id=model_id,
                provider=model_payload['provider'],
                capabilities=model_payload['capabilities'],
                pricing=model_payload['pricing'],
                latency_s=model_payload['latency_s'],
                energy_kwh=model_payload['energy_kwh'],
                local=model_payload['local'],
                vlm=model_payload['vlm'],
            )
            for model_id, model_payload in payload.items()
        }
        self.elo_state = self._load_elo_state()

        # Merge external model sources.
        if app_settings is not None:
            self._merge_user_models(app_settings)
            ollama_base_url = getattr(app_settings, 'ollama_base_url', '')
        else:
            ollama_base_url = settings.local_llm_base_url

        # Configure and lazily trigger the live-discovery cache.
        if ollama_base_url:
            ollama_cache.configure(ollama_base_url)
        self._merge_live_ollama()
        self._prune_unauthorized_remotes(app_settings)

    # ------------------------------------------------------------------
    # Model injection
    # ------------------------------------------------------------------

    def _merge_user_models(self, app_settings) -> None:  # type: ignore[no-untyped-def]
        """Inject control-room configured Ollama and OpenRouter models."""
        for model_name in (app_settings.selected_ollama_models or []):
            model_id = f'ollama/{model_name}' if not model_name.startswith('ollama/') else model_name
            if model_id not in self.models:
                self.models[model_id] = ModelSpec(
                    model_id=model_id,
                    provider='ollama',
                    local=True,
                    vlm=False,
                    pricing={'input_per_1k': 0.0, 'output_per_1k': 0.0},
                    latency_s=_OLLAMA_DEFAULTS['latency_s'],
                    energy_kwh=_OLLAMA_DEFAULTS['energy_kwh'],
                    capabilities=dict(_DEFAULT_CAPABILITIES),
                )
        for model_name in (app_settings.paid_openrouter_models or []):
            model_id = f'openrouter/{model_name}' if not model_name.startswith('openrouter/') else model_name
            if model_id not in self.models:
                self.models[model_id] = ModelSpec(
                    model_id=model_id,
                    provider='openrouter',
                    local=False,
                    vlm=False,
                    pricing=dict(_OPENROUTER_DEFAULTS['pricing']),
                    latency_s=_OPENROUTER_DEFAULTS['latency_s'],
                    energy_kwh=_OPENROUTER_DEFAULTS['energy_kwh'],
                    capabilities=dict(_DEFAULT_CAPABILITIES),
                )
        # Multi-engine OpenAI-compatible registration. Each engine in
        # ``custom_engines`` has its own ``label / base_url / api_key
        # / models`` and registers as
        # ``openai-compatible/<label>/<model_name>`` so two engines
        # serving the same model name (e.g. both Together and
        # Anyscale serving ``mistralai/Mistral-7B``) don't collide.
        for engine in (getattr(app_settings, 'custom_engines', None) or []):
            label = (engine.label or '').strip()
            if not label:
                continue
            for model_name in (engine.models or []):
                model_id = f'openai-compatible/{label}/{model_name}'
                if model_id not in self.models:
                    self.models[model_id] = ModelSpec(
                        model_id=model_id,
                        provider='openai-compatible',
                        local=False,
                        vlm=False,
                        pricing=dict(_OPENAI_COMPATIBLE_DEFAULTS['pricing']),
                        latency_s=_OPENAI_COMPATIBLE_DEFAULTS['latency_s'],
                        energy_kwh=_OPENAI_COMPATIBLE_DEFAULTS['energy_kwh'],
                        capabilities=dict(_DEFAULT_CAPABILITIES),
                    )

    def _merge_live_ollama(self) -> None:
        """Reconcile the model pool with what Ollama actually serves.

        Two responsibilities:

        1. **Add** models reported by ``/api/tags`` that bootstrap and
           user config don't already cover (with conservative default
           priors).
        2. **Drop** ``provider='ollama'`` bootstrap entries that the user
           hasn't actually pulled — otherwise the router happily picks a
           high-prior model whose tag 404s at inference time. This filter
           only runs when discovery actually returned a list; if Ollama
           is unreachable the cache is empty and we leave bootstrap alone
           so users can still pre-load priors before pulling.
        """
        discovered = list(ollama_cache.models)
        # Add live-discovered models that nothing else has registered yet.
        for model_name in discovered:
            model_id = f'ollama/{model_name}' if not model_name.startswith('ollama/') else model_name
            if model_id not in self.models:
                logger.debug('Live-discovered Ollama model registered: %s', model_id)
                self.models[model_id] = ModelSpec(
                    model_id=model_id,
                    provider='ollama',
                    local=True,
                    vlm=False,
                    pricing={'input_per_1k': 0.0, 'output_per_1k': 0.0},
                    latency_s=_OLLAMA_DEFAULTS['latency_s'],
                    energy_kwh=_OLLAMA_DEFAULTS['energy_kwh'],
                    capabilities=dict(_DEFAULT_CAPABILITIES),
                )
        # Drop bootstrap ``ollama/*`` entries the user hasn't pulled.
        # Skip the prune when discovery is empty (Ollama unreachable) so
        # offline pre-flight planning still sees the full curated pool.
        if discovered:
            discovered_ids = {
                f'ollama/{name}' if not name.startswith('ollama/') else name
                for name in discovered
            }
            stale = [
                mid for mid, spec in self.models.items()
                if spec.provider == 'ollama' and mid not in discovered_ids
            ]
            for model_id in stale:
                logger.debug('Dropping un-pulled Ollama model: %s', model_id)
                self.models.pop(model_id, None)

    def _prune_unauthorized_remotes(self, app_settings) -> None:  # type: ignore[no-untyped-def]
        """Drop non-local model specs whose provider has no API key set.

        Why: otherwise a high-prior remote (e.g. ``openai/gpt-4.1``) wins
        routing, the provider call 401s, and the synthesis judge has nothing
        to fuse — surfacing as "(no answer)" in the UI.

        ``openai-compatible/<label>/<model>`` specs are checked
        per-engine: only dropped if the matching ``custom_engines``
        entry has an empty ``api_key``. Bare ``openai-compatible/<model>``
        ids (no label) are still gated by the single
        ``openai_compatible_api_key`` since the same endpoint is used
        for image generation via :mod:`core.providers.openai_images`.
        """
        settings = get_settings()
        runtime = app_settings
        def _key(*names: str) -> str:
            for n in names:
                val = getattr(runtime, n, '') if runtime is not None else ''
                if val:
                    return val
                val = getattr(settings, n, '') if settings is not None else ''
                if val:
                    return val
            return ''
        provider_keys = {
            'openai': _key('openai_api_key'),
            'openrouter': _key('openrouter_api_key'),
            'openai-compatible': _key('openai_compatible_api_key'),
            'anthropic': _key('anthropic_api_key'),
            'gemini': _key('gemini_api_key'),
            'perplexity': _key('perplexity_api_key'),
        }
        engines_by_label = {
            e.label: e for e in (getattr(runtime, 'custom_engines', None) or [])
        }

        def _is_authorized(mid: str, spec: ModelSpec) -> bool:
            if spec.local or spec.provider not in provider_keys:
                return True
            if spec.provider != 'openai-compatible':
                return bool(provider_keys[spec.provider])
            # ``openai-compatible/<label>/<model>`` → check engine.api_key.
            suffix = mid.removeprefix('openai-compatible/')
            head = suffix.split('/', 1)[0] if '/' in suffix else ''
            if head and head in engines_by_label:
                return bool(engines_by_label[head].api_key)
            # No engine prefix → legacy single-endpoint case.
            return bool(provider_keys['openai-compatible'])

        to_drop = [mid for mid, spec in self.models.items() if not _is_authorized(mid, spec)]
        for mid in to_drop:
            logger.debug('Dropping remote model without API key: %s', mid)
            self.models.pop(mid, None)

    # ------------------------------------------------------------------
    # Elo state helpers
    # ------------------------------------------------------------------

    def _load_elo_state(self) -> dict[str, dict[str, float]]:
        """Load rolling Elo adjustments from disk."""
        if not self.elo_path.exists():
            state = {model_id: {'global': 0.0} for model_id in self.models}
            self._save_elo_state(state)
            return state
        return json.loads(self.elo_path.read_text(encoding='utf-8'))

    def _save_elo_state(self, state: dict[str, dict[str, float]]) -> None:
        """Persist Elo adjustments to disk."""
        self.elo_path.parent.mkdir(parents=True, exist_ok=True)
        self.elo_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelSpec]:
        """Return all registered model specs."""
        return list(self.models.values())

    def get(self, model_id: str) -> ModelSpec:
        """Return one model spec by identifier."""
        return self.models[model_id]

    def capability_score(self, model_id: str, capability: str) -> float:
        """Combine bootstrap capability prior with rolling Elo adjustment.

        Parameters
        ----------
        model_id : str
            The registered model identifier.
        capability : str
            The name of the capability to query.

        Returns
        -------
        float
            A clamped Elo score floating point value between 0.0 and 1.5.
        """
        spec = self.get(model_id)
        prior = spec.capabilities.get(capability, 0.45)
        adjustment = self.elo_state.get(model_id, {}).get(capability, 0.0)
        global_adjustment = self.elo_state.get(model_id, {}).get('global', 0.0)
        # Clamp to [0.0, 1.5]: the 1.5 ceiling lets winners overshoot the
        # nominal prior range [0, 1] but caps runaway accumulation so Elo
        # never fully drowns out the bootstrap prior.
        return max(0.0, min(1.5, prior + adjustment + 0.5 * global_adjustment))

    def update_elo(
        self,
        winners: Iterable[str],
        losers: Iterable[str],
        capabilities: dict[str, float],
        k_factor: float = 0.04,
    ) -> None:
        """Apply a simple rolling Elo-style update.

        Parameters
        ----------
        winners:
            Model identifiers considered stronger for the observed prompt.
        losers:
            Model identifiers considered weaker for the observed prompt.
        capabilities:
            Weighted capability distribution for the prompt.
        k_factor:
            Update strength. Kept intentionally small for stability.
        """
        winners = list(winners)
        losers = list(losers)
        if not winners or not losers:
            return
        for model_id in set(winners + losers):
            self.elo_state.setdefault(model_id, {'global': 0.0})
        winner_delta = k_factor / max(1, len(winners))
        loser_delta = k_factor / max(1, len(losers))
        for winner in winners:
            self.elo_state[winner]['global'] = self.elo_state[winner].get('global', 0.0) + winner_delta
            for capability, weight in capabilities.items():
                if capability not in KNOWN_CAPABILITIES:
                    continue
                self.elo_state[winner][capability] = self.elo_state[winner].get(capability, 0.0) + winner_delta * weight
        for loser in losers:
            self.elo_state[loser]['global'] = self.elo_state[loser].get('global', 0.0) - loser_delta
            for capability, weight in capabilities.items():
                if capability not in KNOWN_CAPABILITIES:
                    continue
                self.elo_state[loser][capability] = self.elo_state[loser].get(capability, 0.0) - loser_delta * weight
        self._save_elo_state(self.elo_state)


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    """Return the process-wide :class:`ModelRegistry` instance.

    Cached to avoid rebuilding bootstrap+Elo state on every import; the
    registry itself rebuilds candidate lists per ``route()`` call so
    live-discovered Ollama models still appear without a restart.
    """
    return ModelRegistry()


def __getattr__(name: str):
    """Backwards-compatible lazy access for ``from core.registry import registry``."""
    if name == 'registry':
        return get_registry()
    raise AttributeError(f"module 'core.registry' has no attribute {name!r}")
