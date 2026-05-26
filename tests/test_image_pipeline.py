"""Hermetic tests for the image-generation pipeline.

The fan-out / fusion machinery does not apply here (K=1 by design),
so the surface we have to lock down is small:

1. ``_pick_image_model`` picks the strongest ``image_gen``-capable spec.
2. The independence and cost-budget filters work the same as on the text side.
3. ``NoImageProviderError`` raises cleanly when no image-gen model exists.
4. ``run_roitelet_image_chat`` writes a file under ``data/images/`` and
   records the path in the conversation message metadata.

The OpenAI Images client is stubbed via monkeypatch so no network call
is made and no Ollama dependency is required.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from core.schemas import GeneratedImage, ImageGenRequest, ImageGenResponse, RouterPreferences


def _seed_tmp(tmp_path: Path, with_image_model: bool = True) -> None:
    """Bootstrap a clean data dir with priors that include an image-gen model.

    Copies the real bootstrap then appends a fake ``image_gen``-capable
    entry. Setting the prior > 0 is the on-disk signal that the model
    can serve image generation requests.
    """
    src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
    target_bootstrap = tmp_path / 'bootstrap'
    target_bootstrap.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'model_priors.json', target_bootstrap / 'model_priors.json')

    if with_image_model:
        priors = json.loads((target_bootstrap / 'model_priors.json').read_text())
        priors['openai-compatible/test-image'] = {
            'provider': 'openai-compatible',
            'local': False,
            'vlm': False,
            'pricing': {'input_per_1k': 0.04, 'output_per_1k': 0.0},
            'latency_s': 6.0,
            'energy_kwh': 0.0008,
            'capabilities': {
                'coding': 0.0, 'math': 0.0, 'reasoning': 0.0,
                'writing': 0.0, 'analysis': 0.0,
                'vision': 0.0, 'multilingual': 0.0, 'long_context': 0.0,
                'image_gen': 0.95,
            },
        }
        (target_bootstrap / 'model_priors.json').write_text(json.dumps(priors))

    (tmp_path / 'images').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'conversations').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'telemetry').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'runtime').mkdir(parents=True, exist_ok=True)


def _reset_singletons() -> None:
    """Clear lru_cache-backed singletons so they pick up the new env."""
    from core.config import get_settings
    from core.registry import get_registry, ollama_cache
    from core.storage import get_storage

    get_settings.cache_clear()
    get_storage.cache_clear()
    get_registry.cache_clear()
    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()


class TestPickImageModel:
    """``_pick_image_model`` is the seam between routing and provider call."""

    def test_picks_image_gen_capable(self, tmp_path):
        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_image_model=True)
            _reset_singletons()

            from core.image_pipeline import _pick_image_model
            from core.registry import ModelRegistry
            from core.schemas import AppSettingsPayload

            registry = ModelRegistry(
                app_settings=AppSettingsPayload(
                    openai_compatible_api_key='sk-test',
                )
            )
            spec = _pick_image_model(registry, RouterPreferences())
            assert spec.model_id == 'openai-compatible/test-image'
            assert spec.capabilities['image_gen'] > 0
        _reset_singletons()

    def test_independence_drops_remote(self, tmp_path):
        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_image_model=True)
            _reset_singletons()

            from core.image_pipeline import NoImageProviderError, _pick_image_model
            from core.registry import ModelRegistry
            from core.schemas import AppSettingsPayload

            registry = ModelRegistry(
                app_settings=AppSettingsPayload(openai_compatible_api_key='sk-test')
            )
            # Independence mode + no local image-gen model → 503.
            with pytest.raises(NoImageProviderError):
                _pick_image_model(registry, RouterPreferences(independence=True))
        _reset_singletons()

    def test_no_image_model_raises(self, tmp_path):
        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_image_model=False)
            _reset_singletons()

            from core.image_pipeline import NoImageProviderError, _pick_image_model
            from core.registry import ModelRegistry

            registry = ModelRegistry()
            with pytest.raises(NoImageProviderError):
                _pick_image_model(registry, RouterPreferences())
        _reset_singletons()


class TestRunRoiteletImageChat:
    """End-to-end happy-path with a stubbed image-gen client."""

    @pytest.mark.asyncio
    async def test_end_to_end_writes_image_and_records_message(self, tmp_path, monkeypatch):
        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_image_model=True)
            _reset_singletons()

            # Persist sentinel keys so the registry doesn't auto-prune
            # the test image-gen entry.
            from core.schemas import AppSettingsPayload
            from core.storage import get_storage
            get_storage().save_app_settings(AppSettingsPayload(
                openai_compatible_api_key='sk-test',
                openai_compatible_base_url='https://example.invalid/v1',
            ))

            # Stub the image client: don't actually call any endpoint;
            # synthesise a one-pixel PNG and return its path.
            from core import image_pipeline as ip

            stub_path = tmp_path / 'images' / 'stub.png'
            stub_path.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG magic header

            class _StubClient:
                provider_name = 'openai-compatible'

                async def generate_image(self, **kwargs):
                    return ImageGenResponse(
                        conversation_id='',
                        model_id=kwargs['model_id'],
                        provider='openai-compatible',
                        images=[GeneratedImage(
                            path=str(stub_path),
                            model_id=kwargs['model_id'],
                            provider='openai-compatible',
                        )],
                        latency_s=0.1,
                    )

            monkeypatch.setattr(ip, 'get_image_client', lambda provider: _StubClient())

            response = await ip.run_roitelet_image_chat(
                ImageGenRequest(prompt='A small wren in a forest, oil painting.'),
            )

            assert response.conversation_id
            assert response.images
            assert response.images[0].path == str(stub_path)
            assert stub_path.exists()

            # The conversation should have one user + one assistant message.
            conversation = get_storage().get_conversation(response.conversation_id)
            assert conversation is not None
            roles = [m.role for m in conversation.messages]
            assert roles == ['user', 'assistant']
            assistant = conversation.messages[1]
            assert assistant.metadata['image_paths'] == [str(stub_path)]
            assert assistant.metadata['provider'] == 'openai-compatible'
        _reset_singletons()
