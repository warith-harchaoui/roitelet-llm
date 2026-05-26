"""Client for local Ollama models.

Examples
--------
>>> from core.providers.ollama import OllamaClient
>>> client = OllamaClient(base_url='http://localhost:11434')
>>> client.base_url.startswith('http')
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from ..config import get_settings
from ..device import detect_best_accelerator
from ..energy import estimate_energy_and_carbon
from ..schemas import ChatMessage, ModelResponse


class OllamaClient:
    """Async Ollama API client used for local candidates and synthesis."""

    def __init__(self, base_url: str) -> None:
        """Initialize the client with the Ollama server base URL."""
        self.base_url = base_url.rstrip('/')

    async def generate(self, model_id: str, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Generate one chat completion using the Ollama REST API.

        Parameters
        ----------
        model_id : str
            The target local model format (e.g. `ollama/phi3`).
        messages : Sequence[ChatMessage]
            Chat thread history used to ground the LLM completion.

        Returns
        -------
        ModelResponse
            Inference response packaged with energy estimations.
        """
        started = time.perf_counter()
        settings = get_settings()
        endpoint = f'{self.base_url}/api/chat'
        payload = {
            'model': model_id.split('/', 1)[-1],
            'messages': [message.model_dump() for message in messages],
            'stream': False,
        }
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
            runtime = time.perf_counter() - started
            accelerator = detect_best_accelerator()
            on_accelerator = accelerator in {'mps', 'cuda'}
            average_power = (
                settings.default_gpu_power_watts if on_accelerator
                else settings.default_cpu_power_watts
            )
            energy_kwh, carbon_g = estimate_energy_and_carbon(
                runtime, average_power_watts=average_power, memory_gb=6.0,
            )
            usage = {
                'prompt_eval_count': float(data.get('prompt_eval_count', 0) or 0),
                'eval_count': float(data.get('eval_count', 0) or 0),
                'total_duration_ns': float(data.get('total_duration', 0) or 0),
            }
            content = data.get('message', {}).get('content', '')
            return ModelResponse(
                model_id=model_id,
                provider='ollama',
                content=content,
                latency_s=runtime,
                usage=usage,
                energy_kwh=energy_kwh,
                carbon_g=carbon_g,
                cost_usd=0.0,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            runtime = time.perf_counter() - started
            energy_kwh, carbon_g = estimate_energy_and_carbon(
                runtime,
                average_power_watts=settings.default_cpu_power_watts,
                memory_gb=2.0,
            )
            return ModelResponse(
                model_id=model_id,
                provider='ollama',
                content='',
                latency_s=runtime,
                usage={},
                energy_kwh=energy_kwh,
                carbon_g=carbon_g,
                cost_usd=0.0,
                error=str(exc),
            )

    async def list_models(self) -> list[str]:
        """Return the models currently served by Ollama."""
        endpoint = f'{self.base_url}/api/tags'
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
            return [item['name'] for item in payload.get('models', [])]
        except Exception:
            return []
