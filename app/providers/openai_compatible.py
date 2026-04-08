"""Client for OpenAI-compatible chat completion APIs.

This client works for OpenRouter as well as any custom OpenAI-compatible
endpoint, provided the expected `/chat/completions` contract is respected.

Examples
--------
>>> # Network access is required for real usage.
>>> from app.providers.openai_compatible import OpenAICompatibleClient
>>> client = OpenAICompatibleClient(base_url='https://example.invalid', api_key='x', provider_name='demo')
>>> client.provider_name
'demo'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import time
from typing import Sequence

import httpx

from ..core.energy import estimate_energy_and_carbon
from ..schemas import ChatMessage, ModelResponse


class OpenAICompatibleClient:
    """Small async client for OpenAI-compatible chat completion endpoints."""

    def __init__(self, base_url: str, api_key: str, provider_name: str) -> None:
        """Initialize the provider client.

        Parameters
        ----------
        base_url:
            Base URL without the trailing `/chat/completions` path.
        api_key:
            Bearer token or API key for the remote endpoint.
        provider_name:
            Friendly provider identifier stored in telemetry.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.provider_name = provider_name

    async def generate(self, model_id: str, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Send a chat completion request and normalize the response."""
        started = time.perf_counter()
        endpoint = f'{self.base_url}/chat/completions'
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'model': model_id.split('/', 1)[-1],
            'messages': [message.model_dump() for message in messages],
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
            runtime = time.perf_counter() - started
            message = data['choices'][0]['message']
            usage = data.get('usage', {})
            energy_kwh, carbon_g = estimate_energy_and_carbon(runtime, average_power_watts=35.0, memory_gb=1.0)
            total_tokens = float(usage.get('total_tokens', 0) or 0)
            return ModelResponse(
                model_id=model_id,
                provider=self.provider_name,
                content=message.get('content', ''),
                latency_s=runtime,
                usage={k: float(v) for k, v in usage.items() if isinstance(v, (int, float))},
                energy_kwh=energy_kwh,
                carbon_g=carbon_g,
                cost_usd=0.0,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            runtime = time.perf_counter() - started
            energy_kwh, carbon_g = estimate_energy_and_carbon(runtime, average_power_watts=10.0, memory_gb=0.5)
            return ModelResponse(
                model_id=model_id,
                provider=self.provider_name,
                content='',
                latency_s=runtime,
                usage={},
                energy_kwh=energy_kwh,
                carbon_g=carbon_g,
                cost_usd=0.0,
                error=str(exc),
            )
