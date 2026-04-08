"""Base protocol for model providers.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from ..schemas import ChatMessage, ModelResponse


class ProviderClient(Protocol):
    """Protocol implemented by all upstream providers."""

    async def generate(self, model_id: str, messages: Sequence[ChatMessage]) -> ModelResponse:
        """Generate one completion from an upstream model."""
        ...
