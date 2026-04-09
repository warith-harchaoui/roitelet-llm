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
        """Generate one completion from an upstream model.

        Parameters
        ----------
        model_id : str
            The desired target model ID defined dynamically or statically.
        messages : Sequence[ChatMessage]
            The conversational messages up to this point.

        Returns
        -------
        ModelResponse
            The generated string payload and token cost calculations.
        """
        ...
