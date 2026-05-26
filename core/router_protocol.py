"""Router Protocol.

A typing seam for "how does Roitelet pick which K models to query?"

Today there's exactly one implementation
(:class:`core.router.RoiteletRouter`, capability-conditioned rolling Elo
scoring). MECHANISM.md already gestures at future alternatives — a
prompt → top-K classifier, a contextual bandit, a learned policy trained
on accumulated telemetry. This Protocol pins the contract so swapping in
any of those doesn't need to touch the pipeline.

The pipeline's ``run_roitelet_chat`` accepts an optional ``router``
argument so test suites and alternative implementations can inject one
without monkey-patching globals; the default remains the singleton
returned by :func:`core.pipeline.get_router`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import RouterDecision, RouterPreferences


@runtime_checkable
class Router(Protocol):
    """Public surface every Roitelet router must implement.

    The single ``route`` method is the entire contract — receives the
    user prompt + caller preferences + a desired ``top_k``, returns a
    :class:`RouterDecision` carrying the selected model ids, the full
    candidate list with scores, and detected capability weights.

    Implementations must be **pure with respect to side effects**: the
    method should never write to disk or talk to a network. The pipeline
    relies on routing being cheap enough to run on every request.
    """

    def route(
        self,
        prompt: str,
        preferences: RouterPreferences,
        top_k: int = 3,
    ) -> RouterDecision:
        """Return a :class:`RouterDecision` for one ``(prompt, preferences)`` pair."""
        ...
