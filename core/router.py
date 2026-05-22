"""Adaptive top-K router for Roitelet LLM.

This router scores each candidate model with a blend of:
- benchmark-inspired capability priors,
- capability-conditioned rolling Elo adjustments,
- user preferences for raw power, frugality, and independence,
- prompt modality constraints such as VLM authorization.

Examples
--------
>>> from core.core.router import RoiteletRouter
>>> from core.schemas import RouterPreferences
>>> router = RoiteletRouter()
>>> decision = router.route("Write a Python function", RouterPreferences())
>>> len(decision.selected_model_ids) >= 1
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from typing import Dict, List

from ..schemas import ModelCandidate, ModelCapabilityScore, RouterDecision, RouterPreferences
from ..storage import storage
from .capabilities import detect_capabilities, top_capabilities
from .registry import ModelRegistry


class RoiteletRouter:
    """Rank candidate models and choose the best flight formation."""

    def route(self, prompt: str, preferences: RouterPreferences, top_k: int = 3) -> RouterDecision:
        """Route one prompt to a small set of models.

        Parameters
        ----------
        prompt:
            User prompt or question.
        preferences:
            Runtime preferences edited from the UI.
        top_k:
            Number of models to select.

        Returns
        -------
        RouterDecision
            Full routing decision and per-candidate scores.
        """
        # Build a fresh registry snapshot that merges user-configured models.
        # This ensures Ollama / OpenRouter models added in the control room are
        # immediately visible to the router without requiring a restart.
        app_settings = storage.load_app_settings()
        live_registry = ModelRegistry(app_settings=app_settings)

        categories = detect_capabilities(prompt)
        candidates: List[ModelCandidate] = []
        reasoning: List[str] = []
        dominant = top_capabilities(categories)
        reasoning.append(f'Detected capabilities: {", ".join(dominant)}')

        for spec in live_registry.list_models():
            if preferences.independence and not spec.local:
                continue
            if not preferences.allow_vlms and spec.vlm and 'vision' not in categories:
                continue
            capability_scores: List[ModelCapabilityScore] = []
            quality_score = 0.0
            for capability, weight in categories.items():
                score = live_registry.capability_score(spec.model_id, capability)
                quality_score += weight * score
                capability_scores.append(
                    ModelCapabilityScore(
                        capability=capability,
                        score=score,
                        rationale=f'Bootstrap prior + rolling Elo for {capability}.',
                    )
                )
            frugality_bonus = 1.0 / (
                1.0
                + spec.pricing['output_per_1k'] * 100.0
                + spec.energy_kwh * 1000.0
                + spec.latency_s
            )
            local_bonus = 0.15 if spec.local else 0.0
            final_score = (
                preferences.raw_power * quality_score
                + preferences.frugality * frugality_bonus
                + (1.0 if preferences.independence else 0.0) * local_bonus
            )
            candidates.append(
                ModelCandidate(
                    model_id=spec.model_id,
                    provider=spec.provider,
                    score=final_score,
                    estimated_cost_usd=spec.pricing['input_per_1k'] + spec.pricing['output_per_1k'],
                    estimated_latency_s=spec.latency_s,
                    capability_scores=capability_scores,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        selected_model_ids = [candidate.model_id for candidate in candidates[:max(1, top_k)]]
        for candidate in candidates:
            candidate.selected = candidate.model_id in selected_model_ids
        reasoning.append(f'Selected top-{len(selected_model_ids)} based on weighted quality/cost/energy tradeoff.')
        if preferences.independence:
            reasoning.append('Local-only independence mode filtered remote models out.')
        return RouterDecision(
            prompt=prompt,
            categories=categories,
            candidates=candidates,
            selected_model_ids=selected_model_ids,
            reasoning=reasoning,
        )
