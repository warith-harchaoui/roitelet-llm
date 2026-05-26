"""Adaptive top-K router for Roitelet LLM.

This router scores each candidate model with a blend of:

- benchmark-inspired capability priors,
- capability-conditioned rolling Elo adjustments,
- user preferences for raw power, frugality, and independence,
- prompt modality constraints such as VLM authorization,
- regime-aware adjustments (cost-budget filtering, generalist bias on
  ambiguous prompts, top-K reduction on trivial prompts).

The regime layer makes the routing math **hybrid**: the linear blend
runs by default, but for well-defined regimes (budget-constrained,
trivial, ambiguous) the math composes with regime-specific filters or
biases. See :mod:`core.regimes` for the regime taxonomy and the audit
trail surfaced in :class:`RouterDecision.reasoning`.

Examples
--------
>>> from core.router import RoiteletRouter
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

from . import storage as _storage_mod
from .capabilities import top_capabilities
from .regimes import detect_regime
from .registry import ModelRegistry
from .schemas import ModelCandidate, ModelCapabilityScore, RouterDecision, RouterPreferences


def _detect(prompt: str) -> dict[str, float]:
    """Pick the active capability detector — keyword (default) or embedding.

    Late import on the embedding classifier so the dependency on sklearn
    + httpx for embedding stays inside the opt-in path. The default
    behaviour (keyword detector) is unchanged.
    """
    import os
    if os.environ.get('ROITELET_CAPABILITY_DETECTOR', 'keyword').lower().strip() == 'embedding':
        from .capability_classifier import detect_capabilities_active
        return detect_capabilities_active(prompt)
    from .capabilities import detect_capabilities
    return detect_capabilities(prompt)


# How much weight to put on the global Elo term when the prompt is
# ambiguous (no capability dominates). Bumped from the standard 0.5 in
# ``registry.capability_score`` so that generalists outscore narrow
# specialists when the keyword detector found nothing useful.
_AMBIGUOUS_GLOBAL_BOOST = 0.25


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
            Number of models to select. A regime hint may *reduce*
            this (e.g. trivial prompts collapse to K=1) but never
            inflate it past the caller's requested value.

        Returns
        -------
        RouterDecision
            Full routing decision and per-candidate scores.
        """
        # Build a fresh registry snapshot that merges user-configured models.
        # This ensures Ollama / OpenRouter models added in the control room are
        # immediately visible to the router without requiring a restart.
        app_settings = _storage_mod.get_storage().load_app_settings()
        live_registry = ModelRegistry(app_settings=app_settings)

        categories = _detect(prompt)
        regime = detect_regime(prompt, preferences, categories)

        candidates: list[ModelCandidate] = []
        reasoning: list[str] = []
        dominant = top_capabilities(categories)
        reasoning.append(f'Detected capabilities: {", ".join(dominant)}')
        reasoning.append(f'Regime: {regime.name} — {regime.rationale}')

        for spec in live_registry.list_models():
            if preferences.independence and not spec.local:
                continue
            if not preferences.allow_vlms and spec.vlm and 'vision' not in categories:
                continue

            estimated_cost = spec.pricing['input_per_1k'] + spec.pricing['output_per_1k']
            # Budget regime — drop candidates that violate the cost ceiling
            # *before* scoring. This is the only regime that filters the
            # candidate pool; other regimes only re-weight or post-trim.
            if regime.cost_budget_usd is not None and estimated_cost > regime.cost_budget_usd:
                continue

            capability_scores: list[ModelCapabilityScore] = []
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

            # Ambiguous regime: the keyword detector found no clear
            # signal. Lean on the global Elo term as a generalist proxy
            # so a model with strong overall standing outscores a model
            # whose narrow specialty happens to match a noisy keyword.
            if regime.name == 'ambiguous':
                global_elo = live_registry.elo_state.get(spec.model_id, {}).get('global', 0.0)
                quality_score += _AMBIGUOUS_GLOBAL_BOOST * global_elo

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
                    estimated_cost_usd=estimated_cost,
                    estimated_latency_s=spec.latency_s,
                    capability_scores=capability_scores,
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)

        # Regime ``suggested_top_k`` is **advisory only**. The pipeline's
        # contract is "respect the caller's ``top_k``", and the
        # telemetry log records both the requested K and the suggestion
        # so a future change can opt into honouring the hint. Today the
        # regime label exists to make the *reason* the router chose
        # what it did legible; it does not silently change fan-out.
        selected_model_ids = [candidate.model_id for candidate in candidates[:max(1, top_k)]]
        if regime.suggested_top_k is not None and regime.suggested_top_k < top_k:
            reasoning.append(
                f'Regime suggests top_k={regime.suggested_top_k} (advisory; honouring requested K={top_k}).'
            )
        for candidate in candidates:
            candidate.selected = candidate.model_id in selected_model_ids
        reasoning.append(
            f'Selected top-{len(selected_model_ids)} based on weighted quality/cost/energy tradeoff.'
        )
        if preferences.independence:
            reasoning.append('Local-only independence mode filtered remote models out.')
        if regime.cost_budget_usd is not None:
            reasoning.append(
                f'Cost-budget regime: dropped candidates above ${regime.cost_budget_usd:g}/1k tokens.'
            )
        return RouterDecision(
            prompt=prompt,
            categories=categories,
            candidates=candidates,
            selected_model_ids=selected_model_ids,
            reasoning=reasoning,
        )
