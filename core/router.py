"""Adaptive top-K router for Roitelet.

This router scores each candidate model with a blend of:

- benchmark-inspired capability priors,
- capability-conditioned rolling Elo adjustments,
- user preferences for raw power, ecofrugality, and independence,
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


def _attach_quality_probability(candidates: list[ModelCandidate]) -> None:
    """Annotate each candidate with a normalised quality probability in [0, 1].

    The router's raw ``final_score`` is unit-less — it's a weighted
    blend of capability score, ecofrugality bonus, and local bonus.
    For an operating-point knob to be meaningful (the
    ``quality_threshold`` filter mirrors RouteLLM's threshold), the
    score needs to live on a comparable scale across turns.

    The simplest defensible mapping that preserves rank order:
    ``p = (score - min) / (max - min)`` across the eligible
    pool on this turn. The top candidate is exactly 1.0; the worst
    is 0.0; everyone else interpolates. This isn't a *calibrated*
    probability in the RouteLLM sense — RouteLLM's threshold maps to
    "strong-wins probability" via a logistic head trained on
    preference labels — but it is *monotonic* in quality and
    *threshold-able* in the same way, which is the property the
    Pareto sweep needs. If a future iteration adds a preference-
    trained head, it can populate this field instead.

    Mutates in place; assumes ``candidates`` is already sorted by
    descending ``score``.
    """
    if not candidates:
        return
    top = candidates[0].score
    bottom = candidates[-1].score
    spread = top - bottom
    if spread <= 1e-9:
        # Degenerate: everyone tied. Every candidate is "best" — give
        # them the maximum probability so the threshold filter never
        # accidentally drops the whole pool.
        for c in candidates:
            c.quality_probability = 1.0
        return
    for c in candidates:
        c.quality_probability = max(0.0, min(1.0, (c.score - bottom) / spread))


class RoiteletRouter:
    """Rank candidate models and choose the best flight formation."""

    def route(self, prompt: str, preferences: RouterPreferences, top_k: int = 2) -> RouterDecision:
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

            ecofrugality_bonus = 1.0 / (
                1.0
                + spec.pricing['output_per_1k'] * 100.0
                + spec.energy_kwh * 1000.0
                + spec.latency_s
            )
            local_bonus = 0.15 if spec.local else 0.0
            final_score = (
                preferences.raw_power * quality_score
                + preferences.ecofrugality * ecofrugality_bonus
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
        _attach_quality_probability(candidates)

        # Calibrated quality floor — operating-point knob on the
        # cost/quality Pareto frontier. At threshold 0 every candidate
        # is eligible; at threshold 1 only the top-ranked one is. The
        # filter applies *before* top-K selection so the user's K is
        # honoured among the survivors. Mirrors RouteLLM's threshold
        # in shape (single scalar, monotonic), though derived from
        # Roitelet's rolling-Elo quality blend rather than a
        # preference-trained classifier.
        if preferences.quality_threshold > 0.0 and candidates:
            survivors = [c for c in candidates if c.quality_probability >= preferences.quality_threshold]
            if survivors:
                pruned = len(candidates) - len(survivors)
                if pruned:
                    reasoning.append(
                        f'Quality-threshold filter ({preferences.quality_threshold:.2f}): '
                        f'dropped {pruned} candidate(s) below the floor.'
                    )
                candidates = survivors
            else:
                # Don't ship an empty fan-out on a too-strict floor —
                # keep the single best candidate and surface the fact.
                reasoning.append(
                    f'Quality-threshold {preferences.quality_threshold:.2f} excluded every '
                    f'candidate; falling back to the single best.'
                )
                candidates = candidates[:1]

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
