"""Learned matrix-factorisation router — a RouteLLM-style alternative.

Where the default :class:`core.router.RoiteletRouter` blends curated
priors with rolling Elo, this one blends curated priors with a **learned
quality score** trained on the project's own telemetry.

Design constraints
------------------

1. **No new heavy dependencies.** TF-IDF + truncated SVD is everything
   the prototype needs, and both ship in scikit-learn (already a
   runtime dependency).
2. **Local-first.** No OpenAI embedding API call per prompt — the
   RouteLLM trap. The router reads from
   ``data/telemetry/*.json`` already on disk, no network, no creds.
3. **Graceful degradation.** When telemetry is sparse (fewer than
   ``_MIN_TRAINING_TURNS`` recorded turns) the router silently falls
   back to the heuristic, so a fresh install behaves exactly like the
   old default.
4. **A/B-able.** Selected via ``ROITELET_ROUTER=mf`` env var; never the
   default. The runtime task list / docs make the swap explicit.
5. **No regressions on existing tests.** The heuristic router stays
   untouched; this is a new class that composes it.

What it actually does
---------------------
For each historical turn in telemetry we read:
- the prompt text,
- the ``RouterDecision`` (which models were selected),
- the ``synthesis.winning_model_ids`` (which model(s) the judge crowned).

A model-prompt quality score `q[m, p]` is approximated by treating
winners as `1` and routed-but-not-winning candidates as `0`. We
TF-IDF-vectorise the prompts, run truncated SVD to ~16 components, and
average the SVD coordinates per model to get a model "centre" in
embedding space. At route-time the user prompt is projected the same
way and scored against each model centre via cosine similarity —
yielding a learned quality estimate per candidate.

That score is then **blended** with the heuristic's quality score
(50/50 by default) before the standard ecofrugality / cost / regime layer
runs. Blending preserves the heuristic's transparency while letting
the learned signal pull the ranking when it's confident.

Why this isn't yet the default
------------------------------
We don't have a benchmark showing it beats the heuristic. The
companion runner in :mod:`tests.eval.bench_pareto` is the seam for
proving (or disproving) the win. Until that lands a measured
improvement on the dataset, this router stays opt-in.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import storage as _storage_mod
from .capabilities import detect_capabilities, top_capabilities
from .config import get_settings
from .regimes import detect_regime
from .registry import ModelRegistry
from .router import RoiteletRouter
from .schemas import ModelCandidate, ModelCapabilityScore, RouterDecision, RouterPreferences

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Below this many telemetry turns the learned signal is too noisy to
# trust; degrade to the pure heuristic. The constant is conservative on
# purpose — we'd rather act like the existing default than make wild
# predictions from a tiny corpus.
_MIN_TRAINING_TURNS: int = 32

# How many SVD components to compute. Small enough to stay cheap on a
# laptop CPU, large enough to separate broad task categories.
_SVD_COMPONENTS: int = 16

# Blend weight: how much weight to give the learned signal vs. the
# heuristic's per-capability quality score. 0.5 = equal trust. Tuning
# this is the right Phase A1.5 experiment once the eval harness ships
# real numbers.
_LEARNED_BLEND: float = 0.5


# ---------------------------------------------------------------------------
# Telemetry loading
# ---------------------------------------------------------------------------


def _telemetry_dir() -> Path:
    """Return the on-disk path holding per-turn telemetry records."""
    return get_settings().data_dir / 'telemetry'


def _load_telemetry_records() -> list[dict]:
    """Read every telemetry JSON file under ``data/telemetry/``.

    Best-effort: silently skips malformed files (a truncated write or a
    schema migration) and logs at debug level so a fresh install with
    no telemetry yet doesn't spam the log.
    """
    directory = _telemetry_dir()
    if not directory.is_dir():
        return []
    records: list[dict] = []
    for path in directory.glob('*.json'):
        try:
            records.append(json.loads(path.read_text(encoding='utf-8')))
        except Exception as exc:  # noqa: BLE001
            logger.debug('Skipping unreadable telemetry %s: %s', path.name, exc)
    return records


def _records_to_training_triples(
    records: list[dict],
) -> tuple[list[str], list[list[str]], list[list[str]]]:
    """Reshape raw telemetry into the three parallel arrays the model needs.

    Returns
    -------
    (prompts, winners_per_turn, losers_per_turn)
        Each list has length ``len(records)``. Winners are model_ids
        the synthesis judge crowned; losers are the routed candidates
        that didn't win.
    """
    prompts: list[str] = []
    winners_per_turn: list[list[str]] = []
    losers_per_turn: list[list[str]] = []
    for record in records:
        prompt = record.get('prompt') or ''
        if not prompt:
            continue
        synthesis = record.get('synthesis') or {}
        winners = list(synthesis.get('winning_model_ids') or [])
        responses = record.get('model_responses') or []
        routed = [r.get('model_id') for r in responses if r.get('model_id')]
        losers = [m for m in routed if m not in winners]
        if not routed:
            continue
        prompts.append(prompt)
        winners_per_turn.append(winners)
        losers_per_turn.append(losers)
    return prompts, winners_per_turn, losers_per_turn


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


class _LearnedQualityModel:
    """TF-IDF + truncated SVD averaged per model.

    The training surface is tiny because the input is tiny. We're not
    competing with a 7B classifier — we're competing with a 60-line
    keyword scan, and a sklearn pipeline that fits in 100 ms is the
    right size for that comparison.
    """

    def __init__(self) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            min_df=1,
            max_features=4096,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=_SVD_COMPONENTS, random_state=0)
        self._model_centres: dict[str, np.ndarray] = {}
        self._trained: bool = False

    def fit(
        self,
        prompts: list[str],
        winners_per_turn: list[list[str]],
        losers_per_turn: list[list[str]],
    ) -> None:
        """Fit the TF-IDF + SVD pipeline and average embeddings per model.

        Loser turns are folded in with weight -1 so models that *lose*
        on a topic drift away from the corresponding region of embedding
        space. This is what makes the score predictive rather than just
        descriptive of which models get *called* often.
        """
        if not prompts:
            return
        try:
            tfidf = self._vectorizer.fit_transform(prompts)
            n_components = min(_SVD_COMPONENTS, max(1, tfidf.shape[1] - 1))
            if n_components < _SVD_COMPONENTS:
                # Tiny corpus — re-instantiate the SVD with the safe
                # component count rather than letting sklearn raise.
                from sklearn.decomposition import TruncatedSVD
                self._svd = TruncatedSVD(n_components=n_components, random_state=0)
            embeddings = self._svd.fit_transform(tfidf)
        except Exception as exc:  # noqa: BLE001
            logger.debug('Learned router fit failed: %s', exc)
            return

        accum: dict[str, list[np.ndarray]] = {}
        for i, embedding in enumerate(embeddings):
            for winner in winners_per_turn[i]:
                accum.setdefault(winner, []).append(embedding)
            for loser in losers_per_turn[i]:
                # Negative-weighted sample. Subtracting the embedding
                # is equivalent to telling the average to drift the
                # opposite direction for losers on this topic.
                accum.setdefault(loser, []).append(-embedding)

        self._model_centres = {
            model_id: np.mean(np.stack(samples), axis=0)
            for model_id, samples in accum.items()
            if samples
        }
        self._trained = bool(self._model_centres)

    @property
    def trained(self) -> bool:
        """``True`` iff at least one model has a learned centre."""
        return self._trained

    def score_models(self, prompt: str, candidate_model_ids: list[str]) -> dict[str, float]:
        """Project ``prompt`` and score each candidate by cosine similarity.

        Returns a dict mapping ``model_id`` to a float in roughly
        ``[-1, 1]``. Models without a learned centre score ``0.0``
        (neutral — the heuristic decides those).
        """
        if not self._trained or not candidate_model_ids:
            return {model_id: 0.0 for model_id in candidate_model_ids}
        try:
            tfidf = self._vectorizer.transform([prompt])
            embedding = self._svd.transform(tfidf)[0]
        except Exception as exc:  # noqa: BLE001
            logger.debug('Learned router transform failed: %s', exc)
            return {model_id: 0.0 for model_id in candidate_model_ids}

        norm_p = np.linalg.norm(embedding) or 1.0
        out: dict[str, float] = {}
        for model_id in candidate_model_ids:
            centre = self._model_centres.get(model_id)
            if centre is None:
                out[model_id] = 0.0
                continue
            norm_c = np.linalg.norm(centre) or 1.0
            cosine = float(np.dot(embedding, centre) / (norm_p * norm_c))
            out[model_id] = cosine
        return out


@lru_cache(maxsize=1)
def _get_quality_model() -> _LearnedQualityModel:
    """Build or return the process-wide learned-quality model.

    Fitted lazily on first call. Re-fitting is opt-in (the router has a
    ``refresh()`` method) so a long-running process doesn't pay the
    sklearn cost on every prompt.
    """
    model = _LearnedQualityModel()
    records = _load_telemetry_records()
    if len(records) < _MIN_TRAINING_TURNS:
        logger.info(
            'Learned router: %d telemetry turns < %d minimum — heuristic fallback active.',
            len(records),
            _MIN_TRAINING_TURNS,
        )
        return model  # leaves model.trained == False
    prompts, winners, losers = _records_to_training_triples(records)
    model.fit(prompts, winners, losers)
    if model.trained:
        logger.info(
            'Learned router fitted on %d turns, %d model centres.',
            len(prompts),
            len(model._model_centres),
        )
    return model


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class LearnedMFRouter:
    """Heuristic-plus-learned router behind :class:`Router` Protocol.

    Composes :class:`RoiteletRouter` for the cost / latency / regime
    logic and adds a learned per-(prompt, model) quality term blended
    50/50 with the heuristic's per-capability quality. If the learned
    model has no data, the output is byte-identical to
    :class:`RoiteletRouter` — the bound on regression is precisely zero.
    """

    def __init__(self, blend: float = _LEARNED_BLEND) -> None:
        self._heuristic = RoiteletRouter()
        self._blend = float(min(max(blend, 0.0), 1.0))

    def refresh(self) -> None:
        """Re-read telemetry and re-fit. Call after a wave of new turns."""
        _get_quality_model.cache_clear()
        _get_quality_model()

    def route(
        self,
        prompt: str,
        preferences: RouterPreferences,
        top_k: int = 2,
    ) -> RouterDecision:
        """Pick top-K with the learned-quality term blended into scoring."""
        model = _get_quality_model()
        if not model.trained:
            # Sparse-telemetry safety: act exactly as the heuristic
            # would. This is the contract — no surprise behaviour on a
            # fresh install.
            decision = self._heuristic.route(prompt, preferences, top_k=top_k)
            decision.reasoning.append('LearnedMFRouter: insufficient telemetry — heuristic fallback.')
            return decision

        # Re-implement the heuristic core here so we can fold the
        # learned term into per-candidate scoring instead of post-sorting.
        app_settings = _storage_mod.get_storage().load_app_settings()
        live_registry = ModelRegistry(app_settings=app_settings)
        categories = detect_capabilities(prompt)
        regime = detect_regime(prompt, preferences, categories)

        candidate_ids = [spec.model_id for spec in live_registry.list_models()]
        learned_scores = model.score_models(prompt, candidate_ids)

        candidates: list[ModelCandidate] = []
        reasoning: list[str] = []
        reasoning.append(f'Detected capabilities: {", ".join(top_capabilities(categories))}')
        reasoning.append(f'Regime: {regime.name} — {regime.rationale}')
        reasoning.append(
            f'LearnedMFRouter active (blend={self._blend:.2f}, '
            f'{len(model._model_centres)} model centres).'
        )

        for spec in live_registry.list_models():
            if preferences.independence and not spec.local:
                continue
            if not preferences.allow_vlms and spec.vlm and 'vision' not in categories:
                continue
            estimated_cost = spec.pricing['input_per_1k'] + spec.pricing['output_per_1k']
            if regime.cost_budget_usd is not None and estimated_cost > regime.cost_budget_usd:
                continue

            capability_scores: list[ModelCapabilityScore] = []
            heuristic_quality = 0.0
            for capability, weight in categories.items():
                score = live_registry.capability_score(spec.model_id, capability)
                heuristic_quality += weight * score
                capability_scores.append(
                    ModelCapabilityScore(
                        capability=capability,
                        score=score,
                        rationale=f'Bootstrap prior + rolling Elo for {capability}.',
                    )
                )

            learned = learned_scores.get(spec.model_id, 0.0)
            # Cosine ranges [-1, 1]; rescale to [0, 1.5] to align with
            # ``capability_score``'s clamped range so the blend doesn't
            # silently push the heuristic out of its operating range.
            learned_rescaled = (learned + 1.0) * 0.75
            quality_score = (
                (1.0 - self._blend) * heuristic_quality
                + self._blend * learned_rescaled
            )

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
        # Reuse the heuristic router's quality-probability normaliser so
        # both router implementations expose the same operating-point
        # knob. Threshold filtering is identical.
        from .router import _attach_quality_probability
        _attach_quality_probability(candidates)
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
                reasoning.append(
                    f'Quality-threshold {preferences.quality_threshold:.2f} excluded every '
                    f'candidate; falling back to the single best.'
                )
                candidates = candidates[:1]
        selected_model_ids = [c.model_id for c in candidates[: max(1, top_k)]]
        for candidate in candidates:
            candidate.selected = candidate.model_id in selected_model_ids
        reasoning.append(
            f'Selected top-{len(selected_model_ids)} via heuristic+learned blend.'
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


def get_router_from_env():
    """Pick the router implementation according to ``ROITELET_ROUTER``.

    Three flavours, picked at process start:

    * unset / ``heuristic`` (default) — :class:`core.router.RoiteletRouter`.
      Curated priors + rolling Elo + regimes. Works on a fresh install
      with zero telemetry. The naive baseline; documented behaviour.
    * ``mf`` — :class:`LearnedMFRouter`. TF-IDF + SVD over telemetry to
      blend a learned quality score with the heuristic. Degrades to
      the heuristic when telemetry is sparse.
    * ``calibrated`` — :class:`core.router_calibrated.CalibratedCostRouter`.
      RouteLLM-shaped: a calibrated logistic regression predicts
      ``P(strong wins | prompt)`` for the per-turn head/tail pair; the
      :attr:`RouterPreferences.quality_threshold` becomes the
      Pareto-frontier knob. Also degrades to the heuristic when
      telemetry is sparse.

    The pipeline never calls this directly; the public seam is the
    ``router`` argument on :func:`core.pipeline.run_roitelet_chat`.
    Use this factory from the API startup hook when wiring an
    alternative router in.
    """
    flavour = os.environ.get('ROITELET_ROUTER', 'heuristic').lower().strip()
    if flavour == 'mf':
        return LearnedMFRouter()
    if flavour == 'calibrated':
        from .router_calibrated import CalibratedCostRouter
        return CalibratedCostRouter()
    return RoiteletRouter()
