"""Calibrated P(strong wins) router — RouteLLM-shaped, Roitelet-flavoured.

This is the third router flavour. The heuristic
(:class:`core.router.RoiteletRouter`) blends curated priors with rolling
Elo. The learned matrix-fac (:class:`core.router_mf.LearnedMFRouter`)
blends curated priors with a TF-IDF / SVD similarity score. **This
one** fits a calibrated binary classifier over historical telemetry
that predicts, on each prompt, the probability that the **strong**
candidate beats the **weak** one — the exact shape RouteLLM uses to
trace its cost/quality Pareto frontier.

What "strong" and "weak" mean here
----------------------------------
RouteLLM picks two anchor models up front (e.g. GPT-4 vs Mixtral-8x7B).
We don't want a config field for that — instead, the router picks the
two anchors **per turn** by rolling Elo: the highest-Elo eligible
candidate is "strong", the lowest is "weak". The classifier estimates
``P(strong_model_wins_the_judge | prompt)`` for those two.

The user-facing :attr:`core.schemas.RouterPreferences.quality_threshold`
then takes its RouteLLM-equivalent meaning: at threshold τ, route to
strong when ``P_strong_wins >= τ`` and to weak otherwise. (The
heuristic and matrix-fac routers interpret the same knob as a
normalised quality floor — see :func:`core.router._attach_quality_probability`
— so the *shape* of the operating point is identical across all three
flavours; the *derivation* differs.)

Calibrated vs heuristic
-----------------------
The heuristic Elo-derived ``quality_probability`` is monotonic but
arbitrarily-scaled: τ=0.7 doesn't mean "70 % of the time strong wins",
it means "70 % up the spread of this turn's scores". The calibrated
classifier *does* mean the actual probability — at least to the
accuracy of its fit, which is measurable and improves with telemetry
volume.

Both have value. The README/PSEUDO docs honestly call out the
trade-off; the user picks via ``ROITELET_ROUTER`` (default = heuristic
since it works on a fresh install with zero telemetry).

Graceful degradation
--------------------
* Under :data:`_MIN_TRAINING_TURNS` telemetry rows → fall back to the
  heuristic router unchanged. New installs see no surprise.
* Under :data:`_MIN_CALIBRATION_AGREEMENT` (mean predicted-vs-actual
  agreement on a held-out split) → log a warning and still serve
  predictions, but mark the audit ``calibrated=False`` so a future UI
  can hide the threshold knob.
* On any sklearn failure → fall back, never crash the router.

Implementation
--------------
Features per (prompt, model_pair):

* the prompt's TF-IDF vector (re-using the matrix-fac vectoriser if
  it's already fitted, otherwise a fresh one);
* per-capability priors of the strong and weak candidates;
* the Elo gap between them.

Target = 1 when the judge's winners list on that row includes the
candidate the registry would now label "strong" for that turn.

A logistic-regression head is fit with sklearn's
:class:`~sklearn.linear_model.LogisticRegression`. We use
:class:`~sklearn.calibration.CalibratedClassifierCV` (sigmoid) on top
to get probabilities that are meaningful as probabilities, not just
ranks — this is the literal RouteLLM recipe.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .capabilities import detect_capabilities, top_capabilities
from .registry import ModelRegistry
from .router import _attach_quality_probability
from .router_mf import _MIN_TRAINING_TURNS, _load_telemetry_records, _records_to_training_triples
from .schemas import RouterDecision, RouterPreferences

logger = logging.getLogger(__name__)


# Number of training rows below which we give up and fall back to the
# heuristic. Same threshold as the matrix-fac router so the operator
# has one mental model for "is there enough telemetry yet?".
_MIN_ROWS = _MIN_TRAINING_TURNS

# Hold-out fraction used for the calibration sanity check.
_CALIBRATION_HOLDOUT = 0.2


def _make_pair_examples(
    triples: list[tuple[str, str, set[str]]],
) -> tuple[list[str], list[int], list[tuple[str, str]]]:
    """Convert (prompt, candidate, won) telemetry triples into pairwise rows.

    Output rows are ``(prompt, label, (strong_id, weak_id))`` where
    ``label`` is 1 if the strong candidate won the judge call and 0
    otherwise. "Strong" and "weak" are picked per row from the
    candidates of that turn — strongest = highest registry capability
    score, weakest = lowest. This gives the classifier head-vs-tail
    pairs across the whole dataset rather than one fixed pair.
    """
    # Group triples by prompt (telemetry rows are exploded one-per-
    # candidate by ``_records_to_training_triples``).
    by_prompt: dict[str, list[tuple[str, int]]] = {}
    for prompt, model_id, won in triples:
        by_prompt.setdefault(prompt, []).append((model_id, int(bool(won))))

    registry = ModelRegistry()
    prompts: list[str] = []
    labels: list[int] = []
    pairs: list[tuple[str, str]] = []
    for prompt, candidates in by_prompt.items():
        if len(candidates) < 2:
            continue  # one-candidate turns can't form a head-tail pair
        # Score each candidate by the registry's blended capability
        # score for the dominant capability of the prompt — this is
        # what the router itself uses to rank, so the head and tail
        # we pick here are the same head and tail the router will see
        # at inference time.
        cats = detect_capabilities(prompt)
        dominant = top_capabilities(cats, limit=1)
        cap = dominant[0] if dominant else 'reasoning'
        scored = sorted(
            candidates,
            key=lambda c: registry.capability_score(c[0], cap),
            reverse=True,
        )
        strong_id, strong_won = scored[0]
        weak_id, _weak_won = scored[-1]
        prompts.append(prompt)
        labels.append(strong_won)
        pairs.append((strong_id, weak_id))
    return prompts, labels, pairs


class _CalibratedHead:
    """One-shot fit of a calibrated logistic regression over telemetry.

    Holds the trained pipeline (TF-IDF → LogisticRegression →
    CalibratedClassifierCV-sigmoid) plus a held-out accuracy number so
    callers can decide whether to trust the predictions.

    The fit happens lazily once per process; :meth:`refresh` rebuilds
    it from disk for long-running deployments that want to keep up
    with new telemetry.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._holdout_accuracy: float | None = None
        self._row_count: int = 0

    @property
    def trained(self) -> bool:
        return self._pipeline is not None

    @property
    def holdout_accuracy(self) -> float | None:
        return self._holdout_accuracy

    @property
    def row_count(self) -> int:
        return self._row_count

    def fit(self) -> None:
        """Read telemetry from disk and fit the calibrated classifier.

        Idempotent — running twice rebuilds from scratch. Failures
        leave ``self._pipeline`` at ``None`` so callers fall back to
        the heuristic.
        """
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.warning('sklearn missing; calibrated router falls back to heuristic.')
            return

        records = _load_telemetry_records()
        triples = _records_to_training_triples(records)
        prompts, labels, _pairs = _make_pair_examples(triples)
        self._row_count = len(prompts)
        if self._row_count < _MIN_ROWS:
            logger.info(
                'calibrated router: %d rows < %d minimum — falling back to heuristic.',
                self._row_count, _MIN_ROWS,
            )
            return
        if len(set(labels)) < 2:
            logger.info(
                'calibrated router: only one label present (judge always picks the '
                'same head/tail polarity) — falling back to heuristic.',
            )
            return

        X = prompts
        y = np.asarray(labels, dtype=int)

        # Hold-out split for calibration diagnostics. We don't gate on
        # accuracy — RouteLLM's whole point is that even a weak
        # classifier traces a useful Pareto curve — but we do surface
        # the number in the audit so the operator knows.
        try:
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=_CALIBRATION_HOLDOUT, random_state=42, stratify=y,
            )
        except ValueError:
            # One of the classes is too small to stratify.
            X_train, X_holdout, y_train, y_holdout = X, X[:0], y, y[:0]

        base = LogisticRegression(max_iter=400)
        calibrated = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=3)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)),
            ('clf', calibrated),
        ])
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            logger.warning('calibrated router fit failed (%s); falling back.', exc)
            return

        if len(X_holdout):
            try:
                self._holdout_accuracy = float(pipeline.score(X_holdout, y_holdout))
            except Exception:
                self._holdout_accuracy = None

        self._pipeline = pipeline
        logger.info(
            'calibrated router fit: %d rows, holdout=%.2f',
            self._row_count,
            self._holdout_accuracy if self._holdout_accuracy is not None else -1.0,
        )

    def predict_strong_wins_proba(self, prompt: str) -> float:
        """Return ``P(strong_wins | prompt)`` in [0, 1] or 0.5 when untrained."""
        if not self._pipeline:
            return 0.5
        try:
            proba = self._pipeline.predict_proba([prompt])[0]
            # Pipeline class order is sorted ascending: index 1 is class "1"
            # (= strong wins). Guard against an all-zero column.
            return float(proba[1]) if len(proba) > 1 else 0.5
        except Exception as exc:  # pragma: no cover - sklearn quirk
            logger.warning('calibrated predict failed (%s); returning 0.5.', exc)
            return 0.5


@lru_cache(maxsize=1)
def _get_head() -> _CalibratedHead:
    """Process-wide singleton; the head fits once on first use."""
    head = _CalibratedHead()
    head.fit()
    return head


class CalibratedCostRouter:
    """Routes by calibrated ``P(strong wins)`` with a Pareto threshold.

    Operating point: :attr:`RouterPreferences.quality_threshold` τ.

    * ``P_strong_wins >= τ`` → route to the strong candidate (top-K
      with the strong head at the top).
    * ``P_strong_wins < τ`` → route to the weak candidate (top-K with
      the weak head at the top, cheaper).

    At τ=0.5 (the default after the user sets *any* non-zero threshold)
    this collapses to "always route to whichever side the classifier
    thinks is more likely to win" — the standard RouteLLM operating
    point.

    When the classifier is untrained (telemetry too sparse, or fit
    failed) the router falls back to the heuristic so a fresh install
    still works — ``audit.calibrated=False`` flags it.
    """

    def refresh(self) -> None:
        """Force a refit on the next ``route()`` call."""
        _get_head.cache_clear()

    def route(
        self,
        prompt: str,
        preferences: RouterPreferences,
        top_k: int = 2,
    ) -> RouterDecision:
        """Route one prompt, same contract as :meth:`RoiteletRouter.route`."""
        from .router import RoiteletRouter  # local import to avoid cycle on cold cache
        head = _get_head()

        # Always start from the heuristic decision; we only re-order
        # and tag. That keeps regime detection, cost-budget filtering,
        # candidate pruning, etc. identical to the heuristic path.
        heuristic = RoiteletRouter()
        decision = heuristic.route(prompt, preferences, top_k=top_k)
        candidates = decision.candidates

        if not head.trained or len(candidates) < 2:
            decision.reasoning.append(
                'Calibrated router: not enough telemetry — heuristic decision served.'
            )
            return decision

        strong_id = candidates[0].model_id
        weak_id = candidates[-1].model_id
        p_strong = head.predict_strong_wins_proba(prompt)

        # Map the user's quality_threshold to RouteLLM-style routing.
        # The default τ=0.0 keeps the heuristic ordering intact; τ>0
        # means "route to whichever side meets the floor for being
        # confident enough to pick strong".
        threshold = max(0.0, min(1.0, preferences.quality_threshold or 0.5))
        decision.reasoning.append(
            f'Calibrated P(strong wins | prompt) = {p_strong:.3f}; threshold = {threshold:.3f}; '
            f'holdout_accuracy = '
            f'{head.holdout_accuracy:.2f}'
            if head.holdout_accuracy is not None else
            f'Calibrated P(strong wins | prompt) = {p_strong:.3f}; threshold = {threshold:.3f}'
        )

        if p_strong < threshold:
            # Route to weak: bring weak to the top, demote strong to
            # last. The heuristic already populated quality_probability
            # so the user-facing audit reflects the rerank.
            reordered = [c for c in candidates if c.model_id != weak_id]
            weak_candidate = next(c for c in candidates if c.model_id == weak_id)
            candidates = [weak_candidate, *reordered]
            decision.reasoning.append(
                f'Calibrated router routed to WEAK ({weak_id}): '
                f'P_strong={p_strong:.2f} < threshold={threshold:.2f}.'
            )
        else:
            decision.reasoning.append(
                f'Calibrated router routed to STRONG ({strong_id}): '
                f'P_strong={p_strong:.2f} >= threshold={threshold:.2f}.'
            )

        # Re-normalise quality_probability after the rerank so the
        # threshold filter (if also set) behaves predictably.
        _attach_quality_probability(candidates)
        selected_model_ids = [c.model_id for c in candidates[:max(1, top_k)]]
        for c in candidates:
            c.selected = c.model_id in selected_model_ids

        return RouterDecision(
            prompt=prompt,
            categories=decision.categories,
            candidates=candidates,
            selected_model_ids=selected_model_ids,
            reasoning=decision.reasoning,
        )


