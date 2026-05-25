"""Regime detection for hybrid routing math.

Different (prompt, preferences) combinations call for different routing
math. A single linear blend of quality + frugality is the right answer
*most* of the time, but it's the wrong answer in well-defined regimes:

* a trivial 5-word factual question doesn't need top-3 fan-out + fusion
  (one local model is enough — anything else is wasted latency),
* a request with a tight ``max_cost_usd`` budget shouldn't sort by
  quality before filtering by cost,
* a long-context prompt should never route to a model whose
  ``long_context`` prior is weak, even if its other priors are strong,
* an independence-mode prompt shouldn't waste cycles scoring remote
  candidates that will get filtered anyway,
* an ambiguous prompt (no keyword fired, no capability dominates) is
  better served by a globally-strong generalist than by a specialist
  whose narrow score won.

This module factors that regime detection out of the router into a pure
function. The router consults it to decide *which* scoring math to use
for this turn; the regime label is surfaced in :class:`RouterDecision`
``reasoning`` so the choice is auditable in telemetry.

Examples
--------
>>> from core.schemas import RouterPreferences
>>> regime = detect_regime('hi', RouterPreferences(), {'reasoning': 1.0})
>>> regime.name
'trivial'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .schemas import RouterPreferences


@dataclass(frozen=True, slots=True)
class Regime:
    """The detected regime for one routing call.

    Attributes
    ----------
    name:
        Short identifier (``trivial``, ``budget_constrained``,
        ``long_context``, ``capability_dominant``, ``ambiguous``,
        ``default``).
    rationale:
        One-sentence explanation for the audit log. Always non-empty.
    suggested_top_k:
        The number of candidates that makes sense for this regime.
        ``None`` means "use the caller's requested value." The router
        applies this as ``min(requested_top_k, suggested_top_k)`` so a
        user request for K=3 never gets silently inflated to K=5 by a
        regime hint.
    cost_budget_usd:
        The active per-turn budget if ``max_cost_usd`` is set on
        preferences. Lifted onto the regime for convenience so the
        router doesn't need to re-check preferences.
    """

    name: str
    rationale: str
    suggested_top_k: Optional[int] = None
    cost_budget_usd: Optional[float] = None


# Heuristic constants.  Kept conservative — these gate behavioural
# changes, so getting them wrong shifts what the router does. Bias toward
# *not* triggering a regime when uncertain; the default path is fine.
_TRIVIAL_CHAR_LIMIT = 80           # 5–12 words of text in most languages
_TRIVIAL_TOKEN_HEURISTIC = 16      # rough word count
_LONG_CONTEXT_CHAR_LIMIT = 4000    # ~700 tokens — same threshold the
                                   # capability detector uses for the
                                   # ``long_context`` weight
_CAP_DOMINANCE_THRESHOLD = 0.55    # one capability holds > 55 % of the
                                   # normalised weight
_AMBIGUOUS_TOP_WEIGHT = 0.30       # no capability above 30 % suggests
                                   # the keyword detector found nothing


def detect_regime(
    prompt: str,
    preferences: RouterPreferences,
    capabilities: Dict[str, float],
) -> Regime:
    """Classify the current routing call into one of six regimes.

    Order of checks is **deliberately exclusive** — once a check fires,
    the regime is set. Later checks don't override it. This makes the
    behaviour easy to reason about in telemetry (the chosen regime is
    the *first* one whose precondition matched).

    Parameters
    ----------
    prompt:
        The raw user prompt. Length and content matter; the function
        does no tokenisation beyond ``len()`` and a whitespace split.
    preferences:
        The user's :class:`RouterPreferences`. The budget knob and the
        independence flag both factor in.
    capabilities:
        The normalised capability distribution from
        :func:`core.capabilities.detect_capabilities`. Used to spot
        dominance and ambiguity.

    Returns
    -------
    Regime
        A populated regime record. ``rationale`` is always set;
        ``suggested_top_k`` and ``cost_budget_usd`` are populated only
        for regimes that need them.
    """
    budget = preferences.max_cost_usd

    # 1. Budget regime trumps everything else: if the user said "spend
    #    at most $X", we filter by cost first and then sort. A trivial
    #    prompt with a budget is still a budget-regime turn — the cost
    #    behaviour is what the user asked for.
    if budget is not None and budget >= 0:
        return Regime(
            name='budget_constrained',
            rationale=f'max_cost_usd={budget:g} — filter candidates by cost before scoring.',
            cost_budget_usd=budget,
        )

    word_count = len(prompt.split())
    char_count = len(prompt)

    # 2. Trivial prompts: short, no domain markers, no special demands.
    #    K=1 is enough; fan-out + fusion is wasted latency. We *only*
    #    auto-reduce K when the prompt is unambiguously trivial — long
    #    or domain-specific prompts always get the full fan-out.
    if (
        char_count <= _TRIVIAL_CHAR_LIMIT
        and word_count <= _TRIVIAL_TOKEN_HEURISTIC
        and not preferences.allow_vlms
    ):
        return Regime(
            name='trivial',
            rationale=(
                f'short prompt ({char_count} chars, {word_count} words) — '
                'one cheap candidate is enough.'
            ),
            suggested_top_k=1,
        )

    # 3. Long-context regime: the prompt itself is the signal. Same
    #    threshold as ``detect_capabilities``'s long_context bump so the
    #    two heuristics agree.
    if char_count > _LONG_CONTEXT_CHAR_LIMIT:
        return Regime(
            name='long_context',
            rationale=(
                f'prompt is {char_count} chars long — '
                'prefer candidates with strong long_context priors.'
            ),
        )

    # 4. Dominant-capability regime: one capability owns most of the
    #    weight. The per-capability score will drive selection cleanly;
    #    no special-casing needed beyond surfacing the label.
    if capabilities:
        top_weight = max(capabilities.values())
        if top_weight >= _CAP_DOMINANCE_THRESHOLD:
            top_capability = max(capabilities.items(), key=lambda kv: kv[1])[0]
            return Regime(
                name='capability_dominant',
                rationale=(
                    f'capability "{top_capability}" carries {top_weight:.0%} '
                    'of the prompt weight — specialist routing.'
                ),
            )

        # 5. Ambiguous regime: no capability rises meaningfully above
        #    the noise. The keyword detector found nothing useful; bias
        #    toward globally-strong generalists by surfacing the label
        #    (the router weights this in by leaning on the ``global``
        #    Elo term rather than capability-specific ones).
        if top_weight < _AMBIGUOUS_TOP_WEIGHT:
            return Regime(
                name='ambiguous',
                rationale=(
                    f'no capability exceeds {_AMBIGUOUS_TOP_WEIGHT:.0%} of '
                    'the prompt weight — prefer generalists.'
                ),
            )

    # 6. Default: nothing special detected. The standard linear blend
    #    (quality + frugality + independence bonus) is the right math.
    return Regime(
        name='default',
        rationale='no special regime detected — standard linear blend applies.',
    )
