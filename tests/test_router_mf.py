"""Tests for the learned matrix-fac router and the ``ROITELET_ROUTER`` selector.

Three story-level tests:

1. **Sparse telemetry → heuristic fallback.** A fresh install with
   no historical winners is the documented degraded path.
2. **Enough telemetry → the learned signal promotes the consistent
   winner and still respects the cost-budget regime.** This is the
   *interesting* combination: a learned blend that wants the strong
   model and a budget that forbids paid candidates must compose.
3. **Env selector picks the right flavour.** Three flavours are
   advertised in the docs (heuristic / mf / calibrated); the
   selector test pins all three at once.

The synthetic telemetry crowns ``ollama/qwen2.5:14b-instruct`` on every
coding prompt so the learned router is fit on a clear, learnable signal.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _seed_tmp(tmp_path: Path, with_telemetry: int = 0) -> None:
    """Bootstrap a temp data dir with priors and optional synthetic telemetry."""
    src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
    (tmp_path / 'bootstrap').mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'model_priors.json', tmp_path / 'bootstrap' / 'model_priors.json')
    (tmp_path / 'telemetry').mkdir(parents=True, exist_ok=True)

    chosen_winner = 'ollama/qwen2.5:14b-instruct'
    losers = ['ollama/llama3.2:3b', 'openrouter/openai/gpt-4.1']
    for i in range(with_telemetry):
        record = {
            'record_id': str(uuid.uuid4()),
            'created_at': datetime.now(UTC).isoformat(),
            'conversation_id': str(uuid.uuid4()),
            'prompt': f'Write a Python function fizzbuzz_{i}(n) that prints fizz buzz.',
            'router_decision': {
                'prompt': 'placeholder',
                'categories': {'coding': 1.0},
                'candidates': [],
                'selected_model_ids': [chosen_winner, *losers],
                'reasoning': [],
            },
            'model_responses': [
                {'model_id': chosen_winner, 'provider': 'ollama', 'content': 'ok',
                 'latency_s': 1.0, 'usage': {}},
                *[
                    {'model_id': mid, 'provider': 'openrouter', 'content': 'ok',
                     'latency_s': 1.0, 'usage': {}}
                    for mid in losers
                ],
            ],
            'synthesis': {
                'model_id': chosen_winner, 'provider': 'ollama',
                'content': 'placeholder', 'judge_summary': '',
                'winning_model_ids': [chosen_winner],
            },
            'reward_model_ids': [chosen_winner],
            'shadow_reference_model_ids': [],
            'metadata': {},
        }
        (tmp_path / 'telemetry' / f'{record["record_id"]}.json').write_text(json.dumps(record))


def _reset_singletons() -> None:
    """Reset every lazy cache the router touches so a tmp_path is honoured."""
    from core.config import get_settings
    from core.registry import get_registry, ollama_cache
    from core.router_mf import _get_quality_model
    from core.storage import get_storage

    get_settings.cache_clear()
    get_storage.cache_clear()
    get_registry.cache_clear()
    _get_quality_model.cache_clear()
    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_learned_router_falls_back_to_heuristic_on_sparse_telemetry(tmp_path):
    """Below ``_MIN_TRAINING_TURNS`` (32) the learned router silently
    degrades to the heuristic. The reasoning trail must say so —
    otherwise an operator who set ``ROITELET_ROUTER=mf`` on a fresh
    install would think the learned path was active when it isn't."""
    from core.schemas import RouterPreferences

    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _seed_tmp(tmp_path, with_telemetry=0)
        _reset_singletons()
        from core.router_mf import LearnedMFRouter

        decision = LearnedMFRouter().route(
            'Refactor this Python module.', RouterPreferences(), top_k=3,
        )
        assert any('heuristic fallback' in r for r in decision.reasoning)
        assert len(decision.selected_model_ids) == 3
    _reset_singletons()


def test_learned_router_promotes_consistent_winner_under_cost_budget(tmp_path):
    """With 64 synthetic turns crowning the same coding winner the
    learned blend must rank it into the routed set. And — the case
    that actually matters — a tight cost budget must still filter
    paid candidates out, even when the learned signal favours them.
    Composition of the two layers is the hard part."""
    from core.schemas import RouterPreferences

    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _seed_tmp(tmp_path, with_telemetry=64)
        _reset_singletons()
        from core.router_mf import LearnedMFRouter

        # Trained path: the learned reasoning label must be present.
        decision = LearnedMFRouter().route(
            'Write a Python function fizzbuzz_check that prints fizz buzz.',
            RouterPreferences(), top_k=3,
        )
        assert any('LearnedMFRouter active' in r for r in decision.reasoning)
        assert 'ollama/qwen2.5:14b-instruct' in decision.selected_model_ids

        # Cost-budget regime composes with the learned blend.
        budget_decision = LearnedMFRouter().route(
            'Write a Python function fizzbuzz_check that prints fizz buzz.',
            RouterPreferences(max_cost_usd=0.0), top_k=3,
        )
        for c in budget_decision.candidates:
            if c.selected:
                assert c.estimated_cost_usd <= 0.0, c.model_id
    _reset_singletons()


def test_env_selector_returns_the_router_named_by_ROITELET_ROUTER():
    """Three router flavours are advertised in the docs; the env selector
    must hand back the right class for each, with the heuristic as the
    default when the env var is unset."""
    from core.router_mf import get_router_from_env

    with pytest.MonkeyPatch().context() as m:
        m.delenv('ROITELET_ROUTER', raising=False)
        assert get_router_from_env().__class__.__name__ == 'RoiteletRouter'

    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_ROUTER', 'mf')
        assert get_router_from_env().__class__.__name__ == 'LearnedMFRouter'

    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_ROUTER', 'calibrated')
        assert get_router_from_env().__class__.__name__ == 'CalibratedCostRouter'
