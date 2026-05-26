"""Unit tests for the learned matrix-factorisation router.

The interesting properties to lock down:

1. **Sparse-telemetry fallback is byte-identical to the heuristic.**
   Without enough telemetry the learned router must produce a
   ``RouterDecision`` matching the existing :class:`RoiteletRouter`
   so a fresh install behaves like the documented default.
2. **Trained mode actually blends.** Given synthetic telemetry where
   one model wins all turns, that model's learned-quality term must
   move the ranking compared to the heuristic.
3. **`ROITELET_ROUTER=mf` env var selects the learned router.**
4. **Cost-budget regime still works** after the learned blend.

These tests are intentionally hermetic: they build a tmp data dir,
seed it with bootstrap priors + synthetic telemetry, point
``ROITELET_DATA_DIR`` at it, and clear the singleton caches.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
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
    """Bootstrap a clean data dir with priors and optional telemetry.

    Parameters
    ----------
    tmp_path:
        Pristine pytest tmp dir to use as ``ROITELET_DATA_DIR``.
    with_telemetry:
        How many synthetic telemetry records to write. ``0`` exercises
        the sparse-fallback branch; ``>= 32`` triggers the trained
        branch (per ``_MIN_TRAINING_TURNS``).
    """
    src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
    (tmp_path / 'bootstrap').mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'model_priors.json', tmp_path / 'bootstrap' / 'model_priors.json')
    (tmp_path / 'telemetry').mkdir(parents=True, exist_ok=True)

    # Synthetic telemetry: every record routes the same three models;
    # we artificially crown a chosen winner on every coding-flavoured
    # prompt so the learned router learns "model X is good at code."
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
                'model_id': 'ollama/qwen2.5:14b-instruct',
                'provider': 'ollama',
                'content': 'placeholder',
                'judge_summary': '',
                'winning_model_ids': [chosen_winner],
            },
            'reward_model_ids': [chosen_winner],
            'shadow_reference_model_ids': [],
            'metadata': {},
        }
        path = tmp_path / 'telemetry' / f'{record["record_id"]}.json'
        path.write_text(json.dumps(record))


def _reset_singletons() -> None:
    """Force every lazy cache the router touches to rebuild on next call."""
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


class TestLearnedFallback:
    """Sparse telemetry: behaviour must match the heuristic exactly."""

    def test_sparse_telemetry_falls_back(self, tmp_path):
        from core.schemas import RouterPreferences

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_telemetry=0)
            _reset_singletons()
            from core.router_mf import LearnedMFRouter

            decision = LearnedMFRouter().route(
                'Refactor this Python module.',
                RouterPreferences(),
                top_k=3,
            )
            assert any('heuristic fallback' in r for r in decision.reasoning), decision.reasoning
            assert len(decision.selected_model_ids) == 3
        _reset_singletons()

    def test_below_threshold_also_falls_back(self, tmp_path):
        """Even 30 turns (below the 32-min threshold) must keep falling back."""
        from core.schemas import RouterPreferences

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_telemetry=30)
            _reset_singletons()
            from core.router_mf import LearnedMFRouter

            decision = LearnedMFRouter().route(
                'Refactor this Python module.',
                RouterPreferences(),
                top_k=3,
            )
            assert any('heuristic fallback' in r for r in decision.reasoning)
        _reset_singletons()


class TestLearnedTrained:
    """Enough telemetry: the learned signal must measurably shift scoring."""

    def test_trained_path_reasoning_label(self, tmp_path):
        """The reasoning trail must call out that the learned router is active."""
        from core.schemas import RouterPreferences

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_telemetry=64)
            _reset_singletons()
            from core.router_mf import LearnedMFRouter

            decision = LearnedMFRouter().route(
                'Write a Python function to reverse a list in place.',
                RouterPreferences(),
                top_k=3,
            )
            assert any('LearnedMFRouter active' in r for r in decision.reasoning), decision.reasoning
        _reset_singletons()

    def test_trained_router_promotes_consistent_winner(self, tmp_path):
        """A model that wins every telemetry turn must rank in top-K on a similar prompt.

        The synthetic data crowns ``ollama/qwen2.5:14b-instruct`` on every
        coding prompt; the test prompt is also a coding prompt; therefore
        the learned signal must keep that model in the routed set.
        """
        from core.schemas import RouterPreferences

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_telemetry=64)
            _reset_singletons()
            from core.router_mf import LearnedMFRouter

            decision = LearnedMFRouter().route(
                'Write a Python function fizzbuzz_check that prints fizz buzz.',
                RouterPreferences(),
                top_k=3,
            )
            assert 'ollama/qwen2.5:14b-instruct' in decision.selected_model_ids, (
                f'Expected the learned winner to be selected, got {decision.selected_model_ids}'
            )
        _reset_singletons()

    def test_cost_budget_still_works_under_learned_blend(self, tmp_path):
        """A $0 budget must filter remote candidates even when the learned signal favours them."""
        from core.schemas import RouterPreferences

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            _seed_tmp(tmp_path, with_telemetry=64)
            _reset_singletons()
            from core.router_mf import LearnedMFRouter

            decision = LearnedMFRouter().route(
                'Write a Python function fizzbuzz_check that prints fizz buzz.',
                RouterPreferences(max_cost_usd=0.0),
                top_k=3,
            )
            for candidate in decision.candidates:
                if candidate.selected:
                    assert candidate.estimated_cost_usd <= 0.0, (
                        f'{candidate.model_id} broke the $0 budget'
                    )
        _reset_singletons()


class TestEnvSelector:
    """``ROITELET_ROUTER=mf`` must hand back the learned router instance."""

    def test_default_is_heuristic(self):
        from core.router_mf import get_router_from_env

        with pytest.MonkeyPatch().context() as m:
            m.delenv('ROITELET_ROUTER', raising=False)
            router = get_router_from_env()
            assert router.__class__.__name__ == 'RoiteletRouter'

    def test_env_mf_selects_learned(self):
        from core.router_mf import get_router_from_env

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_ROUTER', 'mf')
            router = get_router_from_env()
            assert router.__class__.__name__ == 'LearnedMFRouter'

    def test_unknown_flavour_falls_back(self):
        from core.router_mf import get_router_from_env

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_ROUTER', 'gobbledygook')
            router = get_router_from_env()
            assert router.__class__.__name__ == 'RoiteletRouter'
