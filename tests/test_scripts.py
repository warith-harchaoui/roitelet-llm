"""Tests for helper scripts that ship under ``scripts/``.

Scripts live outside the ``core`` package but the project depends on them
during setup and bootstrap-refresh workflows. The Elo-rescaling logic
inside ``scripts/crawl_arena.py`` in particular is a quiet correctness
risk: get the rescaling wrong and the entire bootstrap prior file
silently shifts off-distribution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'


@pytest.fixture(scope='module')
def crawl_arena():
    """Load ``scripts/crawl_arena.py`` as a module without running ``__main__``.

    The script is not packaged, so a regular ``import`` won't find it. We
    use ``importlib`` to load it by path; nothing in the script runs at
    import time because the CLI wiring is gated behind ``if __name__ ==
    '__main__'`` (well, actually ``main()``).
    """
    spec = importlib.util.spec_from_file_location(
        'crawl_arena', _SCRIPTS_DIR / 'crawl_arena.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['crawl_arena'] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('crawl_arena', None)


class TestNormalizeElo:
    """The Elo→Roitelet rescaling pins the bootstrap priors. A single
    monotonic + bounded test pins the contract without inventing four
    micro-cases for the same trapezoid.
    """

    def test_monotonic_with_bounds_anchored(self, crawl_arena):
        n = crawl_arena.normalize_elo
        lo = crawl_arena.ELO_MIN
        hi = crawl_arena.ELO_MAX
        # Below floor → 0.5 (minimum-credible-baseline).
        assert n(lo - 50) == 0.5
        # At ceiling → ROITELET_MAX.
        assert n(hi) == crawl_arena.ROITELET_MAX
        # Above ceiling → same cap, no overflow.
        assert n(hi + 100) == crawl_arena.ROITELET_MAX
        # Midpoint anchors at the middle of the [0.5, 1.5] output band.
        assert n((lo + hi) / 2.0) == pytest.approx(1.0, abs=1e-9)
        # Strictly non-decreasing across a representative sweep.
        sweep = [n(e) for e in (950, 1000, 1100, 1200, 1300, 1400)]
        assert sweep == sorted(sweep)


def test_update_priors_only_touches_known_models(crawl_arena, tmp_path, monkeypatch):
    """Models not present in model_priors.json must be ignored, not auto-added.

    Roitelet treats bootstrap as the curated source of truth; the crawler
    only refines existing entries.
    """
    import json as _json

    fixture = tmp_path / 'data' / 'bootstrap'
    fixture.mkdir(parents=True)
    fake_priors = {
        'openrouter/openai/gpt-4.1': {
            'provider': 'openrouter',
            'local': False,
            'vlm': False,
            'pricing': {'input_per_1k': 0.005, 'output_per_1k': 0.015},
            'latency_s': 3.8,
            'energy_kwh': 0.0006,
            'capabilities': {'reasoning': 0.93, 'analysis': 0.91},
        }
    }
    priors_path = fixture / 'model_priors.json'
    priors_path.write_text(_json.dumps(fake_priors), encoding='utf-8')

    # Redirect the script's hard-coded priors path to our temp copy.
    monkeypatch.setattr(crawl_arena, '__file__', str(tmp_path / 'scripts' / 'crawl_arena.py'))
    (tmp_path / 'scripts').mkdir(exist_ok=True)

    crawl_arena.update_priors([
        {'model': 'gpt-4.1', 'elo': 1280},               # matches a known model
        {'model': 'totally-unknown-model', 'elo': 1290},  # must be ignored
    ])

    updated = _json.loads(priors_path.read_text())
    gpt = updated['openrouter/openai/gpt-4.1']['capabilities']
    # The bumped fields are 'reasoning' and 'analysis' — both should have moved.
    assert gpt['reasoning'] != 0.93 or gpt['analysis'] != 0.91, (
        'update_priors did not modify the matched entry'
    )
    # No new entries should have been added.
    assert set(updated.keys()) == {'openrouter/openai/gpt-4.1'}


def test_update_priors_writes_meta_block(crawl_arena, tmp_path, monkeypatch):
    """Touched entries must gain a ``_meta`` block recording provenance."""
    import json as _json

    fixture = tmp_path / 'data' / 'bootstrap'
    fixture.mkdir(parents=True)
    fake_priors = {
        'openrouter/openai/gpt-4.1': {
            'provider': 'openrouter',
            'local': False,
            'vlm': False,
            'pricing': {'input_per_1k': 0.005, 'output_per_1k': 0.015},
            'latency_s': 3.8,
            'energy_kwh': 0.0006,
            'capabilities': {'reasoning': 0.93, 'analysis': 0.91},
        }
    }
    priors_path = fixture / 'model_priors.json'
    priors_path.write_text(_json.dumps(fake_priors), encoding='utf-8')

    monkeypatch.setattr(crawl_arena, '__file__', str(tmp_path / 'scripts' / 'crawl_arena.py'))
    (tmp_path / 'scripts').mkdir(exist_ok=True)

    crawl_arena.update_priors(
        [{'model': 'gpt-4.1', 'elo': 1280}],
        source='https://example.com/leaderboard',
    )

    updated = _json.loads(priors_path.read_text())
    meta = updated['openrouter/openai/gpt-4.1'].get('_meta')
    assert meta is not None, '_meta block missing on a touched entry'
    assert meta['source'] == 'https://example.com/leaderboard'
    assert meta['elo_raw'] == 1280
    assert 'refreshed_at' in meta
    # Timestamp must be a parseable ISO-8601 string.
    import datetime
    datetime.datetime.fromisoformat(meta['refreshed_at'])
