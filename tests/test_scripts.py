"""Tests for helper scripts that ship under ``scripts/``.

Two tests. The bootstrap refresh path is a quiet correctness risk —
get the rescaling wrong and the whole priors file silently shifts —
so we pin the rescaling shape and the update contract.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / 'scripts'


@pytest.fixture(scope='module')
def crawl_arena():
    """Load ``scripts/crawl_arena.py`` as a module without running ``__main__``."""
    spec = importlib.util.spec_from_file_location(
        'crawl_arena', _SCRIPTS_DIR / 'crawl_arena.py',
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['crawl_arena'] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop('crawl_arena', None)


def test_normalize_elo_is_monotonic_with_bounds_anchored(crawl_arena):
    """The Elo→Roitelet rescaling: below the floor → 0.5, above the
    ceiling → ROITELET_MAX, midpoint anchors at the middle of the
    [0.5, 1.5] output band, and the function is non-decreasing on a
    representative sweep."""
    n = crawl_arena.normalize_elo
    lo, hi, top = crawl_arena.ELO_MIN, crawl_arena.ELO_MAX, crawl_arena.ROITELET_MAX
    assert n(lo - 50) == 0.5
    assert n(hi) == top and n(hi + 100) == top
    assert n((lo + hi) / 2.0) == pytest.approx(1.0, abs=1e-9)
    sweep = [n(e) for e in (950, 1000, 1100, 1200, 1300, 1400)]
    assert sweep == sorted(sweep)


def test_update_priors_only_touches_known_models_and_records_provenance(
    crawl_arena, tmp_path, monkeypatch,
):
    """The crawler refines bootstrap entries; it must not invent them.

    Roitelet treats ``model_priors.json`` as the curated source of
    truth. An unknown model id in the leaderboard feed must be
    silently dropped (not auto-added with default values), and a
    matched entry must gain a ``_meta`` block recording the
    provenance — so a future ``git blame`` on the priors file points
    at the leaderboard URL + the raw Elo it was derived from.
    """
    fixture = tmp_path / 'data' / 'bootstrap'
    fixture.mkdir(parents=True)
    priors_path = fixture / 'model_priors.json'
    priors_path.write_text(json.dumps({
        'openrouter/openai/gpt-4.1': {
            'provider': 'openrouter', 'local': False, 'vlm': False,
            'pricing': {'input_per_1k': 0.005, 'output_per_1k': 0.015},
            'latency_s': 3.8, 'energy_kwh': 0.0006,
            'capabilities': {'reasoning': 0.93, 'analysis': 0.91},
        },
    }), encoding='utf-8')

    monkeypatch.setattr(crawl_arena, '__file__',
                        str(tmp_path / 'scripts' / 'crawl_arena.py'))
    (tmp_path / 'scripts').mkdir(exist_ok=True)

    crawl_arena.update_priors(
        [
            {'model': 'gpt-4.1', 'elo': 1280},
            {'model': 'totally-unknown-model', 'elo': 1290},
        ],
        source='https://example.com/leaderboard',
    )
    updated = json.loads(priors_path.read_text())

    # The matched entry was refreshed.
    gpt = updated['openrouter/openai/gpt-4.1']
    assert gpt['capabilities']['reasoning'] != 0.93 or gpt['capabilities']['analysis'] != 0.91

    # The unknown model was NOT added.
    assert set(updated.keys()) == {'openrouter/openai/gpt-4.1'}

    # Provenance recorded.
    meta = gpt['_meta']
    assert meta['source'] == 'https://example.com/leaderboard'
    assert meta['elo_raw'] == 1280
    assert 'refreshed_at' in meta
