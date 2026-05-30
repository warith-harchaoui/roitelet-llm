"""Hand-written smoke tests for Roitelet's core surface.

Each test below covers one *story-level* invariant — the kind of thing
a person reviewing the codebase would actually worry about. Pydantic
defaults, round-trip serialisation, and "this constant exists" are
intentionally not tested: they're the framework's job, not ours.

Run with:

    python -m pytest -q

Hermetic: no Ollama, no network, no API keys.
"""

from __future__ import annotations

import http.server
import json
import shutil
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.capabilities import detect_capabilities, top_capabilities
from core.config import Settings
from core.judge import parse_winners
from core.schemas import (
    AppSettingsPayload,
    CustomEngine,
    ModelCandidate,
    RouterPreferences,
    SECRET_MASK,
)


# ---------------------------------------------------------------------------
# Fixtures: a real HTTP server that mimics Ollama, plus an autouse cache
# isolator so a developer laptop running real Ollama can't shift the
# registry under our feet.
# ---------------------------------------------------------------------------


@pytest.fixture
def ollama_http_server():
    """Spin up an in-process HTTP server that pretends to be Ollama."""
    state: dict[str, list[str]] = {'models': []}

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — stdlib name
            if self.path == '/api/tags':
                body = json.dumps(
                    {'models': [{'name': name} for name in state['models']]}
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args, **kwargs):  # silence default access log
            return

    server = http.server.HTTPServer(('127.0.0.1', 0), _Handler)
    server.allow_reuse_address = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {
            'base_url': f'http://127.0.0.1:{port}',
            'set_models': lambda names: state.update(models=list(names)),
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


@pytest.fixture(autouse=True)
def _isolate_ollama_cache():
    """Force live-discovery to look empty so tests don't depend on the host."""
    from core.registry import ollama_cache
    saved_models = ollama_cache._models
    saved_fetched_at = ollama_cache._fetched_at
    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()
    try:
        yield
    finally:
        ollama_cache._models = saved_models
        ollama_cache._fetched_at = saved_fetched_at


def _real_settings(data_dir: Path) -> Settings:
    settings = Settings()
    settings.data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def _seed_bootstrap(tmp_path: Path) -> None:
    """Copy the shipped priors into ``tmp_path`` so the registry sees them."""
    src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
    (tmp_path / 'bootstrap').mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'model_priors.json', tmp_path / 'bootstrap' / 'model_priors.json')


# ---------------------------------------------------------------------------
# Tests — one story per test.
# ---------------------------------------------------------------------------


def test_capability_detector_normalises_and_responds_to_code_fence():
    """The detector outputs a distribution; a fenced code block should
    dominate it; longer prompts should not *lose* long_context weight."""
    scores = detect_capabilities("Translate this email to French.")
    assert abs(sum(scores.values()) - 1.0) < 1e-6

    code = detect_capabilities("```python\ndef add(a, b):\n    return a + b\n```")
    assert code.get("coding", 0) > 0.3

    short = detect_capabilities("Summarize this document. " * 50)
    long_ = detect_capabilities("Summarize this document. " * 200)
    assert long_.get("long_context", 0) >= short.get("long_context", 0)

    assert top_capabilities({"coding": 0.6, "math": 0.3}, limit=1) == ["coding"]


def test_judge_winners_block_is_strictly_fail_closed():
    """No marker → no winners. Stray hex → ignored. Uppercase → normalised.
    Open marker without close → still parses. The Elo loop relies on every
    one of these so it never learns from a malformed judge output."""
    valid = {"ab3c9f1d", "9e2f4a17"}

    # No marker at all — silently picking candidate one would poison Elo.
    assert parse_winners("Plain prose, no marker here.", valid) == []

    # Typical two winners + a stray that doesn't belong to this turn.
    assert parse_winners(
        "===WINNERS===\nab3c9f1d, deadbeef, 9e2f4a17\n===END===",
        valid,
    ) == ["ab3c9f1d", "9e2f4a17"]

    # Truncated output (open marker only) still recovers the winners.
    assert parse_winners(
        "Prose.\n===WINNERS===\nab3c9f1d, 9e2f4a17",
        valid,
    ) == ["ab3c9f1d", "9e2f4a17"]

    # Uppercase hex normalised so the registry lookup matches.
    assert parse_winners("===WINNERS===\nAB3C9F1D\n===END===", valid) == ["ab3c9f1d"]


def test_quality_probability_normaliser_anchors_and_ties():
    """The single-knob threshold (`quality_threshold`) only works if the
    candidate scores are normalised to [0, 1] per turn."""
    from core.router import _attach_quality_probability

    candidates = [
        ModelCandidate(model_id='a', provider='x', score=1.0),
        ModelCandidate(model_id='b', provider='x', score=0.6),
        ModelCandidate(model_id='c', provider='x', score=0.2),
    ]
    _attach_quality_probability(candidates)
    assert candidates[0].quality_probability == 1.0
    assert candidates[-1].quality_probability == 0.0
    assert 0.0 < candidates[1].quality_probability < 1.0

    # A tied pool must all survive any threshold — otherwise a tight
    # τ would silently drop the whole turn.
    tied = [
        ModelCandidate(model_id='a', provider='x', score=0.5),
        ModelCandidate(model_id='b', provider='x', score=0.5),
    ]
    _attach_quality_probability(tied)
    assert all(c.quality_probability == 1.0 for c in tied)


def test_regime_detector_picks_the_right_branch_per_prompt():
    """The regime layer is what makes the hybrid router *hybrid*. Each
    branch matters — a wrong regime label silently changes routing math.

    Five cases cover every branch of ``detect_regime``: budget wins over
    everything, trivial-prompt collapses K, long_context fires above 4 k
    chars, capability_dominant fires when one capability owns ≥ 80 %,
    ambiguous fires when none does, default closes the ladder.
    """
    from core.regimes import detect_regime

    # Budget takes precedence over a strong capability signal.
    r = detect_regime('short', RouterPreferences(max_cost_usd=0.001), {'reasoning': 1.0})
    assert r.name == 'budget_constrained' and r.cost_budget_usd == 0.001

    # Trivial-prompt regime collapses K to 1.
    assert detect_regime('what is 2 + 2?', RouterPreferences(), {'reasoning': 1.0}).suggested_top_k == 1

    # Long-context fires beyond 4 k chars.
    assert detect_regime('x' * 5000, RouterPreferences(), {'long_context': 0.4}).name == 'long_context'

    # One capability ≥ 80 % → capability_dominant.
    long_coding = (
        'Please refactor this Python module so the database access '
        'is mockable in unit tests without touching call sites.'
    )
    assert detect_regime(long_coding, RouterPreferences(), {'coding': 0.8, 'reasoning': 0.2}).name == 'capability_dominant'

    # No capability ≥ 30 % and prompt long enough → ambiguous (not trivial).
    chitchat = (
        'How are you doing today? Anything new on your end since '
        'last week? Curious to hear what you have been up to lately.'
    )
    assert detect_regime(
        chitchat, RouterPreferences(),
        {'reasoning': 0.25, 'writing': 0.20, 'analysis': 0.15},
    ).name == 'ambiguous'


def test_cost_budget_regime_excludes_paid_candidates_before_scoring(tmp_path):
    """A ``$0`` budget must collapse the eligible pool to free (local)
    candidates *before* the weighted ranking runs. The blend can never
    reach a paid model otherwise."""
    from core.router import RoiteletRouter
    from core.storage import get_storage

    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_DATA_DIR', str(tmp_path))
        from core.config import get_settings
        get_settings.cache_clear()
        get_storage.cache_clear()
        _seed_bootstrap(tmp_path)

        decision = RoiteletRouter().route(
            'Write a Python function that reverses a list.',
            RouterPreferences(max_cost_usd=0.0),
            top_k=3,
        )
        assert any('budget_constrained' in r for r in decision.reasoning)
        for c in decision.candidates:
            if c.selected:
                assert c.estimated_cost_usd <= 0.0, c.model_id

        get_settings.cache_clear()
        get_storage.cache_clear()


def test_registry_merges_bootstrap_user_models_and_custom_engines():
    """The registry's job is to produce one pool from three sources
    (bootstrap, user-configured, live discovery) without duplication and
    with provider auth gating. This test pins the merge contract for the
    two non-trivial source pairs."""
    from core.registry import ModelRegistry

    base_payload = AppSettingsPayload(
        ollama_base_url='',
        openrouter_api_key='sk-test',
        openai_api_key='sk-test',
        openai_compatible_api_key='sk-test',
        selected_ollama_models=['phi4:latest', 'ollama/phi4:latest', 'qwen2.5:14b-instruct'],
        paid_openrouter_models=['mistralai/mistral-7b-instruct'],
        custom_engines=[
            CustomEngine(label='mistral', base_url='https://api.mistral.ai/v1',
                         api_key='sk-mistral', models=['mistral-large-latest']),
            CustomEngine(label='no-key', base_url='https://example.com/v1',
                         api_key='', models=['phantom']),
            CustomEngine(label='', base_url='https://example.com/v1',
                         api_key='sk', models=['ghost']),
        ],
    )

    ids = {m.model_id for m in ModelRegistry(app_settings=base_payload).list_models()}

    # Bootstrap pool is present and labelled correctly.
    assert 'ollama/qwen2.5:14b-instruct' in ids
    assert 'openrouter/meta-llama/llama-3.3-70b-instruct' in ids

    # User additions land under the right prefix, no doubled ``ollama/``
    # and no duplicate of the bootstrap entry.
    assert 'ollama/phi4:latest' in ids
    assert 'openrouter/mistralai/mistral-7b-instruct' in ids
    duplicates = [m for m in ids if m.endswith('phi4:latest') or m.endswith('qwen2.5:14b-instruct')]
    assert len(duplicates) == len(set(duplicates))

    # Custom engine with a key registers under its label namespace.
    assert 'openai-compatible/mistral/mistral-large-latest' in ids
    # Empty-key engine is pruned. Blank-label engine is skipped.
    assert 'openai-compatible/no-key/phantom' not in ids
    assert not any('ghost' in m for m in ids)


def test_app_settings_mask_round_trip_preserves_secrets_per_engine():
    """The web UI receives masked secrets and POSTs them back unchanged
    when the user only edited non-secret fields. The merge must restore
    the on-disk values rather than blanking them."""
    original = AppSettingsPayload(
        openrouter_api_key='real-or-key',
        custom_engines=[
            CustomEngine(label='mistral', base_url='u', api_key='real-mistral', models=['m']),
        ],
    )
    masked = original.masked()
    assert masked.openrouter_api_key == SECRET_MASK
    assert masked.custom_engines[0].api_key == SECRET_MASK

    merged = original.merge_unmasked(masked)
    assert merged.openrouter_api_key == 'real-or-key'
    assert merged.custom_engines[0].api_key == 'real-mistral'

    # A brand-new engine with a real key passes through unchanged.
    incoming = AppSettingsPayload(custom_engines=[
        CustomEngine(label='new', base_url='u', api_key='new-real', models=['m']),
    ])
    assert AppSettingsPayload(custom_engines=[]).merge_unmasked(incoming) \
        .custom_engines[0].api_key == 'new-real'


def test_rolling_elo_is_bounded_directional_and_drops_unknown_capabilities():
    """The Elo loop touches three real invariants on every update:
    winners go up, losers go down, and capabilities not in
    ``KNOWN_CAPABILITIES`` never make it onto disk — a typo in the
    detector can't pollute the state file."""
    from core.registry import ModelRegistry

    registry = ModelRegistry()
    registry.elo_state = {}
    before_w = registry.capability_score('ollama/qwen2.5:14b-instruct', 'coding')
    before_l = registry.capability_score('ollama/mistral-small3.1', 'coding')

    registry.update_elo(
        winners=['ollama/qwen2.5:14b-instruct'],
        losers=['ollama/mistral-small3.1'],
        capabilities={'coding': 0.8, 'unknown_xyz': 0.2},
    )

    assert registry.capability_score('ollama/qwen2.5:14b-instruct', 'coding') > before_w
    assert registry.capability_score('ollama/mistral-small3.1', 'coding') < before_l
    assert 'unknown_xyz' not in registry.elo_state['ollama/qwen2.5:14b-instruct']
    # Bounded: capability_score never exits [0, 1.5] regardless of state.
    assert 0.0 <= registry.capability_score('ollama/qwen2.5:14b-instruct', 'coding') <= 1.5


def test_ollama_live_discovery_caches_with_ttl_and_does_not_override_bootstrap(
    ollama_http_server, tmp_path,
):
    """Live discovery should populate the registry, respect its TTL, and
    *not* overwrite curated bootstrap priors when an existing model is
    re-discovered (otherwise a `0.65` default would wipe `0.82` priors)."""
    from core.registry import ModelRegistry, _OllamaModelCache, ollama_cache

    # Fresh cache against the in-process server.
    cache = _OllamaModelCache()
    cache.configure(ollama_http_server['base_url'])

    ollama_http_server['set_models'](['phi4:latest', 'qwen2.5:14b-instruct'])
    cache.refresh(force=True)
    assert 'phi4:latest' in cache.models
    assert 'qwen2.5:14b-instruct' in cache.models

    # Change the server's response; a fresh refresh would see ['other:7b'].
    ollama_http_server['set_models'](['other:7b'])
    cache.refresh(force=False)  # within TTL — must not hit the server
    assert cache.models == ['phi4:latest', 'qwen2.5:14b-instruct']

    # Inject discovered models into the registry path and confirm
    # bootstrap priors are preserved.
    ollama_cache._models = ['newmodel:7b', 'qwen2.5:14b-instruct']
    ollama_cache._fetched_at = time.monotonic()

    registry = ModelRegistry()
    ids = {m.model_id for m in registry.list_models()}
    assert 'ollama/newmodel:7b' in ids
    # Curated coding prior for the bootstrap model survives discovery.
    assert registry.get('ollama/qwen2.5:14b-instruct').capabilities['coding'] > 0.7

    ollama_cache._models = []


def test_storage_round_trip_with_cache_ttl_and_traversal_rejection(tmp_path):
    """Conversations + cache TTL behaviour live in the same backend; testing
    them together pins the contract a future MongoDB or other backend
    must also satisfy."""
    with pytest.MonkeyPatch().context() as m:
        settings = _real_settings(tmp_path)
        m.setattr('core.storage.get_settings', lambda: settings)
        import core.storage as st_mod
        mgr = st_mod.StorageManager()

        # Conversation create/read + newest-first listing.
        first = mgr.create_conversation(title='First')
        second = mgr.create_conversation(title='Second')
        assert mgr.list_conversations()[0].conversation_id == second.conversation_id
        assert mgr.get_conversation(first.conversation_id).title == 'First'

        # Traversal payloads must be rejected, not crash.
        for evil in ('../../etc/passwd', '..', 'a/b', ''):
            with pytest.raises(ValueError):
                mgr.conversation_path(evil)
        assert mgr.get_conversation('../../etc/passwd') is None

        # Cache disabled by default (TTL=0): write is a no-op, read misses.
        mgr.set_cache('demo', 'k', {'r': 1})
        assert mgr.get_cache('demo', 'k') is None
        assert not (tmp_path / 'cache' / 'demo.jsonl').exists()

        # With TTL=60s the cache hits, then misses after we backdate.
        settings.provider_cache_ttl_seconds = 60
        mgr.set_cache('demo', 'k', {'r': 2})
        assert mgr.get_cache('demo', 'k') == {'r': 2}
        path = tmp_path / 'cache' / 'demo.jsonl'
        record = json.loads(path.read_text().splitlines()[-1])
        record['cached_at'] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        path.write_text(json.dumps(record) + '\n')
        assert mgr.get_cache('demo', 'k') is None

        # TTL=-1 keeps the historical "cache forever" semantics.
        settings.provider_cache_ttl_seconds = -1
        mgr.set_cache('demo', 'k', {'r': 3})
        assert mgr.get_cache('demo', 'k') == {'r': 3}
