"""Minimal smoke tests for Roitelet LLM.

Run with:
    pytest tests/ -q

Notes
-----
These tests do not require a running Ollama instance or API keys.
They validate the pure-Python routing, capability detection, registry,
live discovery cache, and storage logic in total isolation.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
from datetime import UTC
from pathlib import Path

import pytest

from core.capabilities import detect_capabilities, top_capabilities
from core.config import Settings
from core.judge import parse_winners
from core.schemas import RouterPreferences

# ---------------------------------------------------------------------------
# Test fixtures: real HTTP server for Ollama /api/tags
# ---------------------------------------------------------------------------

@pytest.fixture
def ollama_http_server():
    """Start a real HTTP server that mimics Ollama's ``/api/tags`` endpoint.

    Yields a dict with ``base_url`` and a ``set_models`` callable so each test
    can drive the response body deterministically without mocking ``httpx``.
    """
    state = {'models': []}

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

    def set_models(names: list[str]) -> None:
        state['models'] = list(names)

    try:
        yield {
            'base_url': f'http://127.0.0.1:{port}',
            'set_models': set_models,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

class TestDetectCapabilities:
    """The keyword detector is meant to *bias* the router, not classify.

    Three invariants are worth nailing down; the rest is keyword-list
    bookkeeping that the detector + the regime tests already exercise
    transitively.
    """

    def test_scores_sum_to_one(self):
        """A normalised distribution is what the router multiplies against."""
        scores = detect_capabilities("Translate this email to French.")
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-6, f"Scores must sum to 1.0, got {total}"

    def test_code_fence_strongly_signals_coding(self):
        """A fenced code block in the prompt should dominate the distribution."""
        scores = detect_capabilities("```python\ndef add(a, b):\n    return a + b\n```")
        assert scores.get("coding", 0) > 0.3

    def test_long_prompts_trigger_long_context(self):
        """5 000 chars should rank higher than 1 250 on the long_context axis."""
        short = detect_capabilities("Summarize this document. " * 50)
        long = detect_capabilities("Summarize this document. " * 200)
        assert long.get("long_context", 0) >= short.get("long_context", 0)

    def test_top_capabilities_returns_dominant_first(self):
        top = top_capabilities({"coding": 0.6, "math": 0.3, "reasoning": 0.1}, limit=2)
        assert top[0] == "coding"
        assert len(top) == 2


# ---------------------------------------------------------------------------
# Judge output parsing
# ---------------------------------------------------------------------------

class TestParseWinners:
    """The judge's winners block must fail closed — silently picking
    candidate one when the marker is missing would poison the Elo loop.
    """

    VALID = {"ab3c9f1d", "9e2f4a17", "1c0e7b22"}

    def test_typical_two_winners(self):
        assert parse_winners(
            "Both good.\n===WINNERS===\nab3c9f1d, 1c0e7b22\n===END===",
            self.VALID,
        ) == ["ab3c9f1d", "1c0e7b22"]

    def test_no_marker_means_no_winners(self):
        """The whole point of the sentinel — no winners block, no Elo update."""
        assert parse_winners("Plain prose, no marker here.", self.VALID) == []

    def test_stray_hex_outside_the_valid_set_is_ignored(self):
        """A token from a prior turn (or fabricated) must not contaminate."""
        text = "===WINNERS===\nab3c9f1d, deadbeef, 9e2f4a17\n===END==="
        assert parse_winners(text, self.VALID) == ["ab3c9f1d", "9e2f4a17"]

    def test_truncated_output_with_only_open_marker_still_parses(self):
        """A cut-off judge response shouldn't lose its winners."""
        text = "Prose.\n===WINNERS===\nab3c9f1d, 9e2f4a17"
        assert parse_winners(text, self.VALID) == ["ab3c9f1d", "9e2f4a17"]

    def test_uppercase_hex_is_normalised(self):
        text = "===WINNERS===\nAB3C9F1D\n===END==="
        assert parse_winners(text, self.VALID) == ["ab3c9f1d"]


# ---------------------------------------------------------------------------
# Router preferences and schema
# ---------------------------------------------------------------------------

# RouterPreferences itself has no custom logic — Pydantic does the
# defaulting and round-tripping. The behaviour that matters (each
# preference reaches the router and changes a decision) is exercised
# by the cost-budget routing tests below and by the API integration
# tests in test_commands.py.


class TestQualityProbabilityNormaliser:
    """The Elo→probability normaliser is what makes a single threshold knob work.

    Without it, ``quality_threshold`` would compare apples to oranges
    across turns. The normaliser maps every turn's pool to [0, 1] with
    the best candidate at 1.0 — same shape as RouteLLM's threshold knob,
    even though derived from rolling Elo + capability priors rather than
    a preference-trained classifier.
    """

    def test_best_and_worst_anchor_at_one_and_zero(self):
        from core.router import _attach_quality_probability
        from core.schemas import ModelCandidate

        candidates = [
            ModelCandidate(model_id='a', provider='x', score=1.0),
            ModelCandidate(model_id='b', provider='x', score=0.6),
            ModelCandidate(model_id='c', provider='x', score=0.2),
        ]
        _attach_quality_probability(candidates)
        assert candidates[0].quality_probability == 1.0
        assert candidates[-1].quality_probability == 0.0
        assert 0.0 < candidates[1].quality_probability < 1.0

    def test_tied_pool_survives_any_threshold(self):
        """If everyone ties, no candidate should be dropped by a threshold filter."""
        from core.router import _attach_quality_probability
        from core.schemas import ModelCandidate

        candidates = [
            ModelCandidate(model_id='a', provider='x', score=0.5),
            ModelCandidate(model_id='b', provider='x', score=0.5),
        ]
        _attach_quality_probability(candidates)
        assert all(c.quality_probability == 1.0 for c in candidates)


# ---------------------------------------------------------------------------
# Regime detection: hybrid routing math
# ---------------------------------------------------------------------------


class TestDetectRegime:
    """Regime detector must classify routing calls into stable buckets.

    Each test isolates one branch of the if-ladder in
    :func:`core.regimes.detect_regime` so a future re-order shows up as
    a single targeted failure rather than a cascade.
    """

    def test_budget_takes_precedence(self):
        from core.regimes import detect_regime
        regime = detect_regime(
            'short prompt',
            RouterPreferences(max_cost_usd=0.001),
            {'reasoning': 1.0},
        )
        assert regime.name == 'budget_constrained'
        assert regime.cost_budget_usd == 0.001

    def test_trivial_prompt(self):
        from core.regimes import detect_regime
        regime = detect_regime(
            'what is 2 + 2?',
            RouterPreferences(),
            {'reasoning': 1.0},
        )
        assert regime.name == 'trivial'
        assert regime.suggested_top_k == 1

    def test_long_context(self):
        from core.regimes import detect_regime
        long_prompt = 'x' * 5000
        regime = detect_regime(long_prompt, RouterPreferences(), {'long_context': 0.4})
        assert regime.name == 'long_context'

    def test_dominant_capability(self):
        from core.regimes import detect_regime
        # > 80 chars so 'trivial' doesn't fire; one capability owns 80 %.
        prompt = (
            'Please refactor this Python module so the database access '
            'is mockable in unit tests without touching call sites.'
        )
        regime = detect_regime(prompt, RouterPreferences(), {'coding': 0.8, 'reasoning': 0.2})
        assert regime.name == 'capability_dominant'

    def test_ambiguous(self):
        from core.regimes import detect_regime
        # Long enough not to trip the trivial heuristic, but no
        # capability above 30 % — keyword detector found nothing.
        prompt = (
            'How are you doing today? Anything new on your end since '
            'last week? Curious to hear what you have been up to lately, '
            'no particular topic in mind here.'
        )
        regime = detect_regime(
            prompt,
            RouterPreferences(),
            {'reasoning': 0.25, 'writing': 0.20, 'analysis': 0.15},
        )
        assert regime.name == 'ambiguous'

    def test_default_when_none_match(self):
        from core.regimes import detect_regime
        regime = detect_regime(
            'Write a unit test that asserts the cache eviction strategy '
            'fires after a TTL window of 30 seconds.',
            RouterPreferences(),
            {'coding': 0.45, 'reasoning': 0.45, 'analysis': 0.10},
        )
        assert regime.name == 'default'


# ---------------------------------------------------------------------------
# Router: cost-budget regime filters candidates pre-scoring
# ---------------------------------------------------------------------------


class TestCostBudgetRouting:
    """A tight budget must exclude expensive candidates from selection."""

    def test_budget_excludes_paid_models(self, tmp_path):
        """A $0 budget collapses the selection to free (local) candidates."""
        from core.router import RoiteletRouter
        from core.storage import get_storage

        # Pristine working dir so registry sees clean state.
        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            # Drop any cached singletons so they pick up the new env.
            from core.config import get_settings
            get_settings.cache_clear()
            get_storage.cache_clear()
            # Seed the data directory with the project's real bootstrap
            # priors so the router has the standard pool to filter.
            import shutil
            src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
            (tmp_path / 'bootstrap').mkdir(parents=True, exist_ok=True)
            shutil.copy(src / 'model_priors.json', tmp_path / 'bootstrap' / 'model_priors.json')

            from core.registry import ollama_cache
            ollama_cache._models = []
            ollama_cache._fetched_at = time.monotonic()

            router = RoiteletRouter()
            decision = router.route(
                'Write a Python function that reverses a list.',
                RouterPreferences(max_cost_usd=0.0),
                top_k=3,
            )

            # Cost-budget regime must show up in reasoning.
            assert any('budget_constrained' in r for r in decision.reasoning), decision.reasoning
            # Every selected candidate must be free (sum of pricing == 0).
            for candidate in decision.candidates:
                if candidate.selected:
                    assert candidate.estimated_cost_usd <= 0.0, (
                        f'{candidate.model_id} exceeded $0 budget: {candidate.estimated_cost_usd}'
                    )

            get_settings.cache_clear()
            get_storage.cache_clear()

    def test_no_budget_leaves_all_candidates(self, tmp_path):
        """Without a budget, paid candidates remain in the pool."""
        from core.router import RoiteletRouter
        from core.storage import get_storage

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_DATA_DIR', str(tmp_path))
            from core.config import get_settings
            get_settings.cache_clear()
            get_storage.cache_clear()
            import shutil
            src = Path(__file__).resolve().parent.parent / 'data' / 'bootstrap'
            (tmp_path / 'bootstrap').mkdir(parents=True, exist_ok=True)
            shutil.copy(src / 'model_priors.json', tmp_path / 'bootstrap' / 'model_priors.json')

            # Persist API-key sentinels so remote candidates aren't auto-pruned.
            from core.schemas import AppSettingsPayload
            settings_payload = AppSettingsPayload(
                openrouter_api_key='sk-test',
                openai_api_key='sk-test',
            )
            storage = get_storage()
            storage.save_app_settings(settings_payload)

            from core.registry import ollama_cache
            ollama_cache._models = []
            ollama_cache._fetched_at = time.monotonic()

            router = RoiteletRouter()
            decision = router.route(
                'Write a Python function that reverses a list.',
                RouterPreferences(),
                top_k=3,
            )

            cost_values = [c.estimated_cost_usd for c in decision.candidates]
            assert any(cost > 0 for cost in cost_values), (
                'Paid candidates should be present when budget is None'
            )

            get_settings.cache_clear()
            get_storage.cache_clear()


# ---------------------------------------------------------------------------
# Registry: bootstrap loading and model injection
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_ollama_cache():
    """Force the live-discovery cache to behave as if Ollama is empty.

    The registry's un-pulled-model prune now drops bootstrap ``ollama/*``
    entries when discovery returns models that don't include them. On a
    developer laptop with real Ollama running, that would clobber any
    registry assertion that expects the full bootstrap pool. We snapshot
    + reset around every test for hermetic behaviour.
    """
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

def _make_registry(
    extra_ollama: list[str] = None,
    extra_openrouter: list[str] = None,
):
    """Build a registry backed by the real bootstrap file.

    Explicitly resets the live-discovery cache so the registry under test
    sees an empty Ollama discovery — that way the un-pulled-model prune
    in :meth:`ModelRegistry._merge_live_ollama` is skipped and the
    bootstrap pool is preserved end-to-end. Without this, running tests
    on a developer machine with real Ollama installed (returning a
    different set of tags) would drop bootstrap entries the assertions
    rely on.
    """
    from core.registry import ModelRegistry, ollama_cache
    from core.schemas import AppSettingsPayload

    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()  # mark as "fresh and empty"

    payload = AppSettingsPayload(
        ollama_base_url='',  # disable configure() so refresh isn't triggered
        selected_ollama_models=extra_ollama or [],
        paid_openrouter_models=extra_openrouter or [],
        # Sentinel keys so _prune_unauthorized_remotes leaves bootstrap
        # remotes in place. The tests don't make network calls, they only
        # assert registration shape.
        openrouter_api_key='sk-test',
        openai_api_key='sk-test',
        openai_compatible_api_key='sk-test',
    )
    return ModelRegistry(app_settings=payload)


class TestModelRegistry:
    def test_bootstrap_loads(self):
        registry = _make_registry()
        models = registry.list_models()
        assert len(models) >= 8, "Bootstrap should include at least 8 models"

    def test_model_ids_present(self):
        registry = _make_registry()
        ids = {m.model_id for m in registry.list_models()}
        assert "ollama/qwen2.5:14b-instruct" in ids
        assert "openrouter/meta-llama/llama-3.3-70b-instruct" in ids

    def test_local_flag(self):
        registry = _make_registry()
        spec = registry.get("ollama/qwen2.5:14b-instruct")
        assert spec.local is True
        spec_remote = registry.get("openrouter/openai/gpt-4.1")
        assert spec_remote.local is False

    def test_user_ollama_model_injection(self):
        registry = _make_registry(extra_ollama=["phi4:latest"])
        ids = {m.model_id for m in registry.list_models()}
        assert "ollama/phi4:latest" in ids, "User-configured Ollama model should appear in registry"

    def test_user_openrouter_model_injection(self):
        registry = _make_registry(extra_openrouter=["mistralai/mistral-7b-instruct"])
        ids = {m.model_id for m in registry.list_models()}
        assert "openrouter/mistralai/mistral-7b-instruct" in ids

    def test_custom_engines_register_with_label_namespace(self):
        """Each custom engine's models register as openai-compatible/<label>/<model>."""
        from core.registry import ModelRegistry, ollama_cache
        from core.schemas import AppSettingsPayload, CustomEngine

        ollama_cache._models = []
        ollama_cache._fetched_at = time.monotonic()

        payload = AppSettingsPayload(
            ollama_base_url='',
            openai_api_key='sk-test',
            openrouter_api_key='sk-test',
            custom_engines=[
                CustomEngine(
                    label='mistral',
                    base_url='https://api.mistral.ai/v1',
                    api_key='sk-mistral',
                    models=['mistral-large-latest', 'mistral-medium'],
                ),
                CustomEngine(
                    label='together',
                    base_url='https://api.together.xyz/v1',
                    api_key='sk-together',
                    models=['mistralai/Mistral-7B-Instruct'],
                ),
            ],
        )
        registry = ModelRegistry(app_settings=payload)
        ids = {m.model_id for m in registry.list_models()}
        # Label-prefixed namespace prevents collisions between engines
        # that happen to serve the same upstream model name.
        assert 'openai-compatible/mistral/mistral-large-latest' in ids
        assert 'openai-compatible/mistral/mistral-medium' in ids
        assert 'openai-compatible/together/mistralai/Mistral-7B-Instruct' in ids
        for mid in ids:
            if mid.startswith('openai-compatible/mistral/') or mid.startswith('openai-compatible/together/'):
                spec = registry.get(mid)
                assert spec.provider == 'openai-compatible'
                assert spec.local is False

    def test_custom_engine_with_empty_key_is_pruned(self):
        """An engine with no api_key should have its models auto-pruned."""
        from core.registry import ModelRegistry, ollama_cache
        from core.schemas import AppSettingsPayload, CustomEngine

        ollama_cache._models = []
        ollama_cache._fetched_at = time.monotonic()

        payload = AppSettingsPayload(
            ollama_base_url='',
            openai_api_key='sk-test',
            openrouter_api_key='sk-test',
            custom_engines=[
                CustomEngine(label='no-key', base_url='https://example.com/v1',
                             api_key='', models=['phantom']),
                CustomEngine(label='has-key', base_url='https://example.com/v1',
                             api_key='sk-real', models=['real-model']),
            ],
        )
        registry = ModelRegistry(app_settings=payload)
        ids = {m.model_id for m in registry.list_models()}
        # The unauthorised engine's model is pruned; the authorised
        # one stays.
        assert 'openai-compatible/no-key/phantom' not in ids
        assert 'openai-compatible/has-key/real-model' in ids

    def test_custom_engine_with_blank_label_is_skipped(self):
        """A custom engine with an empty label registers nothing."""
        from core.registry import ModelRegistry, ollama_cache
        from core.schemas import AppSettingsPayload, CustomEngine

        ollama_cache._models = []
        ollama_cache._fetched_at = time.monotonic()

        payload = AppSettingsPayload(
            ollama_base_url='',
            openai_api_key='sk-test',
            openrouter_api_key='sk-test',
            custom_engines=[
                CustomEngine(label='', base_url='https://example.com/v1',
                             api_key='sk', models=['ghost']),
            ],
        )
        registry = ModelRegistry(app_settings=payload)
        ids = {m.model_id for m in registry.list_models()}
        assert not any(mid.startswith('openai-compatible/') and 'ghost' in mid for mid in ids)


class TestCustomEngineRoundTrip:
    """``AppSettingsPayload.masked() + merge_unmasked`` round-trips engine keys."""

    def test_mask_and_unmask_preserves_engine_key(self):
        from core.schemas import SECRET_MASK, AppSettingsPayload, CustomEngine

        original = AppSettingsPayload(
            custom_engines=[
                CustomEngine(label='mistral', base_url='https://api.mistral.ai/v1',
                             api_key='real-mistral-key', models=['mistral-large']),
            ],
        )
        masked = original.masked()
        assert masked.custom_engines[0].api_key == SECRET_MASK
        # The user submits the masked payload back unchanged.
        merged = original.merge_unmasked(masked)
        assert merged.custom_engines[0].api_key == 'real-mistral-key'

    def test_unmask_with_new_engine_does_not_invent_key(self):
        """Adding a brand-new engine with a real key persists that key."""
        from core.schemas import AppSettingsPayload, CustomEngine

        stored = AppSettingsPayload(custom_engines=[])
        incoming = AppSettingsPayload(custom_engines=[
            CustomEngine(label='new', base_url='u', api_key='new-real-key', models=['m']),
        ])
        merged = stored.merge_unmasked(incoming)
        assert merged.custom_engines[0].api_key == 'new-real-key'

    def test_ollama_prefix_not_duplicated(self):
        """Models passed with explicit 'ollama/' prefix must not be doubled."""
        registry = _make_registry(extra_ollama=["ollama/phi4:latest"])
        ids = [m.model_id for m in registry.list_models()]
        assert ids.count("ollama/phi4:latest") == 1

    def test_duplicate_bootstrap_not_inserted(self):
        registry = _make_registry(extra_ollama=["qwen2.5:14b-instruct"])
        ids = [m.model_id for m in registry.list_models()]
        assert ids.count("ollama/qwen2.5:14b-instruct") == 1

    def test_capability_score_in_range(self):
        from core.registry import ModelRegistry
        registry = ModelRegistry()
        score = registry.capability_score("ollama/qwen2.5:14b-instruct", "coding")
        assert 0.0 <= score <= 1.5

    def test_elo_update_bounded(self):
        from core.registry import ModelRegistry
        registry = ModelRegistry()
        registry.update_elo(
            winners=["ollama/qwen2.5:14b-instruct"],
            losers=["ollama/mistral-small3.1"],
            capabilities={"coding": 0.8, "unknown_capability_xyz": 0.2},
        )
        winner_state = registry.elo_state.get("ollama/qwen2.5:14b-instruct", {})
        assert "unknown_capability_xyz" not in winner_state, \
            "Unknown capabilities must not pollute Elo state"

    def test_elo_winner_score_increases(self):
        from core.registry import ModelRegistry
        registry = ModelRegistry()
        registry.elo_state = {}  # Start fresh to avoid reaching the 1.5 score cap
        before = registry.capability_score("ollama/qwen2.5:14b-instruct", "coding")
        registry.update_elo(
            winners=["ollama/qwen2.5:14b-instruct"],
            losers=["ollama/mistral-small3.1"],
            capabilities={"coding": 1.0},
        )
        after = registry.capability_score("ollama/qwen2.5:14b-instruct", "coding")
        assert after > before, "Winner Elo score should increase after update"

    def test_elo_loser_score_decreases(self):
        from core.registry import ModelRegistry
        registry = ModelRegistry()
        registry.elo_state = {}  # Start fresh to avoid reaching the 0.0 score floor
        before = registry.capability_score("ollama/mistral-small3.1", "coding")
        registry.update_elo(
            winners=["ollama/qwen2.5:14b-instruct"],
            losers=["ollama/mistral-small3.1"],
            capabilities={"coding": 1.0},
        )
        after = registry.capability_score("ollama/mistral-small3.1", "coding")
        assert after < before, "Loser Elo score should decrease after update"


# ---------------------------------------------------------------------------
# Live Ollama discovery cache
# ---------------------------------------------------------------------------

class TestOllamaModelCache:
    def test_cache_returns_empty_when_no_server(self):
        """With no Ollama running, the cache should return [] gracefully."""
        from core.registry import _OllamaModelCache
        cache = _OllamaModelCache()
        cache.configure("http://localhost:19999")  # nothing listening
        models = cache.models
        assert isinstance(models, list)
        assert models == []

    def test_cache_populates_from_real_response(self, ollama_http_server):
        """Cache should populate from a real /api/tags HTTP response."""
        from core.registry import _OllamaModelCache

        ollama_http_server['set_models'](['phi4:latest', 'llama3.2:3b'])

        cache = _OllamaModelCache()
        cache.configure(ollama_http_server['base_url'])
        cache.refresh(force=True)

        assert 'phi4:latest' in cache.models
        assert 'llama3.2:3b' in cache.models

    def test_cache_respects_ttl(self, ollama_http_server):
        """Cache should NOT re-fetch when TTL has not expired."""
        from core.registry import _OllamaModelCache

        ollama_http_server['set_models'](['phi4:latest'])

        cache = _OllamaModelCache()
        cache.configure(ollama_http_server['base_url'])
        cache.refresh(force=True)
        snapshot = list(cache.models)

        # Change the server's response — a fresh fetch would see ['other:7b'].
        ollama_http_server['set_models'](['other:7b'])

        # Refresh without force, well within TTL — must not hit the server.
        cache.refresh(force=False)
        assert cache.models == snapshot, 'Cache must not re-fetch within TTL'

    def test_live_discovered_models_appear_in_registry(self):
        """Live-discovered Ollama models should be injected into the registry candidates."""
        from core.registry import ModelRegistry, ollama_cache

        # Manually seed the cache with fake discovered models.
        ollama_cache._models = ["newmodel:7b", "anothermodel:13b"]
        ollama_cache._fetched_at = time.monotonic()  # fresh — no TTL refresh

        registry = ModelRegistry()

        ids = {m.model_id for m in registry.list_models()}
        assert "ollama/newmodel:7b" in ids
        assert "ollama/anothermodel:13b" in ids

        # Restore cache to avoid polluting other tests.
        ollama_cache._models = []

    def test_live_discovered_does_not_override_bootstrap(self):
        """A live-discovered model that is already in bootstrap keeps its curated priors."""
        from core.registry import ModelRegistry, ollama_cache

        # Seed the cache with a model already in the bootstrap.
        ollama_cache._models = ["qwen2.5:14b-instruct"]
        ollama_cache._fetched_at = time.monotonic()

        registry = ModelRegistry()
        spec = registry.get("ollama/qwen2.5:14b-instruct")

        # The bootstrap coding score is 0.82, not the default 0.65.
        assert spec.capabilities.get("coding", 0) > 0.7, \
            "Bootstrap prior should be preserved over live-discovery default"

        ollama_cache._models = []


# ---------------------------------------------------------------------------
# Storage: atomic writes and conversation CRUD
# ---------------------------------------------------------------------------

def _real_settings(data_dir: Path) -> Settings:
    """Build a real :class:`Settings` instance with ``data_dir`` redirected.

    Only the ``data_dir`` field is needed by :class:`StorageManager` for the
    tests below; everything else inherits its real default (or env value).
    """
    settings = Settings()
    settings.data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return settings


class TestStorageManager:
    """Patch ``core.storage.get_settings`` directly (the in-module binding
    used by ``StorageManager.__init__``). The earlier pattern of patching
    ``core.config.get_settings`` and then ``reload(core.storage)`` leaked
    the lambda permanently into ``core.storage``'s namespace, because the
    monkeypatch only un-patched ``core.config`` while the reload had
    re-bound ``core.storage.get_settings`` independently.
    """

    def test_conversation_create_and_read(self, tmp_path):
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _real_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            convo = mgr.create_conversation(title="Test flight")
            assert convo.title == "Test flight"
            loaded = mgr.get_conversation(convo.conversation_id)
            assert loaded is not None
            assert loaded.conversation_id == convo.conversation_id

    def test_list_conversations_sorted(self, tmp_path):
        """list_conversations must return newest first."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _real_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            mgr.create_conversation(title="First")
            c2 = mgr.create_conversation(title="Second")
            listed = mgr.list_conversations()
            # Newest is created last, so c2 should be first in the list.
            assert listed[0].conversation_id == c2.conversation_id

    def test_cache_disabled_by_default(self, tmp_path):
        """TTL=0 (default) must short-circuit both reads and writes."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _real_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            mgr.set_cache('demo', 'k', {'response': 'v'})
            assert mgr.get_cache('demo', 'k') is None
            # No file written when caching disabled.
            assert not (tmp_path / 'cache' / 'demo.jsonl').exists()

    def test_cache_honours_ttl(self, tmp_path):
        """A fresh entry hits, a stale entry misses."""
        settings = _real_settings(tmp_path)
        settings.provider_cache_ttl_seconds = 60
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: settings)
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            mgr.set_cache('demo', 'k', {'r': 1})
            assert mgr.get_cache('demo', 'k') == {'r': 1}
            # Backdate the on-disk record to 1 hour ago to simulate staleness.
            path = tmp_path / 'cache' / 'demo.jsonl'
            from datetime import datetime, timedelta
            stale = datetime.now(UTC) - timedelta(hours=1)
            lines = path.read_text().splitlines()
            import json as _json
            record = _json.loads(lines[-1])
            record['cached_at'] = stale.isoformat()
            path.write_text(_json.dumps(record) + '\n')
            assert mgr.get_cache('demo', 'k') is None

    def test_cache_forever_when_ttl_negative(self, tmp_path):
        """TTL < 0 keeps the historical 'cache forever' behaviour."""
        settings = _real_settings(tmp_path)
        settings.provider_cache_ttl_seconds = -1
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: settings)
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            mgr.set_cache('demo', 'k', {'r': 1})
            assert mgr.get_cache('demo', 'k') == {'r': 1}

    def test_conversation_path_rejects_traversal(self, tmp_path):
        """Non-UUID conversation ids must be refused so callers cannot escape the data dir."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _real_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            for evil_id in ("../../../etc/passwd", "..", "a/b", ""):
                with pytest.raises(ValueError):
                    mgr.conversation_path(evil_id)
            # get_conversation must fail closed (404-shaped) rather than crash.
            assert mgr.get_conversation("../../../etc/passwd") is None
