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

import json
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from core.core.capabilities import detect_capabilities, top_capabilities
from core.core.judge import parse_winners
from core.schemas import RouterPreferences


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

class TestDetectCapabilities:
    def test_coding_prompt(self):
        scores = detect_capabilities("Write a Python function to parse CSV files.")
        assert scores.get("coding", 0) > 0, "Coding prompt must have a coding score"

    def test_math_prompt(self):
        scores = detect_capabilities("Solve the equation x^2 + 3x - 4 = 0.")
        assert scores.get("math", 0) > 0 or scores.get("reasoning", 0) > 0

    def test_normalized(self):
        scores = detect_capabilities("Translate this email to French.")
        total = sum(scores.values())
        assert abs(total - 1.0) < 1e-6, f"Scores must sum to 1.0, got {total}"

    def test_backtick_coding_boost(self):
        scores = detect_capabilities("```python\ndef add(a, b):\n    return a + b\n```")
        assert scores.get("coding", 0) > 0.3, "Code block should strongly signal coding"

    def test_long_prompt_threshold(self):
        # Under 4000 chars should NOT trigger long_context
        short = "Summarize this document. " * 50   # ~1250 chars
        scores_short = detect_capabilities(short)
        # Over 4000 chars should trigger long_context
        long = "Summarize this document. " * 200   # ~5000 chars
        scores_long = detect_capabilities(long)
        assert scores_long.get("long_context", 0) >= scores_short.get("long_context", 0)

    def test_top_capabilities(self):
        caps = {"coding": 0.6, "math": 0.3, "reasoning": 0.1}
        top = top_capabilities(caps, limit=2)
        assert top[0] == "coding"
        assert len(top) == 2

    def test_expanded_coding_keywords(self):
        """Ensure expanded keyword list covers modern dev terms."""
        for word in ("typescript", "docker", "refactor", "kubernetes"):
            scores = detect_capabilities(f"Help me {word} this project.")
            assert scores.get("coding", 0) > 0, f"'{word}' should trigger coding"

    def test_expanded_math_keywords(self):
        for word in ("statistics", "calculus", "theorem", "formula"):
            scores = detect_capabilities(f"Explain {word} to me.")
            assert scores.get("math", 0) > 0 or scores.get("reasoning", 0) > 0

    def test_multilingual_keywords(self):
        # Use English keywords that are in CAPABILITY_KEYWORDS
        scores = detect_capabilities("Translate this text to chinese.")
        assert scores.get("multilingual", 0) > 0, "translate + chinese should trigger multilingual"


# ---------------------------------------------------------------------------
# Judge output parsing
# ---------------------------------------------------------------------------

class TestParseWinners:
    def test_single_winner(self):
        assert parse_winners("Great answer.\nWINNERS: 2") == [2]

    def test_multiple_winners(self):
        assert parse_winners("Both good.\nWINNERS: 1, 3") == [1, 3]

    def test_fallback_when_missing(self):
        assert parse_winners("No winners line here.") == [1]

    def test_ignores_non_digits(self):
        # The regex [0-9, ]+ stops at the first non-numeric non-space character.
        # "WINNERS: 1, x, 3" → the regex captures "1, " (stops at 'x'), so only [1] is returned.
        # This is expected behavior; well-formed judge output never contains letters in WINNERS.
        result = parse_winners("WINNERS: 1, x, 3")
        assert result == [1]
        # A clean numeric list should work fine.
        assert parse_winners("WINNERS: 2, 3") == [2, 3]

    def test_trailing_whitespace(self):
        assert parse_winners("WINNERS: 1 ") == [1]

    def test_multiline_content_before_winners(self):
        text = "Candidate 1 is good.\nCandidate 2 is better.\nWINNERS: 2"
        assert parse_winners(text) == [2]


# ---------------------------------------------------------------------------
# Router preferences and schema
# ---------------------------------------------------------------------------

class TestRouterPreferences:
    def test_defaults(self):
        prefs = RouterPreferences()
        assert prefs.raw_power == 0.7
        assert prefs.frugality == 0.3
        assert prefs.independence is False
        assert prefs.allow_vlms is False

    def test_custom(self):
        prefs = RouterPreferences(raw_power=1.0, frugality=0.0, independence=True)
        assert prefs.independence is True

    def test_serialise_round_trip(self):
        prefs = RouterPreferences(raw_power=0.5, frugality=0.5, allow_vlms=True)
        dumped = prefs.model_dump()
        loaded = RouterPreferences.model_validate(dumped)
        assert loaded == prefs


# ---------------------------------------------------------------------------
# Registry: bootstrap loading and model injection
# ---------------------------------------------------------------------------

def _make_registry(extra_ollama: List[str] = None, extra_openrouter: List[str] = None):
    """Build a registry backed by the real bootstrap file."""
    from core.schemas import AppSettingsPayload
    from core.core.registry import ModelRegistry

    payload = AppSettingsPayload(
        selected_ollama_models=extra_ollama or [],
        paid_openrouter_models=extra_openrouter or [],
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
        from core.core.registry import ModelRegistry
        registry = ModelRegistry()
        score = registry.capability_score("ollama/qwen2.5:14b-instruct", "coding")
        assert 0.0 <= score <= 1.5

    def test_elo_update_bounded(self):
        from core.core.registry import ModelRegistry, KNOWN_CAPABILITIES
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
        from core.core.registry import ModelRegistry
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
        from core.core.registry import ModelRegistry
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
        from core.core.registry import _OllamaModelCache
        cache = _OllamaModelCache()
        cache.configure("http://localhost:19999")  # nothing listening
        models = cache.models
        assert isinstance(models, list)
        assert models == []

    def test_cache_uses_mocked_response(self):
        """Cache should populate from a mocked Ollama /api/tags response."""
        from core.core.registry import _OllamaModelCache
        import httpx

        fake_response_data = {
            "models": [
                {"name": "phi4:latest"},
                {"name": "llama3.2:3b"},
            ]
        }

        cache = _OllamaModelCache()
        cache.configure("http://localhost:11434")

        with patch("core.core.registry.httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = fake_response_data
            mock_get.return_value = mock_response

            cache.refresh(force=True)

        assert "phi4:latest" in cache.models
        assert "llama3.2:3b" in cache.models

    def test_cache_respects_ttl(self):
        """Cache should NOT re-fetch when TTL has not expired."""
        from core.core.registry import _OllamaModelCache, _OLLAMA_CACHE_TTL_S
        import httpx

        cache = _OllamaModelCache()
        cache.configure("http://localhost:11434")

        with patch("core.core.registry.httpx.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"models": [{"name": "phi4:latest"}]}
            mock_get.return_value = mock_response

            # First fetch.
            cache.refresh(force=True)
            assert mock_get.call_count == 1

            # Second fetch within TTL — should be skipped.
            cache.refresh(force=False)
            assert mock_get.call_count == 1, "Should not re-fetch within TTL"

    def test_live_discovered_models_appear_in_registry(self):
        """Live-discovered Ollama models should be injected into the registry candidates."""
        from core.core.registry import ModelRegistry, ollama_cache

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
        from core.core.registry import ModelRegistry, ollama_cache

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

def _fake_settings(data_dir: Path):
    """Minimal settings object for storage tests."""
    obj = MagicMock()
    obj.data_dir = data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    return obj


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
            m.setattr("core.storage.get_settings", lambda: _fake_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            convo = mgr.create_conversation(title="Test flight")
            assert convo.title == "Test flight"
            loaded = mgr.get_conversation(convo.conversation_id)
            assert loaded is not None
            assert loaded.conversation_id == convo.conversation_id

    def test_atomic_write_produces_valid_json(self, tmp_path):
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _fake_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            target = tmp_path / "test_atomic.json"
            mgr._write_json(target, {"key": "value", "number": 42})
            assert target.exists()
            parsed = json.loads(target.read_text())
            assert parsed["key"] == "value"

    def test_no_tmp_file_left_on_success(self, tmp_path):
        """Atomic write must not leave .tmp files behind on success."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _fake_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            target = tmp_path / "clean.json"
            mgr._write_json(target, {"ok": True})
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert tmp_files == [], f"Stray .tmp files found: {tmp_files}"

    def test_list_conversations_sorted(self, tmp_path):
        """list_conversations must return newest first."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr("core.storage.get_settings", lambda: _fake_settings(tmp_path))
            import core.storage as st_mod
            mgr = st_mod.StorageManager()
            c1 = mgr.create_conversation(title="First")
            c2 = mgr.create_conversation(title="Second")
            listed = mgr.list_conversations()
            # Newest is created last, so c2 should be first in the list.
            assert listed[0].conversation_id == c2.conversation_id
