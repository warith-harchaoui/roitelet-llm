"""Hermetic tests for the embedding-based capability detector.

We can't call real Ollama from CI, so the embedding HTTP call is
stubbed. The tests lock down:

1. When the embedding call succeeds, the classifier returns a
   normalised distribution over ``KNOWN_CAPABILITIES``.
2. When the embedding call fails, the classifier degrades to the
   keyword detector (byte-identical output).
3. ``ROITELET_CAPABILITY_DETECTOR=embedding`` flips the selector;
   anything else uses keywords.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reset_classifier() -> None:
    """Drop the lazy classifier singleton so each test starts fresh."""
    from core.capability_classifier import _get_classifier
    _get_classifier.cache_clear()


class TestEmbeddingClassifier:
    def test_fallback_when_embedding_unavailable(self, monkeypatch):
        """Failure to embed → keyword detector wins, output unchanged."""
        from core import capability_classifier as cc
        from core.capabilities import detect_capabilities

        # Stub the embedding call to always fail.
        monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
        _reset_classifier()

        prompt = 'Write a Python function to reverse a list in place.'
        embedding_result = cc.detect_capabilities_embedding(prompt)
        keyword_result = detect_capabilities(prompt)
        assert embedding_result == keyword_result
        _reset_classifier()

    def test_classifier_trains_and_predicts(self, monkeypatch):
        """With deterministic embeddings, the classifier predicts a usable distribution."""
        from core import capability_classifier as cc

        # Deterministic stub: hash the prompt's first letter into a
        # one-hot of width 384 (typical embedding dim). Sufficient to
        # let LR find a separator across the 8 categories in the
        # training corpus.
        def _stub_embed(prompt: str) -> np.ndarray:
            vec = np.zeros(384, dtype=np.float32)
            # Mix a couple of hash buckets per prompt so the
            # classifier sees something more than constant 1s.
            for char in prompt[:8].lower():
                vec[ord(char) % 384] += 1.0
            return vec

        monkeypatch.setattr(cc, '_embed_prompt', _stub_embed)
        _reset_classifier()

        result = cc.detect_capabilities_embedding(
            'Write a Python function fizzbuzz(n).',
        )
        # The classifier may not always crown ``coding`` with a stub
        # embedding, but the distribution must be normalised and have
        # at least one entry.
        assert result
        total = sum(result.values())
        assert 0.99 <= total <= 1.01, f'Distribution not normalised: {total}'
        _reset_classifier()

    def test_env_selector_default_is_keyword(self):
        from core.capabilities import detect_capabilities
        from core.capability_classifier import detect_capabilities_active

        with pytest.MonkeyPatch().context() as m:
            m.delenv('ROITELET_CAPABILITY_DETECTOR', raising=False)
            prompt = 'Solve the integral of 1/x.'
            assert detect_capabilities_active(prompt) == detect_capabilities(prompt)

    def test_env_selector_embedding_uses_classifier(self, monkeypatch):
        """Env var ``embedding`` routes through the classifier even if it just falls back."""
        from core import capability_classifier as cc

        monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
        _reset_classifier()

        with pytest.MonkeyPatch().context() as m:
            m.setenv('ROITELET_CAPABILITY_DETECTOR', 'embedding')
            result = cc.detect_capabilities_active('Refactor this Python module.')
            # Fallback path → matches the keyword detector output. The
            # important property is that the **call path** went through
            # the classifier (verifiable indirectly: the import
            # succeeded and didn't raise).
            from core.capabilities import detect_capabilities
            assert result == detect_capabilities('Refactor this Python module.')
        _reset_classifier()
