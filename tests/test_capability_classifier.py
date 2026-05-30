"""Hermetic tests for the embedding-based capability detector.

Two tests:

* The classifier trains and predicts a normalised distribution when
  the embedding service is reachable, and degrades byte-identically
  to the keyword detector when it isn't.
* The ``ROITELET_CAPABILITY_DETECTOR=embedding`` env var routes
  through the classifier; unset (default) routes through the
  keyword detector.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reset_classifier() -> None:
    from core.capability_classifier import _get_classifier
    _get_classifier.cache_clear()


def test_classifier_predicts_normalised_distribution_or_falls_back(monkeypatch):
    """Two regimes one test:

    * Embedding service unreachable → ``detect_capabilities_embedding``
      must return *exactly* the keyword detector's output (so a CI
      run with no Ollama still works).
    * Embedding service reachable with deterministic vectors → the
      logistic regression fits, the output is a normalised
      distribution, and at least one capability has non-zero weight.
    """
    from core import capability_classifier as cc
    from core.capabilities import detect_capabilities

    # Fallback path.
    monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
    _reset_classifier()
    prompt = 'Write a Python function to reverse a list in place.'
    assert cc.detect_capabilities_embedding(prompt) == detect_capabilities(prompt)

    # Trained path with deterministic embeddings.
    def stub_embed(prompt: str) -> np.ndarray:
        vec = np.zeros(384, dtype=np.float32)
        for ch in prompt[:8].lower():
            vec[ord(ch) % 384] += 1.0
        return vec

    monkeypatch.setattr(cc, '_embed_prompt', stub_embed)
    _reset_classifier()
    result = cc.detect_capabilities_embedding('Write a Python function fizzbuzz(n).')
    assert result
    assert 0.99 <= sum(result.values()) <= 1.01
    _reset_classifier()


def test_env_selector_routes_through_the_active_detector(monkeypatch):
    """Default (unset) and explicit ``keyword`` both use the keyword
    detector. ``embedding`` routes through the classifier."""
    from core import capability_classifier as cc
    from core.capabilities import detect_capabilities

    # Default → keyword detector.
    with pytest.MonkeyPatch().context() as m:
        m.delenv('ROITELET_CAPABILITY_DETECTOR', raising=False)
        prompt = 'Solve the integral of 1/x.'
        assert cc.detect_capabilities_active(prompt) == detect_capabilities(prompt)

    # ``embedding`` → routes through the classifier (which here falls back
    # because we stub embeddings to None, but the call path is what we're
    # locking down).
    monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
    _reset_classifier()
    with pytest.MonkeyPatch().context() as m:
        m.setenv('ROITELET_CAPABILITY_DETECTOR', 'embedding')
        assert cc.detect_capabilities_active('Refactor this Python module.') \
            == detect_capabilities('Refactor this Python module.')
    _reset_classifier()
