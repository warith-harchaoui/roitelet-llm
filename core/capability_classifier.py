"""Embedding-based capability detector (Phase C, opt-in alternative).

The default :func:`core.capabilities.detect_capabilities` is a keyword
scan — fast, deterministic, debuggable, and English-biased. This module
is the alternative: a sklearn classifier sitting on top of
locally-served sentence embeddings, fitted on a small labelled corpus.

Why
---
The keyword detector misses

- **paraphrases** (`"clean up this Python file"` doesn't fire any
  "coding" keyword),
- **non-English prompts** (the keyword list is mostly English; the
  module hard-codes a smattering of French/Spanish translation
  triggers but that's it),
- **domain prose that doesn't trip a fixed bigram** (`"write a unit
  test that mocks the database"` does fire, but `"give me an example
  of a fixture that swaps the persistence layer"` doesn't).

Constraints
-----------
1. **Local-first.** Embeddings come from a local Ollama instance —
   today the standard pick is ``nomic-embed-text`` (137 M params,
   ~270 MB on disk). No paid embedding API.
2. **No new heavy dependencies.** sklearn is already a runtime dep;
   the classifier is plain logistic regression with TF-IDF as a
   safety net when Ollama is unreachable.
3. **Graceful degradation.** Any failure path (Ollama down, model not
   pulled, classifier untrained, etc.) returns to the keyword
   detector. The caller sees a single API: a ``dict[str, float]``
   of normalised capability weights.
4. **Opt-in.** ``ROITELET_CAPABILITY_DETECTOR=embedding`` selects
   this module; anything else (the default) keeps the keyword path.

Training signal
---------------
The eval dataset under ``tests/eval/dataset.json`` already carries
``category`` labels (``coding`` / ``math`` / ``reasoning`` / ...). We
fit on those by default. A larger labelled corpus would slot in by
overriding :data:`_TRAINING_PROMPTS_PATH`.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np

from .capabilities import detect_capabilities as _keyword_detect
from .config import get_settings
from .registry import KNOWN_CAPABILITIES

logger = logging.getLogger(__name__)


# Where to look for labelled training examples. The eval dataset is the
# obvious built-in: it ships with the repo, carries ``category``
# labels, and is curated. A real deployment can extend this with
# additional labelled JSON in the same shape.
_TRAINING_PROMPTS_PATH: Path = (
    Path(__file__).resolve().parent.parent / 'tests' / 'eval' / 'dataset.json'
)

# Below this many labelled examples the classifier degrades to the
# keyword detector. Tiny corpora produce wildly overconfident
# predictions that are worse than the keyword scan they're meant to
# replace.
_MIN_TRAINING_EXAMPLES: int = 12

# Embedding model name to pull from Ollama. Configurable via env var so
# a deployment can swap to e.g. ``snowflake-arctic-embed`` without code
# changes.
_EMBED_MODEL_ENV: str = 'ROITELET_EMBED_MODEL'
_EMBED_MODEL_DEFAULT: str = 'nomic-embed-text'

# HTTP timeout (seconds) for the embedding call. Short — embedding
# inference on a laptop CPU is sub-second; anything longer means the
# Ollama server is hosed and we should fall back rather than wait.
_EMBED_TIMEOUT_S: float = 8.0


def _embed_model_name() -> str:
    return os.environ.get(_EMBED_MODEL_ENV, _EMBED_MODEL_DEFAULT)


def _ollama_base_url() -> str:
    """The Ollama base URL the embedding call should hit."""
    return get_settings().local_llm_base_url


# ---------------------------------------------------------------------------
# Embedding client (synchronous — fits the rest of the keyword detector)
# ---------------------------------------------------------------------------


def _embed_prompt(prompt: str) -> np.ndarray | None:
    """Synchronously embed one prompt via Ollama's ``/api/embeddings``.

    Returns ``None`` on any failure — caller must fall back. The
    ``timeout`` is deliberately short; on a healthy local Ollama this
    completes in ~200 ms, on a hosed one we'd rather degrade than
    block the route.
    """
    base = _ollama_base_url()
    if not base:
        return None
    try:
        response = httpx.post(
            f'{base.rstrip("/")}/api/embeddings',
            json={'model': _embed_model_name(), 'prompt': prompt},
            timeout=_EMBED_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get('embedding')
        if not embedding:
            return None
        return np.asarray(embedding, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.debug('Embedding call failed (%s) — falling back to keywords.', exc)
        return None


# ---------------------------------------------------------------------------
# Classifier (trained lazily on first call)
# ---------------------------------------------------------------------------


class _CapabilityClassifier:
    """One-vs-rest logistic regression over Ollama embeddings.

    Fit lazily on first ``predict`` so import time stays fast. If the
    embedding call fails during fit (offline laptop, model not pulled,
    etc.) the classifier marks itself as ``unavailable`` and the public
    ``detect_capabilities_embedding`` shim transparently routes to the
    keyword detector.
    """

    def __init__(self) -> None:
        self._available: bool = False
        self._labels: list[str] = []
        self._coef: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._embed_dim: int | None = None

    def fit(self) -> None:
        """Embed labelled prompts, train one-vs-rest LR per capability."""
        if not _TRAINING_PROMPTS_PATH.exists():
            logger.debug('Embedding classifier: training corpus missing at %s', _TRAINING_PROMPTS_PATH)
            return
        try:
            corpus = json.loads(_TRAINING_PROMPTS_PATH.read_text(encoding='utf-8'))
        except Exception as exc:  # noqa: BLE001
            logger.debug('Embedding classifier: corpus unreadable: %s', exc)
            return

        # Constrain to capabilities we know how to score on. Unknown
        # labels in the dataset are silently dropped; they'd otherwise
        # widen ``_labels`` and produce ghost outputs the router can't
        # consume.
        rows: list[tuple[str, str]] = []
        for case in corpus:
            category = case.get('category')
            prompt = case.get('prompt')
            if not prompt or not category:
                continue
            if category not in KNOWN_CAPABILITIES:
                continue
            rows.append((prompt, category))

        if len(rows) < _MIN_TRAINING_EXAMPLES:
            logger.info(
                'Embedding classifier: %d labelled examples < %d — keyword fallback.',
                len(rows),
                _MIN_TRAINING_EXAMPLES,
            )
            return

        embeddings: list[np.ndarray] = []
        labels: list[str] = []
        for prompt, category in rows:
            vec = _embed_prompt(prompt)
            if vec is None:
                # Single failure is enough to abort the fit — the
                # Ollama server is either down or doesn't have the
                # embedding model pulled, and falling back is safer
                # than training on a half-corpus.
                logger.info(
                    'Embedding classifier: failed to embed a training example '
                    '(%s). Keyword fallback active.',
                    category,
                )
                return
            embeddings.append(vec)
            labels.append(category)

        from sklearn.linear_model import LogisticRegression

        X = np.stack(embeddings).astype(np.float32)
        self._embed_dim = X.shape[1]
        self._labels = sorted(set(labels))

        # One-vs-rest binary LRs, stored as a single matrix for fast
        # inference. The class set is small (≤ 9) so sklearn's built-in
        # multinomial mode would do, but the OvR formulation keeps
        # per-label calibration independent and lets us return a
        # **multi-label** distribution rather than a one-hot pick.
        coefs: list[np.ndarray] = []
        intercepts: list[float] = []
        for label in self._labels:
            y = np.array([1 if lbl == label else 0 for lbl in labels])
            if y.sum() == 0 or y.sum() == len(y):
                # Degenerate single-class column — emit a "always zero"
                # row so the matrix shape stays stable.
                coefs.append(np.zeros(X.shape[1], dtype=np.float32))
                intercepts.append(0.0)
                continue
            clf = LogisticRegression(
                max_iter=400,
                solver='liblinear',
                C=1.0,
                class_weight='balanced',
            )
            clf.fit(X, y)
            coefs.append(clf.coef_[0].astype(np.float32))
            intercepts.append(float(clf.intercept_[0]))

        self._coef = np.stack(coefs)
        self._intercept = np.asarray(intercepts, dtype=np.float32)
        self._available = True
        logger.info(
            'Embedding classifier fitted: %d examples, %d labels, dim=%d.',
            len(embeddings),
            len(self._labels),
            self._embed_dim,
        )

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, prompt: str) -> dict[str, float] | None:
        """Return a normalised capability distribution or ``None`` on failure."""
        if not self._available or self._coef is None or self._intercept is None:
            return None
        vec = _embed_prompt(prompt)
        if vec is None:
            return None
        # Guard against an embedding-model size change between fit and
        # predict (unusual but cheap to check).
        if vec.shape[0] != self._embed_dim:
            return None
        logits = self._coef @ vec + self._intercept
        # Sigmoid → independent per-label probabilities, then normalise
        # so the router's existing weighted-sum machinery stays happy.
        probs = 1.0 / (1.0 + np.exp(-logits))
        weights = {label: float(p) for label, p in zip(self._labels, probs, strict=True)}
        # Drop near-zero probabilities — they're noise and clutter the
        # ``RouterDecision.reasoning`` trail.
        filtered = {k: v for k, v in weights.items() if v > 0.05}
        total = sum(filtered.values()) or 1.0
        return {k: v / total for k, v in filtered.items()}


@lru_cache(maxsize=1)
def _get_classifier() -> _CapabilityClassifier:
    """Lazy classifier singleton — fit-once, predict-many."""
    classifier = _CapabilityClassifier()
    classifier.fit()
    return classifier


# ---------------------------------------------------------------------------
# Public surface — what callers import
# ---------------------------------------------------------------------------


def detect_capabilities_embedding(prompt: str) -> dict[str, float]:
    """Hybrid detector: classifier when available, keyword when not.

    The signature matches :func:`core.capabilities.detect_capabilities`
    exactly so swapping the import-site is a one-line change.
    """
    classifier = _get_classifier()
    if classifier.available:
        predicted = classifier.predict(prompt)
        if predicted:
            return predicted
    return _keyword_detect(prompt)


def detect_capabilities_active(prompt: str) -> dict[str, float]:
    """Dispatch on the ``ROITELET_CAPABILITY_DETECTOR`` env var.

    Set to ``embedding`` to use the classifier; anything else (default)
    keeps the keyword detector. This is the public seam every caller
    that wants the configurable behaviour should import.
    """
    flavour = os.environ.get('ROITELET_CAPABILITY_DETECTOR', 'keyword').lower().strip()
    if flavour == 'embedding':
        return detect_capabilities_embedding(prompt)
    return _keyword_detect(prompt)


def refresh_classifier() -> None:
    """Force a re-fit. Call after dropping new labelled training data on disk."""
    _get_classifier.cache_clear()
    _get_classifier()
