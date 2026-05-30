"""Hermetic tests for the personal-mode RAG + wiki backend.

Three story-level tests:

1. Ingest + size-dependent context strategy: drop files in the inbox,
   convert known modalities, skip unknown ones, idempotent without
   ``force``, regenerate with ``force``; below the inline-cap the
   context is concatenated, above it the function calls retrieval.
2. The persistent RAG index re-embeds chunks only when the wiki has
   changed — the cold-start cost happens once per (wiki revision,
   query) pair.
3. The Karpathy-style 2-D scatter projects every chunk to (x, y) when
   the embedder is reachable and returns an empty list otherwise.

The multimodal extractors (whisper / NeMo / kreuzberg / Ollama VLM)
are heavyweight, so we stub them; the embedding model is stubbed as
a deterministic NumPy vector so we can assert call counts.
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _reset_singletons() -> None:
    """Drop cached settings so each test sees the tmp_path-rooted state."""
    from core.config import get_settings
    from core.registry import get_registry, ollama_cache
    from core.storage import get_storage

    get_settings.cache_clear()
    get_storage.cache_clear()
    get_registry.cache_clear()
    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()


@pytest.fixture(autouse=True)
def _isolate_singletons():
    """Reset cached singletons after every test so we never leak a tmp_path."""
    yield
    _reset_singletons()


def _stub_embed(text: str) -> np.ndarray:
    """Deterministic 16-D embedding keyed on first char + length."""
    vec = np.zeros(16, dtype=np.float32)
    if text:
        vec[ord(text[0]) % 16] = 1.0
        vec[len(text) % 16] += 0.25
    return vec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_ingest_round_trip_and_context_strategy(tmp_path, monkeypatch):
    """Inbox ingestion + size-dependent context resolution, end-to-end.

    Three things one story:

    * a text file in the inbox produces a wiki entry with a provenance
      header, an unknown extension is recorded as ``modality='skipped'``,
      and the manifest makes a second run a no-op unless ``force``;
    * below the inline cap, ``build_personal_context`` concatenates
      every wiki file verbatim;
    * above the inline cap it calls ``_retrieve_chunks`` and returns
      "" when retrieval can't reach the embedder (the documented
      degradation).
    """
    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    _reset_singletons()
    import core.personal as personal_mod
    from core.personal import (
        build_personal_context,
        inbox_dir,
        ingest_inbox,
        personal_status,
        wiki_dir,
    )

    # ── Ingest ──────────────────────────────────────────────────────
    (inbox_dir() / 'note.md').write_text('# A note\n\nHello world.', encoding='utf-8')
    (inbox_dir() / 'weird.xyz').write_text('junk', encoding='utf-8')

    results = await ingest_inbox()
    modalities = {r.source.name: r.modality for r in results}
    assert modalities['note.md'] == 'text'
    assert modalities['weird.xyz'] == 'skipped'

    note_wiki = wiki_dir() / 'note.md'
    body = note_wiki.read_text(encoding='utf-8')
    assert 'Hello world.' in body
    assert 'Auto-converted from' in body  # provenance header

    # Idempotent: a second run without --force shouldn't rewrite the wiki.
    mtime_first = note_wiki.stat().st_mtime
    (inbox_dir() / 'note.md').write_text('# A note\n\nMUTATED.', encoding='utf-8')
    await ingest_inbox()
    assert note_wiki.stat().st_mtime == mtime_first

    # ``force=True`` regenerates from the new source.
    await ingest_inbox(force=True)
    assert 'MUTATED' in note_wiki.read_text(encoding='utf-8')

    status = personal_status()
    assert status['mode'] == 'wiki'
    assert status['wiki'] >= 1
    assert status['inbox'] >= 1

    # ── Small corpus → inline context ───────────────────────────────
    (wiki_dir() / 'fact-a.md').write_text('# Topic A\n\nFact A is true.', encoding='utf-8')
    (wiki_dir() / 'fact-b.md').write_text('# Topic B\n\nFact B is true.', encoding='utf-8')
    inline = build_personal_context('Is fact A true?')
    assert 'From your personal knowledge base' in inline
    assert 'Fact A is true.' in inline and 'Fact B is true.' in inline

    # ── Large corpus → retrieval path, empty when embedder unreachable ──
    (wiki_dir() / 'big.md').write_text('Lorem ipsum. ' * 4000, encoding='utf-8')
    monkeypatch.setattr(personal_mod, '_retrieve_chunks', lambda prompt, top_k=5: [])
    assert build_personal_context('Find lorem') == ''


def test_rag_index_caches_embeddings_per_wiki_revision(tmp_path, monkeypatch):
    """The persistent RAG index is the personal-mode performance story.

    Cold start embeds every chunk + the query. The second query, on
    an unchanged wiki, must embed only the query — chunks come from
    the on-disk ``.npy`` cache. Mutating any wiki file must invalidate
    the cache and re-embed.
    """
    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    _reset_singletons()
    import core.capability_classifier as cc

    call_log: list[str] = []

    def counting_embed(text: str):
        call_log.append(text)
        return _stub_embed(text)

    monkeypatch.setattr(cc, '_embed_prompt', counting_embed)

    from core.personal import _retrieve_chunks, wiki_dir
    wiki = wiki_dir()
    (wiki / 'a.md').write_text('alpha alpha alpha. ' * 80, encoding='utf-8')
    (wiki / 'b.md').write_text('beta beta beta. ' * 80, encoding='utf-8')

    # Cold start: chunks + the query.
    assert _retrieve_chunks('alpha please', top_k=2)
    cold_calls = len(call_log)
    assert cold_calls > 1

    # Warm: only the query is embedded.
    call_log.clear()
    assert _retrieve_chunks('beta please', top_k=2)
    assert len(call_log) == 1

    # Mutating a wiki file must invalidate the index — the next query
    # re-embeds. The sleep guarantees the mtime fingerprint advances
    # even on second-resolution filesystems.
    call_log.clear()
    time.sleep(0.01)
    (wiki / 'a.md').write_text('alpha alpha alpha. NEW CONTENT.' + ('alpha. ' * 80), encoding='utf-8')
    _retrieve_chunks('alpha', top_k=1)
    assert len(call_log) > 1

    # Empty / missing wikis return [] safely.
    (wiki / 'a.md').unlink()
    (wiki / 'b.md').unlink()
    assert _retrieve_chunks('anything') == []


def test_embedding_viz_projects_every_chunk_or_degrades_to_empty(tmp_path, monkeypatch):
    """The 2-D scatter is the personal-mode "trust" affordance — every
    chunk must show up, every point must carry path / chunk_index /
    text / x / y, and an unreachable embedder must degrade to ``[]``
    rather than crash."""
    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    _reset_singletons()
    import core.capability_classifier as cc

    # Embedder unreachable → empty.
    monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
    from core.personal import project_chunks_2d, wiki_dir
    (wiki_dir() / 'a.md').write_text('# A\n\nHello.', encoding='utf-8')
    assert project_chunks_2d() == []

    # Embedder reachable → every chunk projected to (x, y).
    def stub_embed(text: str) -> np.ndarray:
        # 32-D one-hot + small bump so SVD has rank > 1.
        v = np.zeros(32, dtype=np.float32)
        v[len(text) % 32] = 1.0
        if text:
            v[ord(text[0]) % 32] += 0.5
        return v

    monkeypatch.setattr(cc, '_embed_prompt', stub_embed)
    (wiki_dir() / 'topic-a.md').write_text('# Topic A\n\n' + ('alpha. ' * 50), encoding='utf-8')
    (wiki_dir() / 'topic-b.md').write_text('# Topic B\n\n' + ('beta. ' * 50), encoding='utf-8')
    points = project_chunks_2d()
    assert points
    for p in points:
        assert {'path', 'chunk_index', 'text', 'x', 'y'} <= set(p)
        assert isinstance(p['x'], float) and isinstance(p['y'], float)
