"""Hermetic tests for the personal-mode RAG + wiki backend.

The multimodal extractors (whisper / NeMo / kreuzberg / Ollama VLM)
are heavyweight, so we stub them with monkeypatch and only exercise
the parts of ``core.personal`` that don't require them: text files in
the inbox + wiki rendering + size-dependent mode switching.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _reset_singletons() -> None:
    """Drop cached settings so each test sees the tmp_path-rooted state."""
    from core.config import get_settings
    from core.registry import ollama_cache

    get_settings.cache_clear()
    ollama_cache._models = []
    ollama_cache._fetched_at = time.monotonic()


class TestPersonalPaths:
    """The dir helpers must create folders idempotently."""

    def test_creates_personal_tree(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import inbox_dir, personal_root, wiki_dir

        root = personal_root()
        assert root.is_dir()
        assert root.name == 'personal'
        assert inbox_dir().is_dir()
        assert wiki_dir().is_dir()


class TestIngest:
    """Ingestion converts known modalities and skips the rest idempotently."""

    @pytest.mark.asyncio
    async def test_text_file_is_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import ingest_inbox, inbox_dir, wiki_dir

        (inbox_dir() / 'note.md').write_text('# A note\n\nHello world.', encoding='utf-8')
        results = await ingest_inbox()
        assert len(results) == 1
        result = results[0]
        assert result.modality == 'text'
        assert result.error is None
        assert result.wiki_path is not None and result.wiki_path.exists()

        body = result.wiki_path.read_text(encoding='utf-8')
        assert 'Hello world.' in body
        assert 'Auto-converted from' in body  # provenance header
        assert (wiki_dir() / 'note.md').exists()

    @pytest.mark.asyncio
    async def test_unknown_extension_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import ingest_inbox, inbox_dir

        (inbox_dir() / 'weird.xyz').write_text('junk', encoding='utf-8')
        results = await ingest_inbox()
        assert len(results) == 1
        assert results[0].modality == 'skipped'
        assert results[0].wiki_path is None
        assert results[0].error is not None

    @pytest.mark.asyncio
    async def test_ingest_is_idempotent(self, tmp_path, monkeypatch):
        """Running ingest twice must not regenerate already-processed files."""
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import ingest_inbox, inbox_dir

        (inbox_dir() / 'note.md').write_text('# Note\n\nFirst version.', encoding='utf-8')
        first = await ingest_inbox()
        wiki_path = first[0].wiki_path
        assert wiki_path is not None
        mtime_first = wiki_path.stat().st_mtime

        # Mutate the source, but the manifest already recorded it.
        # Without --force, the wiki should NOT be rewritten.
        (inbox_dir() / 'note.md').write_text('# Note\n\nMutated.', encoding='utf-8')
        await ingest_inbox()
        mtime_second = wiki_path.stat().st_mtime
        assert mtime_first == mtime_second

        # With force=True, the wiki must be regenerated.
        await ingest_inbox(force=True)
        mtime_third = wiki_path.stat().st_mtime
        assert mtime_third >= mtime_first
        assert 'Mutated' in wiki_path.read_text(encoding='utf-8')

    @pytest.mark.asyncio
    async def test_audio_extractor_is_stubbed(self, tmp_path, monkeypatch):
        """The audio extractor is invoked by extension classification."""
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()

        import core.personal as personal_mod
        # Stub the heavyweight audio extractor before any classification fires.
        async def fake_transcribe(path):
            return f'[SPEAKER_00] transcript of {path.name}'
        monkeypatch.setattr(personal_mod, '_convert',
                            lambda path, modality: fake_transcribe(path) if modality == 'audio'
                            else _real_convert_stub(path, modality))

        async def _real_convert_stub(path, modality):
            return f'(text from {modality})'

        from core.personal import ingest_inbox, inbox_dir

        (inbox_dir() / 'recording.m4a').write_bytes(b'\x00\x00')  # fake bytes
        results = await ingest_inbox()
        assert len(results) == 1
        assert results[0].modality == 'audio'
        assert results[0].wiki_path is not None
        body = results[0].wiki_path.read_text(encoding='utf-8')
        assert 'transcript of recording.m4a' in body


class TestPersonalContext:
    """``build_personal_context`` picks the right strategy by corpus size."""

    def test_empty_corpus_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import build_personal_context

        assert build_personal_context('anything') == ''

    def test_small_corpus_inlines_everything(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import build_personal_context, wiki_dir

        (wiki_dir() / 'a.md').write_text('# Topic A\n\nFact A is true.', encoding='utf-8')
        (wiki_dir() / 'b.md').write_text('# Topic B\n\nFact B is true.', encoding='utf-8')

        body = build_personal_context('Is fact A true?')
        assert 'From your personal knowledge base' in body
        assert 'Fact A is true.' in body
        assert 'Fact B is true.' in body
        # Wiki mode → no "(top matches)" / "(excerpt)" framing.
        assert 'top matches' not in body
        assert 'excerpt' not in body

    def test_large_corpus_triggers_rag(self, tmp_path, monkeypatch):
        """Above the inline threshold the function should call retrieval.

        We stub the retrieval to short-circuit (no embedding model
        configured in CI), then assert the function returned an empty
        string — the documented behaviour for retrieval failure.
        """
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import build_personal_context, wiki_dir

        # Write more than _WIKI_MAX_INLINE_CHARS (32 000) of text.
        (wiki_dir() / 'big.md').write_text('Lorem ipsum. ' * 4000, encoding='utf-8')

        # Stub the embedding call to always fail → retrieval returns []
        import core.personal as personal_mod
        monkeypatch.setattr(personal_mod, '_retrieve_chunks', lambda prompt, top_k=5: [])
        body = build_personal_context('Find lorem')
        assert body == ''


class TestPersonalStatus:
    """``personal_status`` summarises the corpus for the API + GUI."""

    def test_status_reports_counts_and_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import inbox_dir, personal_status, wiki_dir

        (inbox_dir() / 'pending.pdf').write_bytes(b'%PDF-1.4\n')
        (wiki_dir() / 'topic.md').write_text('Some content.', encoding='utf-8')

        status = personal_status()
        assert status['inbox'] == 1
        assert status['wiki'] == 1
        assert status['mode'] == 'wiki'
        assert status['wiki_chars'] > 0

    def test_empty_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
        _reset_singletons()
        from core.personal import personal_status

        status = personal_status()
        assert status == {'inbox': 0, 'wiki': 0, 'wiki_chars': 0, 'mode': 'empty'}


class TestSlashCommandPersonal:
    """The `/personal` parser branch must set the personal_override flag."""

    def test_personal_command_sets_override(self):
        from core.commands import parse_command

        parsed = parse_command('/personal what did I write about RAG?')
        assert parsed.route_to == 'chat'
        assert parsed.personal_override is True
        assert parsed.stripped_prompt == 'what did I write about RAG?'

    def test_personal_chained_with_local(self):
        from core.commands import parse_command

        parsed = parse_command('/local /personal summarise my wiki')
        assert parsed.route_to == 'chat'
        assert parsed.personal_override is True
        assert parsed.independence_override is True
