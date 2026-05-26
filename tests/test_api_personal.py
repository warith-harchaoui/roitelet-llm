"""API integration tests for the Personal-mode endpoints.

Exercises the FastAPI surface — ``/api/personal``, ``/api/personal/ingest``,
``/api/personal/embeddings`` — plus the ``/personal`` slash-command branch
of ``/api/chat``. The pipeline + multimodal extractors are stubbed so
the tests are hermetic (no Ollama, no kreuzberg).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


def _reset_singletons() -> None:
    """Reset cached config + storage so each test sees the tmp_path data dir."""
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
    """Tear-down singletons after every test so tmp_path leakage can't accumulate."""
    yield
    _reset_singletons()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient bound to a fresh tmp data dir per test."""
    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    _reset_singletons()
    # Import the app *after* the env var lands so the lifespan + module
    # globals pick up the tmp data dir.
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestPersonalStatusEndpoint:
    """``GET /api/personal`` reports counts and mode."""

    def test_empty_returns_empty_mode(self, client):
        response = client.get('/api/personal')
        assert response.status_code == 200
        body = response.json()
        assert body == {'inbox': 0, 'wiki': 0, 'wiki_chars': 0, 'mode': 'empty'}

    def test_counts_reflect_disk(self, client, tmp_path):
        # Drop a wiki file directly; status should reflect it.
        wiki = tmp_path / 'personal' / 'wiki'
        wiki.mkdir(parents=True, exist_ok=True)
        (wiki / 'note.md').write_text('# Hello\n\nbody', encoding='utf-8')

        response = client.get('/api/personal')
        body = response.json()
        assert body['wiki'] == 1
        assert body['mode'] == 'wiki'
        assert body['wiki_chars'] > 0


class TestPersonalIngestEndpoint:
    """``POST /api/personal/ingest`` runs the inbox-to-wiki conversion."""

    def test_text_files_get_ingested(self, client, tmp_path):
        inbox = tmp_path / 'personal' / 'inbox'
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / 'note.md').write_text('# Topic\n\nA note.', encoding='utf-8')

        response = client.post('/api/personal/ingest')
        assert response.status_code == 200
        body = response.json()
        assert len(body['results']) == 1
        assert body['results'][0]['modality'] == 'text'
        assert body['results'][0]['error'] is None
        assert body['status']['wiki'] == 1

    def test_unknown_extension_returns_error_field(self, client, tmp_path):
        inbox = tmp_path / 'personal' / 'inbox'
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / 'weird.xyz').write_text('junk', encoding='utf-8')

        response = client.post('/api/personal/ingest')
        body = response.json()
        assert body['results'][0]['modality'] == 'skipped'
        assert body['results'][0]['error']
        # Status still reports 0 wiki entries because nothing was converted.
        assert body['status']['wiki'] == 0


class TestPersonalEmbeddingsEndpoint:
    """``GET /api/personal/embeddings`` returns the 2-D scatter."""

    def test_empty_wiki_returns_empty_points(self, client):
        response = client.get('/api/personal/embeddings')
        assert response.status_code == 200
        assert response.json() == {'points': []}

    def test_embedding_failure_returns_empty_points(self, client, monkeypatch, tmp_path):
        """When the embedding model is unreachable, return [] cleanly."""
        wiki = tmp_path / 'personal' / 'wiki'
        wiki.mkdir(parents=True, exist_ok=True)
        (wiki / 'a.md').write_text('# A\n\nHello.', encoding='utf-8')

        import core.capability_classifier as cc
        monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)

        response = client.get('/api/personal/embeddings')
        assert response.json() == {'points': []}

    def test_returns_coordinates_when_embedding_works(self, client, monkeypatch, tmp_path):
        """Stub the embedder, write wiki content, expect coordinates."""
        import numpy as np

        wiki = tmp_path / 'personal' / 'wiki'
        wiki.mkdir(parents=True, exist_ok=True)
        (wiki / 'a.md').write_text('# A\n\n' + ('alpha. ' * 50), encoding='utf-8')
        (wiki / 'b.md').write_text('# B\n\n' + ('beta. ' * 50), encoding='utf-8')

        def _stub_embed(text):
            vec = np.zeros(32, dtype=np.float32)
            vec[len(text) % 32] = 1.0
            if text:
                vec[ord(text[0]) % 32] += 0.5
            return vec

        import core.capability_classifier as cc
        monkeypatch.setattr(cc, '_embed_prompt', _stub_embed)

        response = client.get('/api/personal/embeddings')
        body = response.json()
        assert body['points']
        for p in body['points']:
            assert {'path', 'chunk_index', 'text', 'x', 'y'} <= set(p)


class TestPersonalSlashCommand:
    """``/personal`` in a chat prompt prepends the wiki to the prompt."""

    def test_personal_prefix_injects_wiki_context(self, client, monkeypatch, tmp_path):
        """End-to-end: write a wiki file, send /personal, verify the
        pipeline received an augmented prompt with the wiki content.
        """
        # Drop a wiki entry so build_personal_context returns a block.
        wiki = tmp_path / 'personal' / 'wiki'
        wiki.mkdir(parents=True, exist_ok=True)
        (wiki / 'fact.md').write_text(
            '# Important\n\nThe wren weighs 9 grams.', encoding='utf-8'
        )

        # Stub the pipeline to echo whatever prompt it received so we
        # can assert on the augmented value.
        captured: dict = {}

        async def _stub_run(payload, router=None):
            from core.schemas import (
                ChatResponse,
                ModelResponse,
                RouterDecision,
                SynthesisResult,
            )
            captured['prompt'] = payload.prompt
            return ChatResponse(
                conversation_id='conv',
                router=RouterDecision(
                    prompt=payload.prompt, categories={},
                    candidates=[], selected_model_ids=[], reasoning=[],
                ),
                responses=[ModelResponse(model_id='m', provider='p',
                                          content='ok', latency_s=0.1)],
                synthesis=SynthesisResult(model_id='m', provider='p',
                                           content='stub',
                                           judge_summary='', winning_model_ids=[]),
                telemetry_id='tel',
            )

        monkeypatch.setattr('api.main.run_roitelet_chat', _stub_run)

        response = client.post(
            '/api/chat',
            json={'prompt': '/personal How much does a wren weigh?'},
        )
        assert response.status_code == 200, response.text
        # The pipeline saw a prompt with the wiki content + the user
        # question. The slash command itself is stripped.
        assert 'How much does a wren weigh?' in captured['prompt']
        assert 'wren weighs 9 grams' in captured['prompt']
        assert '/personal' not in captured['prompt'][:40]  # slash stripped at the head

    def test_personal_without_wiki_still_runs_chat(self, client, monkeypatch):
        """An empty knowledge base must not break the slash command."""
        called: list = []

        async def _stub_run(payload, router=None):
            from core.schemas import (
                ChatResponse,
                ModelResponse,
                RouterDecision,
                SynthesisResult,
            )
            called.append(payload.prompt)
            return ChatResponse(
                conversation_id='conv',
                router=RouterDecision(
                    prompt=payload.prompt, categories={},
                    candidates=[], selected_model_ids=[], reasoning=[],
                ),
                responses=[ModelResponse(model_id='m', provider='p',
                                          content='ok', latency_s=0.1)],
                synthesis=SynthesisResult(model_id='m', provider='p',
                                           content='stub',
                                           judge_summary='', winning_model_ids=[]),
                telemetry_id='tel',
            )

        monkeypatch.setattr('api.main.run_roitelet_chat', _stub_run)

        response = client.post('/api/chat', json={'prompt': '/personal anything'})
        assert response.status_code == 200
        # Empty wiki → no context injected, just the stripped prompt.
        assert called == ['anything']
