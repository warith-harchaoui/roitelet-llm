"""API integration tests for the Personal-mode endpoints.

Two tests:

* ``GET /api/personal`` + ``POST /api/personal/ingest`` +
  ``GET /api/personal/embeddings`` all round-trip on the same wiki
  state. One ``client`` exercises the lifecycle end-to-end.
* The ``/personal`` slash on ``/api/chat`` prepends wiki content
  to the prompt the pipeline receives (and tolerates an empty
  knowledge base).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _reset_singletons() -> None:
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
    yield
    _reset_singletons()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('ROITELET_DATA_DIR', str(tmp_path))
    _reset_singletons()
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_personal_endpoints_round_trip_status_ingest_and_embeddings(
    client, tmp_path, monkeypatch,
):
    """The three personal-mode endpoints share one wiki state — one test
    walks the lifecycle:

    1. an empty install reports ``mode='empty'`` everywhere;
    2. dropping a text file in the inbox and posting ``/ingest``
       converts it, after which ``status`` reports the new wiki entry;
    3. ``/embeddings`` returns ``[]`` cleanly when the embedder is
       unreachable, and real ``{x, y, …}`` records when it is.
    """
    # 1. Empty install.
    assert client.get('/api/personal').json() == {
        'inbox': 0, 'wiki': 0, 'wiki_chars': 0, 'mode': 'empty',
    }

    # 2. Drop a known and an unknown file, ingest.
    inbox = tmp_path / 'personal' / 'inbox'
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / 'note.md').write_text('# Topic\n\nA note.', encoding='utf-8')
    (inbox / 'weird.xyz').write_text('junk', encoding='utf-8')

    ingest = client.post('/api/personal/ingest').json()
    modalities = {r['source'].split('/')[-1]: r['modality'] for r in ingest['results']}
    assert modalities['note.md'] == 'text'
    assert modalities['weird.xyz'] == 'skipped'
    assert ingest['status']['wiki'] == 1
    # Status agrees with the ingest result on a follow-up call.
    assert client.get('/api/personal').json()['mode'] == 'wiki'

    # 3a. Embeddings — empty when the embedder is unreachable.
    import core.capability_classifier as cc
    monkeypatch.setattr(cc, '_embed_prompt', lambda prompt: None)
    assert client.get('/api/personal/embeddings').json() == {'points': []}

    # 3b. Embeddings — real coordinates when the embedder works.
    def stub_embed(text: str) -> np.ndarray:
        v = np.zeros(32, dtype=np.float32)
        v[len(text) % 32] = 1.0
        if text:
            v[ord(text[0]) % 32] += 0.5
        return v

    monkeypatch.setattr(cc, '_embed_prompt', stub_embed)
    wiki = tmp_path / 'personal' / 'wiki'
    (wiki / 'b.md').write_text('# B\n\n' + ('beta. ' * 50), encoding='utf-8')
    body = client.get('/api/personal/embeddings').json()
    assert body['points']
    for p in body['points']:
        assert {'path', 'chunk_index', 'text', 'x', 'y'} <= set(p)


def test_personal_slash_command_prepends_wiki_to_the_pipeline_prompt(
    client, tmp_path, monkeypatch,
):
    """The ``/personal`` slash branch on ``/api/chat`` should prepend the
    wiki context block before the prompt reaches the pipeline, strip
    the slash, and tolerate an empty knowledge base (no context →
    just the stripped prompt)."""
    captured: list[str] = []

    async def stub_run(payload, router=None):
        from core.schemas import (
            ChatResponse,
            ModelResponse,
            RouterDecision,
            SynthesisResult,
        )
        captured.append(payload.prompt)
        return ChatResponse(
            conversation_id='conv',
            router=RouterDecision(
                prompt=payload.prompt, categories={},
                candidates=[], selected_model_ids=[], reasoning=[],
            ),
            responses=[ModelResponse(model_id='m', provider='p',
                                     content='ok', latency_s=0.1)],
            synthesis=SynthesisResult(model_id='m', provider='p',
                                      content='stub', judge_summary='',
                                      winning_model_ids=[]),
            telemetry_id='tel',
        )

    monkeypatch.setattr('api.main.run_roitelet_chat', stub_run)

    # 1. With a wiki entry → the pipeline gets wiki content + the question.
    wiki = tmp_path / 'personal' / 'wiki'
    wiki.mkdir(parents=True, exist_ok=True)
    (wiki / 'fact.md').write_text(
        '# Important\n\nThe wren weighs 9 grams.', encoding='utf-8',
    )
    client.post('/api/chat', json={'prompt': '/personal How much does a wren weigh?'})
    augmented = captured[-1]
    assert 'How much does a wren weigh?' in augmented
    assert 'wren weighs 9 grams' in augmented
    assert not augmented.startswith('/personal')

    # 2. With an empty wiki → just the stripped prompt.
    for p in wiki.glob('*'):
        p.unlink()
    client.post('/api/chat', json={'prompt': '/personal anything'})
    assert captured[-1] == 'anything'
