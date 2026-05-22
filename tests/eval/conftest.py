"""Pytest configuration for the answer-quality eval suite.

The eval suite is opt-in: it requires ``deepeval`` to be installed and a
running Ollama instance. If either is missing every test in this folder
collects as **skipped** rather than failing the run.

A custom :class:`DeepEvalBaseLLM` subclass routes DeepEval's grading calls
through the project's own ``OllamaClient`` so the grader uses the same
local model the production judge uses. That means:

* no paid API keys are needed to grade,
* the grader's notion of "good" matches the judge's notion of "good",
* swapping the local model (``LOCAL_LLM_MODEL``) updates both at once.
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Hard requirement: deepeval. Skip the entire eval folder gracefully if missing.
deepeval = pytest.importorskip(
    'deepeval',
    reason='deepeval not installed — install with `pip install -e .[eval]`',
)

from deepeval.models.base_model import DeepEvalBaseLLM  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.providers.ollama import OllamaClient  # noqa: E402
from core.schemas import ChatMessage  # noqa: E402


class OllamaJudge(DeepEvalBaseLLM):
    """Adapter that lets DeepEval grade with the project's local Ollama model.

    DeepEval's stock backends assume OpenAI/Anthropic credentials; this
    subclass plugs in the same :class:`OllamaClient` the rest of Roitelet
    uses, so the eval grader and the production synthesis judge share a
    single source of truth for "what counts as a good answer."
    """

    def __init__(self, model_name: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.local_llm_model
        self.client = OllamaClient(base_url=base_url or settings.local_llm_base_url)

    def load_model(self):  # noqa: D401 — DeepEval API contract
        return self

    def get_model_name(self) -> str:
        return f'ollama/{self.model_name}'

    def generate(self, prompt: str) -> str:
        return asyncio.get_event_loop().run_until_complete(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        response = await self.client.generate(
            model_id=f'ollama/{self.model_name}',
            messages=[ChatMessage(role='user', content=prompt)],
        )
        return response.content or ''


@pytest.fixture(scope='session')
def ollama_judge() -> OllamaJudge:
    """Shared local-Ollama grader for every eval test."""
    return OllamaJudge()


@pytest.fixture(scope='session', autouse=True)
def _require_ollama_reachable(ollama_judge):
    """Skip the whole eval suite if no Ollama server answers on the configured port.

    Running 15+ eval prompts against an unreachable server produces 15+
    misleading failures. Skipping once at the top keeps the signal clean.
    """
    import httpx

    url = ollama_judge.client.base_url + '/api/tags'
    try:
        response = httpx.get(url, timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f'Ollama not reachable at {url}: {exc}')


# Eval tests are unconditionally marked so they're easy to in/exclude in CI.
def pytest_collection_modifyitems(config, items):
    """Auto-apply the ``eval`` marker to every test in this folder."""
    eval_root = os.path.dirname(__file__)
    for item in items:
        if item.fspath.dirname.startswith(eval_root):
            item.add_marker(pytest.mark.eval)
