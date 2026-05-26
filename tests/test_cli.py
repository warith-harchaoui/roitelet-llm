"""CLI tests for Roitelet LLM.

These tests drive ``cli.main.main()`` with real ``sys.argv`` and a real
``input()`` substitute. The only seam stubbed out is the pipeline call
``run_roitelet_chat`` — replaced via :meth:`MonkeyPatch.setattr` with a
hand-written async function (no ``unittest.mock``).
"""

from __future__ import annotations

import sys

import pytest

from cli.main import main
from core.schemas import (
    ChatResponse,
    ModelResponse,
    RouterDecision,
    SynthesisResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_chat_response(content: str = 'Stub synthesis.') -> ChatResponse:
    """Construct a minimal valid :class:`ChatResponse` for stubbing."""
    return ChatResponse(
        conversation_id='conv-test',
        router=RouterDecision(
            prompt='stub',
            categories={'reasoning': 1.0},
            candidates=[],
            selected_model_ids=['ollama/qwen2.5:14b-instruct'],
            reasoning=['stubbed'],
        ),
        responses=[
            ModelResponse(
                model_id='ollama/qwen2.5:14b-instruct',
                provider='ollama',
                content=content,
                latency_s=0.01,
            )
        ],
        synthesis=SynthesisResult(
            model_id='ollama/qwen2.5:14b-instruct',
            provider='ollama',
            content=content,
            judge_summary=f'{content}\nWINNERS: 1',
            winning_model_ids=['ollama/qwen2.5:14b-instruct'],
        ),
        telemetry_id='tel-test',
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCliAskCommand:
    """``roitelet ask "..."`` runs single-shot and prints synthesis content."""

    def test_ask_invokes_pipeline_with_prompt(self, monkeypatch, capsys):
        captured: dict = {}

        async def stub_run(request):
            captured['prompt'] = request.prompt
            return _stub_chat_response(content='42')

        monkeypatch.setattr('cli.main.run_roitelet_chat', stub_run)
        monkeypatch.setattr(sys, 'argv', ['roitelet', 'ask', 'What is the answer?'])

        main()

        assert captured['prompt'] == 'What is the answer?'
        out = capsys.readouterr().out
        assert '42' in out

    def test_ask_exits_nonzero_on_pipeline_error(self, monkeypatch, capsys):
        async def stub_run(request):
            raise RuntimeError('boom')

        monkeypatch.setattr('cli.main.run_roitelet_chat', stub_run)
        monkeypatch.setattr(sys, 'argv', ['roitelet', 'ask', 'anything'])

        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert 'Error processing prompt' in out
        assert 'boom' in out


class TestCliChatCommand:
    """``roitelet chat`` runs the REPL until 'exit' or EOF."""

    def test_chat_repl_processes_then_exits(self, monkeypatch, capsys):
        invocations: list[str] = []

        async def stub_run(request):
            invocations.append(request.prompt)
            return _stub_chat_response(content=f'echo: {request.prompt}')

        monkeypatch.setattr('cli.main.run_roitelet_chat', stub_run)

        prompts = iter(['hello world', 'exit'])
        monkeypatch.setattr('builtins.input', lambda _prompt: next(prompts))
        monkeypatch.setattr(sys, 'argv', ['roitelet', 'chat'])

        main()

        # Only the non-exit prompt is dispatched to the pipeline.
        assert invocations == ['hello world']
        out = capsys.readouterr().out
        assert 'Welcome to Roitelet LLM CLI' in out
        assert 'echo: hello world' in out

    def test_chat_repl_skips_blank_prompts(self, monkeypatch, capsys):
        invocations: list[str] = []

        async def stub_run(request):
            invocations.append(request.prompt)
            return _stub_chat_response()

        monkeypatch.setattr('cli.main.run_roitelet_chat', stub_run)

        prompts = iter(['', '   ', 'real prompt', 'quit'])
        monkeypatch.setattr('builtins.input', lambda _prompt: next(prompts))
        monkeypatch.setattr(sys, 'argv', ['roitelet', 'chat'])

        main()

        assert invocations == ['real prompt']

    def test_chat_repl_exits_on_eof(self, monkeypatch, capsys):
        async def stub_run(request):
            return _stub_chat_response()

        monkeypatch.setattr('cli.main.run_roitelet_chat', stub_run)

        def raise_eof(_prompt):
            raise EOFError()

        monkeypatch.setattr('builtins.input', raise_eof)
        monkeypatch.setattr(sys, 'argv', ['roitelet', 'chat'])

        main()  # must not raise

        out = capsys.readouterr().out
        assert 'Exiting' in out


class TestCliHelp:
    """``roitelet`` with no subcommand prints help."""

    def test_no_command_prints_help(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, 'argv', ['roitelet'])
        main()
        out = capsys.readouterr().out
        assert 'usage' in out.lower()
        assert 'chat' in out
        assert 'ask' in out
