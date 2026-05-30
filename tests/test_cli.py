"""CLI tests for the ``roitelet`` console-script.

Two tests:

* ``roitelet ask`` round-trips a prompt through the pipeline and exits
  non-zero on a pipeline error.
* ``roitelet chat`` runs the REPL, dispatches every non-empty prompt,
  skips blanks, and exits cleanly on ``exit`` / Ctrl-D.

The only seam stubbed out is the pipeline call; argparse, the REPL
loop, and the welcome banner all run for real.
"""

from __future__ import annotations

import sys

import pytest

from cli.main import main
from core.schemas import ChatResponse, ModelResponse, RouterDecision, SynthesisResult


def _stub_response(content: str = 'Stub.') -> ChatResponse:
    return ChatResponse(
        conversation_id='conv',
        router=RouterDecision(
            prompt='stub', categories={'reasoning': 1.0}, candidates=[],
            selected_model_ids=['ollama/qwen2.5:14b-instruct'],
            reasoning=['stubbed'],
        ),
        responses=[ModelResponse(
            model_id='ollama/qwen2.5:14b-instruct', provider='ollama',
            content=content, latency_s=0.01,
        )],
        synthesis=SynthesisResult(
            model_id='ollama/qwen2.5:14b-instruct', provider='ollama',
            content=content, judge_summary=f'{content}\nWINNERS: 1',
            winning_model_ids=['ollama/qwen2.5:14b-instruct'],
        ),
        telemetry_id='tel',
    )


def test_ask_dispatches_to_pipeline_and_propagates_errors(monkeypatch, capsys):
    """``roitelet ask <prompt>`` should forward the prompt verbatim
    and print the synthesis. A pipeline failure must exit with status 1
    and print a clear error rather than crashing with a traceback."""
    captured: dict = {}

    async def stub_ok(request):
        captured['prompt'] = request.prompt
        return _stub_response('42')

    monkeypatch.setattr('cli.main.run_roitelet_chat', stub_ok)
    monkeypatch.setattr(sys, 'argv', ['roitelet', 'ask', 'What is the answer?'])
    main()
    assert captured['prompt'] == 'What is the answer?'
    assert '42' in capsys.readouterr().out

    async def stub_boom(request):
        raise RuntimeError('boom')

    monkeypatch.setattr('cli.main.run_roitelet_chat', stub_boom)
    monkeypatch.setattr(sys, 'argv', ['roitelet', 'ask', 'anything'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert 'Error processing prompt' in out and 'boom' in out


def test_chat_repl_dispatches_non_blank_prompts_until_exit(monkeypatch, capsys):
    """The REPL prints the welcome banner, dispatches every non-blank
    line through the pipeline, and exits cleanly on ``exit`` *or* EOF.

    Blank prompts must be skipped silently — otherwise hitting Enter
    by mistake would burn a real fan-out + judge call."""
    invocations: list[str] = []

    async def stub(request):
        invocations.append(request.prompt)
        return _stub_response(f'echo: {request.prompt}')

    monkeypatch.setattr('cli.main.run_roitelet_chat', stub)
    monkeypatch.setattr(sys, 'argv', ['roitelet', 'chat'])

    # Mix of blank, whitespace-only, real prompt, then 'exit'.
    prompts = iter(['', '   ', 'real prompt', 'exit'])
    monkeypatch.setattr('builtins.input', lambda _p: next(prompts))
    main()
    assert invocations == ['real prompt']

    out = capsys.readouterr().out
    assert 'Welcome to Roitelet CLI' in out
    assert 'echo: real prompt' in out

    # EOF (Ctrl-D) exits cleanly without raising.
    invocations.clear()
    monkeypatch.setattr('builtins.input', lambda _p: (_ for _ in ()).throw(EOFError()))
    monkeypatch.setattr(sys, 'argv', ['roitelet', 'chat'])
    main()
    assert 'Exiting' in capsys.readouterr().out
