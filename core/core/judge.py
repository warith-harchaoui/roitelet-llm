"""Local judging and synthesis.

Builds the judge prompt, calls a local synthesis model, parses the
``WINNERS:`` line, and returns a single fused answer.

Examples
--------
>>> from core.core.judge import parse_winners
>>> parse_winners('Candidate 1 is best.\\nWINNERS: 1, 3')
[1, 3]

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import re
from typing import List

from ..providers.factory import get_provider_client
from ..schemas import ChatMessage, ModelResponse, SynthesisResult
from ..storage import storage


_JUDGE_SYSTEM_PROMPT = (
    "You are the Roitelet synthesis judge. Your role is to evaluate multiple LLM responses "
    "to the same user prompt and produce the single best fused answer.\n\n"
    "Rules:\n"
    "- Be objective and rigorous. Prefer accuracy over length.\n"
    "- Cite the strengths and weaknesses of each candidate briefly.\n"
    "- Your final fused answer must be in clean Markdown.\n"
    "- Always end your output with the machine-readable line:\n"
    "  WINNERS: <comma-separated candidate numbers, e.g. 1, 3>\n"
    "- Do NOT add anything after the WINNERS line."
)


def build_judge_messages(prompt: str, responses: List[ModelResponse]) -> List[ChatMessage]:
    """Build the system + user messages sent to the local synthesis model."""
    formatted = []
    for index, response in enumerate(responses, start=1):
        status = f"[ERROR: {response.error}]" if response.error else response.content
        formatted.append(f"Candidate {index} | {response.model_id}\n{status}")

    user_prompt = (
        f"Original prompt:\n{prompt}\n\n"
        "Tasks:\n"
        "1. Explain the strengths and weaknesses of each candidate.\n"
        "2. Choose the best subset of answers worth fusing.\n"
        "3. Produce a final fused answer in Markdown.\n"
        "4. End with exactly:\nWINNERS: candidate numbers separated by commas\n\n"
        "Candidates:\n" + "\n\n".join(formatted)
    )
    return [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


def parse_winners(text: str) -> List[int]:
    """Extract winning candidate indices from judge output.

    Tolerates non-digit junk between indices ("1 and 3", "1, x, 3") so a
    stray token doesn't drop later winners.
    """
    match = re.search(r'WINNERS:\s*(.+)', text)
    if not match:
        return [1]
    winners = [int(value) for value in re.findall(r'\d+', match.group(1))]
    return winners or [1]


async def judge_and_synthesize(prompt: str, responses: List[ModelResponse]) -> SynthesisResult:
    """Use the local synthesis model to judge and fuse candidate answers.

    Respects the ``local_llm_provider`` runtime setting so non-Ollama local
    backends (e.g. a local OpenAI-compatible server) are also supported. If
    the judge returns empty content the result falls back to the top
    candidate verbatim, and ``judge_summary`` records that fact instead of
    pretending the judge spoke.
    """
    runtime_settings = storage.load_app_settings()
    local_synthesis_model = runtime_settings.local_synthesis_model

    # Derive the provider from the model name prefix (e.g. "ollama/…" → "ollama").
    if '/' in local_synthesis_model:
        provider_hint, bare_model = local_synthesis_model.split('/', 1)
    else:
        provider_hint = 'ollama'
        bare_model = local_synthesis_model

    provider_key = (
        provider_hint
        if provider_hint in ('ollama', 'openrouter', 'openai-compatible')
        else 'ollama'
    )
    model_id = f"{provider_key}/{bare_model}"

    client = get_provider_client(provider_key)
    judge_response = await client.generate(
        model_id=model_id,
        messages=build_judge_messages(prompt, responses),
    )
    judge_text = judge_response.content or ''

    if not judge_text.strip():
        # Judge unreachable / empty completion. Surface the top candidate
        # unchanged and record what actually happened — don't pretend the
        # judge spoke.
        top = responses[0]
        return SynthesisResult(
            model_id=model_id,
            provider=provider_key,
            content=top.content,
            judge_summary=(
                f'Judge unavailable (provider={provider_key}, model={bare_model}). '
                f'Returning candidate 1 ({top.model_id}) verbatim.'
            ),
            winning_model_ids=[top.model_id],
        )

    winners = parse_winners(judge_text)
    winning_model_ids = (
        [responses[index - 1].model_id for index in winners if 0 < index <= len(responses)]
        or [responses[0].model_id]
    )
    final_content = judge_text.split('WINNERS:')[0].strip() or responses[0].content
    return SynthesisResult(
        model_id=model_id,
        provider=provider_key,
        content=final_content,
        judge_summary=judge_text,
        winning_model_ids=winning_model_ids,
    )
