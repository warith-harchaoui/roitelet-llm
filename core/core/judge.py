"""Local judging and synthesis utilities.

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
from ..storage import storage
from ..schemas import ModelResponse, SynthesisResult
from .fusion import build_judge_messages


def parse_winners(text: str) -> List[int]:
    """Extract winning candidate indices from judge output.

    Parameters
    ----------
    text:
        Judge completion text.

    Returns
    -------
    list of int
        1-based winning candidate indices.
    """
    match = re.search(r'WINNERS:\s*([0-9, ]+)', text)
    if not match:
        return [1]
    values = [chunk.strip() for chunk in match.group(1).split(',') if chunk.strip()]
    winners = [int(value) for value in values if value.isdigit()]
    return winners or [1]


async def judge_and_synthesize(prompt: str, responses: List[ModelResponse]) -> SynthesisResult:
    """Use the local synthesis model to judge and fuse candidate answers.

    Respects the ``local_llm_provider`` runtime setting so that non-Ollama
    local backends (e.g. a local OpenAI-compatible server) are also supported.
    """
    runtime_settings = storage.load_app_settings()
    local_synthesis_model = runtime_settings.local_synthesis_model

    # Derive the provider from the model name prefix (e.g. "ollama/…" → "ollama").
    # Fall back gracefully to Ollama for bare model names.
    if '/' in local_synthesis_model:
        provider_hint, bare_model = local_synthesis_model.split('/', 1)
    else:
        provider_hint = 'ollama'
        bare_model = local_synthesis_model

    # Map the provider hint to a registered provider key.
    provider_key = provider_hint if provider_hint in ('ollama', 'openrouter', 'openai-compatible') else 'ollama'
    model_id = f"{provider_key}/{bare_model}"

    client = get_provider_client(provider_key)
    judge_response = await client.generate(model_id=model_id, messages=build_judge_messages(prompt, responses))
    judge_text = judge_response.content or 'Judge unavailable. Returning the top candidate answer.'
    winners = parse_winners(judge_text)
    winning_model_ids = (
        [responses[index - 1].model_id for index in winners if 0 < index <= len(responses)]
        or [responses[0].model_id]
    )
    final_content = judge_text.split('WINNERS:')[0].strip()
    if not final_content:
        final_content = responses[0].content
    return SynthesisResult(
        model_id=model_id,
        provider=provider_key,
        content=final_content,
        judge_summary=judge_text,
        winning_model_ids=winning_model_ids,
    )
