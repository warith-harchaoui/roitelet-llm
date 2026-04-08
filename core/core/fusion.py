"""Prompt builders for evaluation and local synthesis.

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from typing import List

from ..schemas import ChatMessage, ModelResponse

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
    """Create judge instructions for a local synthesis model.

    Parameters
    ----------
    prompt:
        Original user prompt.
    responses:
        Candidate model responses to compare.

    Returns
    -------
    list of ChatMessage
        System + user messages for the local judge.
    """
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
