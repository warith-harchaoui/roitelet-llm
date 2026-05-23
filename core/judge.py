"""Local judging and synthesis.

Hardened version: the judge never sees model identities (each candidate is
relabelled with a fresh per-call hex handle and the order is shuffled), the
rubric is explicitly anchored to the user prompt, and the winners marker is
a distinctive sentinel that fails closed — a missing or garbled marker
yields *no* winners rather than silently crowning candidate 1.

Examples
--------
>>> from core.judge import parse_winners
>>> parse_winners('Some prose.\\n===WINNERS===\\nab3c, 9e2f', {'ab3c', '9e2f'})
['ab3c', '9e2f']

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import re
import secrets
from typing import Dict, List, Set, Tuple

from . import storage as _storage_mod
from .providers.factory import get_provider_client
from .schemas import ChatMessage, ModelResponse, SynthesisResult


# Sentinel chosen to be visually distinct from anything a typical answer
# might produce. Triple-equals on a line of its own does not appear in
# normal prose, code, or Markdown.
_WINNERS_OPEN = '===WINNERS==='
_WINNERS_CLOSE = '===END==='


_JUDGE_SYSTEM_PROMPT = (
    "You are the Roitelet synthesis judge. You receive the user's original "
    "prompt and several candidate answers labelled with opaque handles "
    "(e.g. `Candidate ab3c9f1d`). The handles are random — they reveal "
    "nothing about which model produced which answer. Do not speculate "
    "about model identities; judge purely on the merits of each answer.\n\n"
    "Your judgement must be relative to the user's prompt, scored on:\n"
    "  1. Accuracy — does it answer correctly, without hallucination?\n"
    "  2. Relevance — does it address what was actually asked?\n"
    "  3. Completeness — does it cover the prompt's requirements?\n"
    "  4. Clarity — is it well-structured and easy to follow?\n\n"
    "Then produce a single fused answer in clean Markdown that combines the "
    "strongest elements of the best candidates. Fuse, do not pick.\n\n"
    "Output format — strict:\n"
    f"  1. Your fused answer (Markdown).\n"
    f"  2. A line containing exactly: {_WINNERS_OPEN}\n"
    f"  3. The comma-separated handles of the candidates whose content you\n"
    f"     actually used (e.g. `ab3c9f1d, 9e2f4a17`).\n"
    f"  4. A line containing exactly: {_WINNERS_CLOSE}\n"
    f"Nothing after {_WINNERS_CLOSE}."
)


def _new_handle() -> str:
    """Return a fresh 8-hex-char anonymous candidate handle."""
    return secrets.token_hex(4)


def build_judge_messages(
    prompt: str,
    responses: List[ModelResponse],
) -> Tuple[List[ChatMessage], Dict[str, str]]:
    """Build the judge messages with shuffled, anonymized candidates.

    Returns
    -------
    Tuple[List[ChatMessage], Dict[str, str]]
        The chat messages to send to the local judge, plus a mapping from
        anonymous handle to real ``model_id``. The caller uses the mapping
        to translate parsed winners back into model identifiers without
        ever exposing those identifiers to the judge.
    """
    # Per-call randomized order + fresh handles. Position can no longer
    # leak provider identity (no stable "candidate 1 = openrouter/...").
    indices = list(range(len(responses)))
    secrets.SystemRandom().shuffle(indices)

    handle_to_model: Dict[str, str] = {}
    blocks: List[str] = []
    for i in indices:
        response = responses[i]
        handle = _new_handle()
        # Extremely unlikely, but guarantee uniqueness within this call.
        while handle in handle_to_model:
            handle = _new_handle()
        handle_to_model[handle] = response.model_id
        body = f'[ERROR: {response.error}]' if response.error else response.content
        blocks.append(f'Candidate {handle}\n{body}')

    user_prompt = (
        "Original user prompt:\n"
        f"{prompt}\n\n"
        "Candidate answers (anonymous, randomized order):\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\n"
        "Now produce the fused Markdown answer evaluated against the user's "
        "prompt above, followed by the winners block in the exact format "
        f"specified ({_WINNERS_OPEN} / handles / {_WINNERS_CLOSE})."
    )
    messages = [
        ChatMessage(role='system', content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(role='user', content=user_prompt),
    ]
    return messages, handle_to_model


_TOKEN_RE = re.compile(r'[0-9a-fA-F]{4,}')


def parse_winners(text: str, valid_handles: Set[str]) -> List[str]:
    """Extract winner handles from judge output, fail-closed on garbage.

    Parameters
    ----------
    text:
        The raw text emitted by the judge.
    valid_handles:
        The set of handles actually presented to the judge for this call.
        Any token outside this set is ignored — the judge cannot reward a
        candidate that does not exist, and stray hex in prose cannot
        contaminate the winners list.

    Returns
    -------
    List[str]
        Winning handles in the order they appear inside the block. Empty
        when the marker is missing, malformed, or yields no valid handle.
        An empty result means "no decision" — the caller MUST treat it as
        a no-op rather than defaulting to a winner.
    """
    # Strict path: the explicitly delimited block.
    pattern = re.compile(
        rf'{re.escape(_WINNERS_OPEN)}\s*(.+?)\s*{re.escape(_WINNERS_CLOSE)}',
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        # Tolerant fallback: open marker present but close marker missing
        # (the judge truncated). Take everything after the opener up to
        # the next blank line or end of text.
        open_match = re.search(rf'{re.escape(_WINNERS_OPEN)}\s*(.+?)(?:\n\s*\n|\Z)', text, re.DOTALL)
        if not open_match:
            return []
        block = open_match.group(1)
    else:
        block = match.group(1)

    seen: Set[str] = set()
    ordered: List[str] = []
    for token in _TOKEN_RE.findall(block):
        normalized = token.lower()
        if normalized in valid_handles and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _strip_winners_block(text: str) -> str:
    """Remove the winners block (and anything after it) from user-visible content."""
    # Cut at the first opener regardless of whether the close marker exists.
    idx = text.find(_WINNERS_OPEN)
    if idx == -1:
        return text.rstrip()
    return text[:idx].rstrip()


async def judge_and_synthesize(prompt: str, responses: List[ModelResponse]) -> SynthesisResult:
    """Judge and fuse candidate answers using a local synthesis model.

    Behavior contract:
    - Candidates are presented anonymously in shuffled order.
    - Judgement is explicitly relative to ``prompt``.
    - If the local judge is unreachable or returns empty content, the
      top candidate is surfaced verbatim with *no* winners (so the Elo
      loop receives no spurious reward).
    - If the judge replies but the winners marker is missing or yields no
      valid handle, ``winning_model_ids`` is ``[]`` and ``judge_summary``
      records the parse failure honestly. The user still gets the fused
      content (minus any stray winners block).
    """
    runtime_settings = _storage_mod.get_storage().load_app_settings()
    local_synthesis_model = runtime_settings.local_synthesis_model

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
    model_id = f'{provider_key}/{bare_model}'

    messages, handle_to_model = build_judge_messages(prompt, responses)
    valid_handles = set(handle_to_model.keys())

    client = get_provider_client(provider_key)
    judge_response = await client.generate(model_id=model_id, messages=messages)
    judge_text = judge_response.content or ''

    if not judge_text.strip():
        # Judge unreachable / empty. Surface top candidate verbatim and
        # emit NO winners — there is no signal to reward.
        top = responses[0]
        return SynthesisResult(
            model_id=model_id,
            provider=provider_key,
            content=top.content,
            judge_summary=(
                f'Judge unavailable (provider={provider_key}, model={bare_model}). '
                f'Returning the first candidate verbatim; no winners recorded.'
            ),
            winning_model_ids=[],
        )

    winner_handles = parse_winners(judge_text, valid_handles)
    winning_model_ids = [handle_to_model[h] for h in winner_handles]

    final_content = _strip_winners_block(judge_text) or responses[0].content

    if not winning_model_ids:
        # Judge spoke but its winners marker was missing or unparseable.
        # We keep the fused content (it's the user's answer) but emit no
        # winners so the Elo loop stays silent rather than learning noise.
        summary = (
            f'{judge_text}\n\n'
            '[note] Winners block missing or unparseable — no reward signal recorded.'
        )
        return SynthesisResult(
            model_id=model_id,
            provider=provider_key,
            content=final_content,
            judge_summary=summary,
            winning_model_ids=[],
        )

    return SynthesisResult(
        model_id=model_id,
        provider=provider_key,
        content=final_content,
        judge_summary=judge_text,
        winning_model_ids=winning_model_ids,
    )
