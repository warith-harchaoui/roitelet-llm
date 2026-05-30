"""Slash-command parsing — routes only, no per-turn preference overrides.

Slash commands here mean **route selection**, not per-turn preference
toggles. The user picks a route by what they type at the start of the
prompt; preferences (top_k, independence, ecofrugality, max_cost_usd,
pseudonymize) live on the visible UI control panel — a sliders
popover in the web composer, ``--`` flags in the CLI, JSON booleans
in the API. They are not slash-typed because invisible state is a UX
trap: the user can't see what's active.

The routes:

- ``/image a small wren in a forest`` — image-gen pipeline.
- ``/speech`` — speech-to-text only (attach an audio file via the
  multimodal endpoint); the LLM pipeline is bypassed.
- ``/personal what did I write about Q3?`` — inject the personal
  knowledge base into the prompt.
- ``/help`` — return the catalogue inline as the assistant message.

Design rules:

1. **Only the leading token matters.** No mid-prompt directives —
   that would make prompt rewriting ambiguous and forbid the user
   from *talking about* slash commands ("what does ``/image`` do?").
2. **Routes short-circuit.** A slash route owns the rest of the
   prompt; there is no chaining.
3. **Fail-soft.** Unknown commands are *passed through verbatim* as
   part of the prompt — typos don't silently change behaviour.
4. **No state.** Parsing is pure; the result is a dataclass the
   pipeline / API layer interprets.

Examples
--------
>>> from core.commands import parse_command
>>> parse_command("hello world").route_to
'chat'
>>> parse_command("/image a sunset").route_to
'image'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

RouteTarget = Literal['chat', 'image', 'speech', 'help']


# Catalogue lines, used both by ``/help`` and by the docs surface.
# Tuple-of-tuples (not dict) so ordering is stable.
HELP_LINES: tuple[tuple[str, str], ...] = (
    ('/image <prompt>', 'Generate an image. Routes to the image-gen pipeline (K=1, no fusion).'),
    ('/speech', 'Speech-to-text + diarisation only. Requires an audio attachment via /api/chat/multimodal.'),
    ('/personal <prompt>', 'Inject your personal knowledge base (data/personal/wiki/) into the prompt.'),
    ('/help', 'Show this catalogue.'),
)

# Per-turn preferences (top_k, independence, ecofrugality, max_cost_usd,
# pseudonymize) used to be exposed as slash commands too (``/local``,
# ``/cheap``, ``/k``, ``/pseudo``, ``/nopseudo``). They were removed on
# 2026-05-30 in favour of visible-state controls: a sliders popover in
# the web composer, ``--`` flags in the CLI, JSON booleans in the
# API. Invisible state is a UX trap; visible state is auditable.
PREFERENCE_SURFACES = (
    'Web composer: click the sliders icon next to the send button.',
    'CLI: pass --top-k / --independence / --pseudonymize / --max-cost-usd to roitelet ask|chat.',
    'API: set the corresponding boolean in `preferences` on POST /api/chat.',
)


@dataclass(slots=True)
class ParsedCommand:
    """The result of parsing slash-command prefixes off a prompt.

    Attributes
    ----------
    route_to:
        Which pipeline the API layer should dispatch to.
        ``'chat'`` is the standard text fan-out; ``'image'`` routes to
        :func:`core.image_pipeline.run_roitelet_image_chat`; ``'speech'``
        means STT-only (the multimodal endpoint already handles that);
        ``'help'`` short-circuits to a static catalogue response.
    stripped_prompt:
        The prompt with every recognised leading slash command peeled
        off. May be empty for ``/help`` or ``/speech``.
    independence_override:
        ``True`` when ``/local`` fired; ``None`` to leave the caller's
        preference unchanged.
    max_cost_usd_override:
        Value set by ``/cheap``, or ``None``.
    top_k_override:
        Value set by ``/k``, or ``None``.
    personal_override:
        ``True`` when ``/personal`` fired — the chat endpoint should
        prepend the personal-knowledge-base context block to the prompt
        before fan-out.
    """

    route_to: RouteTarget = 'chat'
    stripped_prompt: str = ''
    personal_override: bool = False
    matched_commands: list[str] = field(default_factory=list)


# Regex for the leading slash-command name. We match the command name
# only.
_LEADING_CMD = re.compile(r'^\s*/([a-zA-Z][a-zA-Z0-9_-]*)\b')


def parse_command(prompt: str) -> ParsedCommand:
    """Peel a recognised leading route slash off ``prompt``.

    All recognised slashes route to a different pipeline (image, STT,
    personal-RAG injection, static help). Per-turn preferences are
    NOT slash-typed — see :data:`PREFERENCE_SURFACES` for where they
    live.

    Parameters
    ----------
    prompt:
        Raw user text. May contain leading whitespace.

    Returns
    -------
    ParsedCommand
        A populated record. If no command fired, the ``route_to`` is
        ``'chat'`` and ``stripped_prompt`` equals ``prompt.strip()``.
    """
    result = ParsedCommand(stripped_prompt=prompt.strip())
    remaining = result.stripped_prompt

    match = _LEADING_CMD.match(remaining)
    if not match:
        return result
    name = match.group(1).lower()
    rest = remaining[match.end():].lstrip()

    if name in ('image', 'image-gen', 'img'):
        result.route_to = 'image'
        result.stripped_prompt = rest
        result.matched_commands.append('/image')
        return result

    if name in ('speech', 'stt', 'transcribe'):
        result.route_to = 'speech'
        result.stripped_prompt = rest
        result.matched_commands.append('/speech')
        return result

    if name == 'help':
        result.route_to = 'help'
        result.stripped_prompt = ''
        result.matched_commands.append('/help')
        return result

    if name == 'personal':
        result.personal_override = True
        result.stripped_prompt = rest
        result.matched_commands.append('/personal')
        return result

    # Unknown command name — fail-soft: leave the command in the prompt
    # as literal text so typos don't silently change behaviour.
    return result


def render_help() -> str:
    """Format the slash-command catalogue as a single Markdown block.

    Used by the ``/help`` short-circuit so the assistant message body
    is a usable answer rather than an empty string.
    """
    lines = ['Roitelet slash-command catalogue:', '']
    for name, description in HELP_LINES:
        lines.append(f'- **`{name}`** — {description}')
    return '\n'.join(lines)
