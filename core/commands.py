"""Slash-command parsing for inline per-turn overrides.

Users type natural-language prompts most of the time, but a small set
of leading slash commands carry per-turn overrides that would otherwise
need new JSON fields on :class:`ChatRequest`. Examples:

- ``/image a small wren in a forest`` — route to the image-gen pipeline.
- ``/speech`` — speech-to-text only (attach an audio file via the
  multimodal endpoint); the LLM pipeline is bypassed.
- ``/local refactor my code`` — force ``independence=True`` for this turn.
- ``/cheap 0.001 summarise this`` — cap ``max_cost_usd`` for this turn.
- ``/k 5 explain quicksort`` — fan out to K=5 candidates instead of 3.
- ``/help`` — return the catalogue inline as the assistant message.

Design rules:

1. **Only the leading token matters.** No mid-prompt directives — that
   would make prompt rewriting ambiguous and forbid the user from
   *talking about* slash commands ("what does ``/image`` do?").
2. **Composable.** ``/local /cheap 0.001 hello`` works: a single parse
   peels one command at a time until none remain. Order doesn't matter
   for overrides; routing commands (``/image``, ``/speech``,
   ``/help``) short-circuit.
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
>>> parse_command("/local /cheap 0.001 refactor").stripped_prompt
'refactor'
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
    ('/local <prompt>', 'Force independence mode (local OSS models only) for this turn.'),
    ('/cheap <usd> <prompt>', 'Set max_cost_usd for this turn. Filters paid candidates above the budget.'),
    ('/k <n> <prompt>', 'Override the top-K fan-out (1–8) for this turn.'),
    ('/help', 'Show this catalogue.'),
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
    independence_override: bool | None = None
    max_cost_usd_override: float | None = None
    top_k_override: int | None = None
    personal_override: bool = False
    matched_commands: list[str] = field(default_factory=list)


# Regex for the leading slash-command name. We match the command name
# only; arguments are parsed per-command below so each one can have
# its own arity rules.
_LEADING_CMD = re.compile(r'^\s*/([a-zA-Z][a-zA-Z0-9_-]*)\b')

# Arity-1 numeric args for /cheap and /k.
_LEADING_NUMBER = re.compile(r'^\s*([0-9]+(?:\.[0-9]+)?)\b')


def parse_command(prompt: str) -> ParsedCommand:
    """Peel recognised leading slash commands off ``prompt``.

    Walks the prompt left-to-right consuming one command per pass.
    Routing commands (``/image``, ``/speech``, ``/help``) short-circuit
    further parsing — they own the rest of the prompt. Override
    commands (``/local``, ``/cheap``, ``/k``) chain so a user can
    combine "force local" with "cap cost" in one line.

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

    # Iterate up to a small fixed number of passes to avoid pathological
    # loops on malformed input. In practice 4 is far more than any real
    # combination needs.
    for _ in range(4):
        match = _LEADING_CMD.match(remaining)
        if not match:
            break
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

        if name == 'local':
            result.independence_override = True
            result.matched_commands.append('/local')
            remaining = rest
            continue

        if name == 'personal':
            result.personal_override = True
            result.matched_commands.append('/personal')
            remaining = rest
            continue

        if name == 'cheap':
            num_match = _LEADING_NUMBER.match(rest)
            if not num_match:
                # Missing argument — pass the command through verbatim
                # rather than guess. The user sees no behaviour change
                # and can correct on the next turn.
                break
            result.max_cost_usd_override = float(num_match.group(1))
            result.matched_commands.append('/cheap')
            remaining = rest[num_match.end():].lstrip()
            continue

        if name == 'k':
            num_match = _LEADING_NUMBER.match(rest)
            if not num_match:
                break
            value = int(float(num_match.group(1)))
            # Clamp K to a sane range. K=0 is meaningless (pipeline
            # needs at least one candidate); K>8 wastes inference for
            # no gain (the judge is already overloaded above ~5).
            value = max(1, min(8, value))
            result.top_k_override = value
            result.matched_commands.append('/k')
            remaining = rest[num_match.end():].lstrip()
            continue

        # Unknown command name — fail-soft: stop peeling, leave the
        # command in the prompt as literal text.
        break

    result.stripped_prompt = remaining.strip()
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
