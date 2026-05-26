"""Command-line interface for Roitelet LLM.

Provides terminal access to the Roitelet pipeline. Mirrors gemini-cli's
shape: a single-shot ``ask`` subcommand and an interactive ``chat``
REPL.

Examples
--------
>>> # Run a single-shot question:
>>> #   python -m cli ask "Explain quicksort"

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from core.personal import (
    build_personal_context,
    ingest_inbox,
    inbox_dir,
    personal_status,
    wiki_dir,
)
from core.pipeline import run_roitelet_chat
from core.schemas import ChatRequest, RouterPreferences


def print_welcome() -> None:
    """Print the welcome banner shown at the top of the interactive REPL.

    Notes
    -----
    Side effects: writes three lines to stdout. Pure formatting; no
    behaviour gates on this output.
    """
    print("=======================================")
    print(" Welcome to Roitelet LLM CLI")
    print("=======================================")


async def chat_repl() -> None:
    """Run the interactive chat loop until ``exit``/``quit``/EOF.

    Each iteration reads a line from stdin, sends it through the
    Roitelet pipeline with default :class:`RouterPreferences`, and
    prints the fused synthesis. ``Ctrl+C`` and ``Ctrl+D`` both exit
    cleanly.

    Notes
    -----
    Exceptions raised by ``run_roitelet_chat`` (provider failures,
    judge fallback, etc.) are caught per-iteration so a single bad
    prompt doesn't kill the session. The error is printed to stdout.
    """
    print_welcome()
    print("Type 'exit' or 'quit' to end the session.\n")
    while True:
        try:
            prompt = input("You> ")
            clean_prompt = prompt.strip()
            if clean_prompt.lower() in ('exit', 'quit'):
                break
            if not clean_prompt:
                continue

            request = ChatRequest(prompt=clean_prompt, preferences=RouterPreferences())
            response = await run_roitelet_chat(request)

            print(f"\nRoitelet> {response.synthesis.content}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


async def single_prompt(prompt: str) -> None:
    """Execute exactly one prompt and exit.

    Parameters
    ----------
    prompt : str
        The user's prompt sent through :func:`run_roitelet_chat` with
        default preferences. Whitespace is preserved (the caller is
        responsible for trimming if relevant).

    Notes
    -----
    On any pipeline failure the error is printed to stdout and the
    process exits with status 1, matching ``ask``-style CLIs.
    """
    try:
        request = ChatRequest(prompt=prompt, preferences=RouterPreferences())
        response = await run_roitelet_chat(request)
        print(response.synthesis.content)
    except Exception as exc:
        print(f"Error processing prompt: {exc}")
        sys.exit(1)


async def personal_ask(prompt: str) -> None:
    """Run one chat turn with the personal knowledge base injected.

    Parameters
    ----------
    prompt : str
        User question. The personal-mode context block (full wiki for
        small corpora, top-K retrieval for large ones) is prepended
        before fan-out.
    """
    context = build_personal_context(prompt)
    augmented = f'{context}\n## Question\n\n{prompt}' if context else prompt
    try:
        response = await run_roitelet_chat(
            ChatRequest(prompt=augmented, preferences=RouterPreferences())
        )
        print(response.synthesis.content)
    except Exception as exc:
        print(f'Error: {exc}')
        sys.exit(1)


def personal_dispatch(args: argparse.Namespace) -> None:
    """Handle the ``personal`` subcommand family.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``args.personal_command`` is one of
        ``'status'``, ``'ingest'``, ``'list'``, ``'ask'``.
    """
    sub = args.personal_command
    if sub == 'status':
        status = personal_status()
        print(f"Personal corpus: {status['wiki']} wiki file(s), "
              f"{status['inbox']} inbox file(s), "
              f"{status['wiki_chars']} chars, mode={status['mode']}")
        print(f"  inbox: {inbox_dir()}")
        print(f"  wiki:  {wiki_dir()}")
    elif sub == 'ingest':
        results = asyncio.run(ingest_inbox(force=args.force))
        for r in results:
            if r.error:
                print(f"  ✗ {r.source.name:30s} ({r.modality}) — {r.error}")
            elif r.modality == 'skipped':
                print(f"  · {r.source.name:30s} skipped (unknown extension)")
            else:
                arrow = '→ existing' if r.wiki_path and r.wiki_path.exists() else '→ ?'
                if r.wiki_path:
                    arrow = f'→ {r.wiki_path.name}'
                print(f"  ✓ {r.source.name:30s} ({r.modality}) {arrow}")
    elif sub == 'list':
        for path in sorted(wiki_dir().iterdir()):
            if path.is_file():
                size = path.stat().st_size
                print(f"  {path.name:40s}  {size:>8d} bytes")
    elif sub == 'ask':
        asyncio.run(personal_ask(args.prompt))


def main() -> None:
    """Parse argv and dispatch the requested subcommand.

    Subcommands
    -----------
    chat
        Start an interactive REPL (``chat_repl``).
    ask PROMPT
        Send one prompt and print the answer (``single_prompt``).
    personal SUBCMD
        Manage the personal knowledge base. Subcommands: ``status``,
        ``ingest`` (``--force`` re-runs everything), ``list``,
        ``ask PROMPT``.
    """
    parser = argparse.ArgumentParser(description="Roitelet LLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode.
    subparsers.add_parser("chat", help="Start an interactive chat REPL session")

    # Single execution mode.
    ask_parser = subparsers.add_parser("ask", help="Send a single question and get an answer")
    ask_parser.add_argument("prompt", help="The string prompt to evaluate")

    # Personal mode.
    personal_parser = subparsers.add_parser(
        "personal", help="Manage the personal knowledge base (RAG + Karpathy wiki)"
    )
    personal_sub = personal_parser.add_subparsers(dest="personal_command")
    personal_sub.add_parser("status", help="Show inbox/wiki counts + context mode")
    ingest_parser = personal_sub.add_parser("ingest", help="Convert inbox files to wiki entries")
    ingest_parser.add_argument("--force", action="store_true",
                               help="Re-ingest every inbox file, ignoring the manifest")
    personal_sub.add_parser("list", help="List wiki files")
    personal_ask_parser = personal_sub.add_parser(
        "ask", help="Ask a question with the personal knowledge base injected"
    )
    personal_ask_parser.add_argument("prompt", help="The question")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(chat_repl())
    elif args.command == "ask":
        asyncio.run(single_prompt(args.prompt))
    elif args.command == "personal":
        if not args.personal_command:
            personal_parser.print_help()
        else:
            personal_dispatch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
