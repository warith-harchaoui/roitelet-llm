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


def main() -> None:
    """Parse argv and dispatch the requested subcommand.

    Subcommands
    -----------
    chat
        Start an interactive REPL (``chat_repl``).
    ask PROMPT
        Send one prompt and print the answer (``single_prompt``).
    """
    parser = argparse.ArgumentParser(description="Roitelet LLM CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode.
    subparsers.add_parser("chat", help="Start an interactive chat REPL session")

    # Single execution mode.
    ask_parser = subparsers.add_parser("ask", help="Send a single question and get an answer")
    ask_parser.add_argument("prompt", help="The string prompt to evaluate")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(chat_repl())
    elif args.command == "ask":
        asyncio.run(single_prompt(args.prompt))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
