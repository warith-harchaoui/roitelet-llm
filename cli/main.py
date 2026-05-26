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
    inbox_dir,
    ingest_inbox,
    personal_status,
    project_chunks_2d,
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
    elif sub == 'viz':
        _personal_viz(args.output)


def _personal_viz(output_path: str) -> None:
    """Project wiki chunks to 2-D and write a standalone HTML scatter.

    Parameters
    ----------
    output_path : str
        Destination file. Overwrites if it exists.
    """
    import json

    points = project_chunks_2d()
    if not points:
        print('No wiki chunks to visualise, or embedding model unreachable.')
        return
    # Embed the points directly into a single-file HTML so the user can
    # open it without serving anything. Avoids JS deps.
    html = _PERSONAL_VIZ_TEMPLATE.replace('"__POINTS__"', json.dumps(points))
    Path(output_path).write_text(html, encoding='utf-8')
    print(f'Wrote {len(points)} points to {output_path}')


_PERSONAL_VIZ_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Roitelet — personal wiki embedding viz</title>
<style>
  body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background: #fafafa; color: #1c1c1e; }
  header { padding: 16px 24px; border-bottom: 1px solid #e5e5ea; background: white; }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header p { margin: 4px 0 0; font-size: 12px; color: #6e6e73; }
  #chart { width: 100vw; height: calc(100vh - 70px); }
  .dot { fill-opacity: 0.7; cursor: pointer; transition: fill-opacity 0.15s; }
  .dot:hover { fill-opacity: 1; }
  #tooltip {
    position: absolute; pointer-events: none; max-width: 360px;
    background: rgba(0,0,0,0.85); color: white; padding: 8px 12px;
    font-size: 12px; line-height: 1.4; border-radius: 6px;
    display: none; z-index: 10;
  }
</style></head><body>
<header>
  <h1>Personal wiki — 2-D embedding projection</h1>
  <p>Each dot is one chunk. Color = source file. Spatial proximity reflects semantic similarity (PCA of nomic-embed-text vectors).</p>
</header>
<svg id="chart"></svg>
<div id="tooltip"></div>
<script>
const points = "__POINTS__";
const palette = ['#0a84ff','#ff9500','#34c759','#ff3b30','#af52de','#5856d6','#ff2d55','#5ac8fa','#ffcc00','#a2845e'];
const fileToColor = new Map();
function colorFor(path) {
  if (!fileToColor.has(path)) fileToColor.set(path, palette[fileToColor.size % palette.length]);
  return fileToColor.get(path);
}
const svg = document.getElementById('chart');
const tooltip = document.getElementById('tooltip');
function render() {
  const W = window.innerWidth, H = window.innerHeight - 70;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.innerHTML = '';
  if (!points.length) return;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const margin = 40;
  const sx = x => margin + (W - 2*margin) * (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin));
  const sy = y => H - margin - (H - 2*margin) * (ymax === ymin ? 0.5 : (y - ymin) / (ymax - ymin));
  for (const p of points) {
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', sx(p.x)); c.setAttribute('cy', sy(p.y));
    c.setAttribute('r', 6); c.setAttribute('fill', colorFor(p.path));
    c.setAttribute('class','dot');
    c.addEventListener('mousemove', e => {
      tooltip.style.display = 'block';
      tooltip.style.left = (e.pageX + 12) + 'px';
      tooltip.style.top = (e.pageY + 12) + 'px';
      tooltip.innerHTML = '<b>' + p.path + '</b> #' + p.chunk_index + '<br>' +
        p.text.slice(0, 240).replace(/</g,'&lt;') + (p.text.length > 240 ? '…' : '');
    });
    c.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    svg.appendChild(c);
  }
}
window.addEventListener('resize', render); render();
</script></body></html>
"""


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
    viz_parser = personal_sub.add_parser(
        "viz", help="Write a Karpathy-style 2-D embedding scatter to a standalone HTML file",
    )
    viz_parser.add_argument(
        "--output", default="personal-viz.html",
        help="HTML file to write (default: personal-viz.html in the cwd)",
    )

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
