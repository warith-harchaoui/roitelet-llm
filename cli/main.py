"""Command-line interface for Roitelet LLM.

Provides terminal access to every Roitelet operation the API and the
web UI expose. The CLI mirrors gemini-cli's surface plus
Roitelet-specific subcommands:

* ``roitelet chat``                          — interactive REPL.
* ``roitelet ask PROMPT``                    — single-shot question.
* ``roitelet personal SUBCMD``               — knowledge-base management.
* ``roitelet settings get [KEY]``            — print persisted settings.
* ``roitelet settings set KEY VALUE``        — edit persisted settings.

Both ``ask`` and ``chat`` honour the same slash commands as the API
(``/local``, ``/cheap``, ``/k``, ``/personal``, ``/pseudo``,
``/nopseudo``, ``/help``) so a user can paste a prompt that worked
in the web UI directly into the terminal. They also expose explicit
flags (``--top-k``, ``--independence``, ``--ecofrugality``,
``--max-cost-usd``, ``--pseudonymize`` / ``--no-pseudonymize``) for
scripted use cases where parsing a slash prefix is awkward — flag
values override slash overrides override request defaults.

Examples
--------
>>> # Run a single-shot question:
>>> #   roitelet ask "Explain quicksort"
>>> # Pseudonymize per turn:
>>> #   roitelet ask --pseudonymize "Email Marie Dupont in Lyon."
>>> # Equivalent with a slash:
>>> #   roitelet ask "/pseudo Email Marie Dupont in Lyon."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from core.commands import parse_command, render_help
from core.personal import (
    build_personal_context,
    inbox_dir,
    ingest_inbox,
    personal_status,
    project_chunks_2d,
    wiki_dir,
)
from core.pipeline import run_roitelet_chat
from core.schemas import (
    AppSettingsPayload,
    ChatRequest,
    ChatResponse,
    RouterPreferences,
)
from core.storage import get_storage


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
    print("Type /help inside the REPL to see slash commands. 'exit' or 'quit' to end.\n")


def _build_request_from_args(
    prompt: str,
    args: argparse.Namespace,
) -> ChatRequest:
    """Compose a :class:`ChatRequest` from a prompt + parsed CLI args.

    Honours the same slash-command catalogue as the API so the CLI is
    fully at parity. Explicit ``--`` flags win over slash overrides.

    The settings-derived persistent defaults (``enable_pseudonymization``)
    are read here so the user doesn't have to repeat them on every turn.

    Parameters
    ----------
    prompt : str
        The raw user prompt — may carry leading slash commands.
    args : argparse.Namespace
        Output of the ``ask`` or ``chat`` argument parser. The flags
        ``--top-k``, ``--independence`` / ``--remote``,
        ``--ecofrugality``, ``--max-cost-usd``, and
        ``--pseudonymize`` / ``--no-pseudonymize`` are recognised when
        present.

    Returns
    -------
    ChatRequest
        Ready to be dispatched to :func:`run_roitelet_chat`.
    """
    settings = get_storage().load_app_settings()

    parsed = parse_command(prompt)
    # Personal mode prepends the wiki context block (same logic as the
    # ``/api/chat`` route in ``api/main.py``).
    if parsed.personal_override:
        base = parsed.stripped_prompt or prompt
        ctx = build_personal_context(base)
        effective_prompt = f'{ctx}\n## Question\n\n{base}' if ctx else base
    else:
        effective_prompt = parsed.stripped_prompt or prompt

    # Default preferences read the persisted settings; CLI flags + slash
    # overrides then layer on top.
    prefs = RouterPreferences(
        raw_power=settings.raw_power_weight,
        ecofrugality=settings.ecofrugality_weight,
        independence=settings.independence_local_only,
        allow_vlms=settings.enable_vlms,
        pseudonymize=settings.enable_pseudonymization,
    )

    # Slash overrides (already parsed by parse_command).
    pref_updates: dict = {}
    if parsed.independence_override is not None:
        pref_updates['independence'] = parsed.independence_override
    if parsed.max_cost_usd_override is not None:
        pref_updates['max_cost_usd'] = parsed.max_cost_usd_override
    if parsed.pseudonymize_override is not None:
        pref_updates['pseudonymize'] = parsed.pseudonymize_override

    # Explicit CLI flags — highest precedence. Use ``getattr`` because
    # the ``personal ask`` subcommand re-uses this helper but doesn't
    # carry every flag.
    if getattr(args, 'independence', None) is True:
        pref_updates['independence'] = True
    if getattr(args, 'remote', None) is True:
        pref_updates['independence'] = False
    if getattr(args, 'ecofrugality', None) is not None:
        pref_updates['ecofrugality'] = float(args.ecofrugality)
    if getattr(args, 'max_cost_usd', None) is not None:
        pref_updates['max_cost_usd'] = float(args.max_cost_usd)
    if getattr(args, 'pseudonymize', None) is True:
        pref_updates['pseudonymize'] = True
    if getattr(args, 'no_pseudonymize', None) is True:
        pref_updates['pseudonymize'] = False

    if pref_updates:
        prefs = prefs.model_copy(update=pref_updates)

    top_k: int = parsed.top_k_override or getattr(args, 'top_k', None) or 2

    return ChatRequest(
        prompt=effective_prompt,
        preferences=prefs,
        top_k=top_k,
    )


def _render_response(response: ChatResponse, verbose: bool) -> None:
    """Print the synthesis and (optionally) the audit affordances.

    Parameters
    ----------
    response : ChatResponse
        Result of :func:`run_roitelet_chat`.
    verbose : bool
        When ``True``, also print the pseudonymization audit (if any)
        and a one-line router summary so the user can see what
        actually ran.
    """
    print(response.synthesis.content)
    if not verbose:
        return
    if response.pseudonymization is not None:
        audit = response.pseudonymization
        print('\n──── pseudonymization audit ────')
        print(f'model: {audit.model_id}')
        print(f'sent to remote candidates: {audit.pseudonymized_prompt}')
        if audit.mappings:
            print('substitutions (original → substitute):')
            for mapping in audit.mappings:
                print(f'  [{mapping.kind:>18s}]  {mapping.original}  →  {mapping.substitute}')
        else:
            print('substitutions: none (no PII detected)')
        print(
            f'forward {audit.forward_latency_s:.2f}s · '
            f'reverse {audit.reverse_latency_s:.2f}s · '
            f"{'repair pass used' if audit.repair_used else 'literal reverse only'}"
        )
    if response.router is not None:
        selected = ', '.join(response.router.selected_model_ids) or '(none)'
        print(f'\nrouter selected: {selected}  ·  total {response.total_latency_s:.2f}s')


async def chat_repl(args: argparse.Namespace) -> None:
    """Run the interactive chat loop until ``exit``/``quit``/EOF.

    Each iteration reads a line from stdin, sends it through the
    Roitelet pipeline with the CLI flags + slash commands applied,
    and prints the fused synthesis. ``Ctrl+C`` and ``Ctrl+D`` both
    exit cleanly.

    Notes
    -----
    Conversation IDs are reused turn-to-turn so the session forms a
    single conversation in the persisted log. Exceptions raised by
    ``run_roitelet_chat`` (provider failures, judge fallback,
    pseudonymizer fail-closed, etc.) are caught per-iteration so a
    single bad prompt doesn't kill the session — the error message
    is printed and the loop continues.
    """
    print_welcome()
    conversation_id: str | None = None
    while True:
        try:
            prompt = input("You> ")
            clean_prompt = prompt.strip()
            if clean_prompt.lower() in ('exit', 'quit'):
                break
            if not clean_prompt:
                continue

            request = _build_request_from_args(clean_prompt, args)
            request = request.model_copy(update={'conversation_id': conversation_id})
            response = await run_roitelet_chat(request)
            conversation_id = response.conversation_id

            print(f"\nRoitelet> {response.synthesis.content}\n")
            if args.verbose and response.pseudonymization is not None:
                _render_response(response, verbose=True)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as exc:
            print(f"\nError: {exc}\n")


async def single_prompt(prompt: str, args: argparse.Namespace) -> None:
    """Execute exactly one prompt and exit.

    Parameters
    ----------
    prompt : str
        The user's prompt sent through :func:`run_roitelet_chat`,
        after slash-command + CLI-flag layering.
    args : argparse.Namespace
        Parsed CLI args (preference flags + ``--verbose``).

    Notes
    -----
    On any pipeline failure the error is printed to stdout and the
    process exits with status 1, matching ``ask``-style CLIs.
    Pseudonymizer fail-closed errors land here too — that is the
    intended behaviour, never silently send the unredacted prompt.
    """
    try:
        request = _build_request_from_args(prompt, args)
        response = await run_roitelet_chat(request)
        _render_response(response, verbose=args.verbose)
    except Exception as exc:
        print(f"Error processing prompt: {exc}")
        sys.exit(1)


def personal_dispatch(args: argparse.Namespace) -> None:
    """Handle the ``personal`` subcommand family.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``args.personal_command`` is one of
        ``'status'``, ``'ingest'``, ``'list'``, ``'ask'``, ``'viz'``.
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
        # ``personal ask`` is a thin wrapper around ``ask`` with the
        # personal-override forced on. Re-uses _build_request_from_args
        # so all the same flag plumbing applies (--pseudonymize works
        # here too).
        prompt = f'/personal {args.prompt}'
        asyncio.run(single_prompt(prompt, args))
    elif sub == 'viz':
        _personal_viz(args.output)


def _personal_viz(output_path: str) -> None:
    """Project wiki chunks to 2-D and write a standalone HTML scatter.

    Parameters
    ----------
    output_path : str
        Destination file. Overwrites if it exists.
    """
    points = project_chunks_2d()
    if not points:
        print('No wiki chunks to visualise, or embedding model unreachable.')
        return
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
  <p>Each dot is one chunk. Color = source file.
     Spatial proximity reflects semantic similarity
     (PCA of nomic-embed-text vectors).</p>
</header>
<svg id="chart"></svg>
<div id="tooltip"></div>
<script>
const points = "__POINTS__";
// Aligned to https://harchaoui.org/warith/colors — Red, Orange, Yellow,
// Green, Blue, Turquoise, Purple, Pink (+ neutral Gray to close the cycle).
const palette = ['#007AFF','#FF9500','#FFCC00','#28CD41','#79DBDC','#AF52DE','#FF2D55','#FF3B30','#808080'];
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


def settings_dispatch(args: argparse.Namespace) -> None:
    """Handle the ``settings`` subcommand family.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``args.settings_command`` is one of
        ``'get'`` or ``'set'``.

    Notes
    -----
    Editing the settings via the CLI hits the exact same
    :class:`AppSettingsPayload` round-trip as the web UI. Secret
    fields are masked on ``get`` for screenshot / paste safety; pass
    ``--show-secrets`` to disable masking. ``set`` accepts JSON
    literals so ``custom_engines`` and other structured fields can be
    edited too.
    """
    sub = args.settings_command
    storage = get_storage()
    settings = storage.load_app_settings()

    if sub == 'get':
        payload = settings if args.show_secrets else settings.masked()
        if args.key:
            value = getattr(payload, args.key, None)
            if value is None and args.key not in payload.model_fields:
                print(f'Unknown setting: {args.key}', file=sys.stderr)
                sys.exit(1)
            print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
        else:
            print(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False, default=str))
        return

    if sub == 'set':
        if args.key not in AppSettingsPayload.model_fields:
            print(f'Unknown setting: {args.key}', file=sys.stderr)
            sys.exit(1)
        try:
            parsed_value = json.loads(args.value)
        except json.JSONDecodeError:
            # Bare strings, numbers, etc. that the user typed without
            # JSON quotes — accept them as a string literal.
            parsed_value = args.value
        updated = settings.model_copy(update={args.key: parsed_value})
        # model_validate enforces the field's type even though
        # model_copy bypasses validation, so we re-validate explicitly.
        validated = AppSettingsPayload.model_validate(updated.model_dump())
        storage.save_app_settings(validated)
        print(f'Saved {args.key} = {json.dumps(parsed_value, ensure_ascii=False, default=str)}')


def _add_pref_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the per-turn preference flags shared by ``ask`` and ``chat``.

    Keeping the flag wiring in one place ensures the two entry points
    accept identical syntax — a property the docs and the tests both
    rely on.
    """
    parser.add_argument('--top-k', dest='top_k', type=int, default=None,
                        help='Fan-out width (1–8). Default: persisted setting → 2.')
    parser.add_argument('--independence', action='store_true', default=None,
                        help='Local-only mode for this run (drops remote candidates).')
    parser.add_argument('--remote', action='store_true', default=None,
                        help='Force-disable local-only mode (override the persisted default).')
    parser.add_argument('--ecofrugality', type=float, default=None,
                        help='Ecofrugality weight (0..1) — blends low cost + low energy.')
    parser.add_argument('--max-cost-usd', dest='max_cost_usd', type=float, default=None,
                        help='Per-turn budget; candidates above this estimated cost are filtered.')
    parser.add_argument('--pseudonymize', action='store_true', default=None,
                        help='Swap PII before remote calls; restore on the way back. See PSEUDO.md.')
    parser.add_argument('--no-pseudonymize', dest='no_pseudonymize', action='store_true', default=None,
                        help='Force pseudonymization off for this turn even if the setting is on.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print the router decision + pseudonymization audit alongside the answer.')


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
        ``ingest`` (``--force`` re-runs everything), ``list``, ``ask
        PROMPT``, ``viz``.
    settings SUBCMD
        Inspect / edit persisted control-room settings. ``get [KEY]``
        prints a single field (or every field when KEY is omitted);
        ``set KEY VALUE`` accepts JSON literals.
    """
    parser = argparse.ArgumentParser(
        prog='roitelet',
        description='Roitelet LLM CLI — same operations as the web UI and the JSON API.',
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode.
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat REPL session")
    _add_pref_flags(chat_parser)

    # Single execution mode.
    ask_parser = subparsers.add_parser("ask", help="Send a single question and get an answer")
    ask_parser.add_argument("prompt", help="The string prompt to evaluate")
    _add_pref_flags(ask_parser)

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
    _add_pref_flags(personal_ask_parser)
    viz_parser = personal_sub.add_parser(
        "viz", help="Write a Karpathy-style 2-D embedding scatter to a standalone HTML file",
    )
    viz_parser.add_argument(
        "--output", default="personal-viz.html",
        help="HTML file to write (default: personal-viz.html in the cwd)",
    )

    # Settings management — full parity with the GUI Settings sheet.
    settings_parser = subparsers.add_parser(
        "settings", help="Inspect or edit persisted control-room settings"
    )
    settings_sub = settings_parser.add_subparsers(dest="settings_command")
    get_parser = settings_sub.add_parser("get", help="Print a setting (KEY) or every setting (no KEY)")
    get_parser.add_argument("key", nargs='?', default=None, help="Setting name (omit for all)")
    get_parser.add_argument("--show-secrets", action='store_true',
                            help="Print API keys in clear text (default: masked)")
    set_parser = settings_sub.add_parser("set", help="Edit one setting; value is a JSON literal or bare string")
    set_parser.add_argument("key", help="Setting name (e.g. enable_pseudonymization)")
    set_parser.add_argument("value", help="JSON literal: true / 0.5 / \"qwen3:8b\" / [\"a\",\"b\"]")

    # Convenience: print the slash-command catalogue without spawning a turn.
    subparsers.add_parser("help-slash", help="Print the slash-command catalogue (same as /help in chat).")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(chat_repl(args))
    elif args.command == "ask":
        asyncio.run(single_prompt(args.prompt, args))
    elif args.command == "personal":
        if not args.personal_command:
            personal_parser.print_help()
        else:
            personal_dispatch(args)
    elif args.command == "settings":
        if not args.settings_command:
            settings_parser.print_help()
        else:
            settings_dispatch(args)
    elif args.command == "help-slash":
        print(render_help())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
