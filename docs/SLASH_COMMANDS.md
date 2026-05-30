# Slash commands

Slash commands here mean **route selection** — picking a different
pipeline from the one that handles plain text. Per-turn preferences
(top-K, local-only, pseudonymize, max cost) are NOT slash-typed; they
live on visible controls so you can see what's active without
re-reading the prompt:

| Surface | Per-turn control |
|---|---|
| Web composer | Sliders icon next to the send button (small blue dot when any non-default is set). |
| CLI | `--top-k`, `--independence` / `--remote`, `--ecofrugality`, `--max-cost-usd`, `--pseudonymize` / `--no-pseudonymize` on `roitelet ask` and `roitelet chat`. |
| API | Booleans on `preferences` in `POST /api/chat`, or the matching `Form` fields on `POST /api/chat/multimodal`. |

Slashes are reserved for routes because routes change which endpoint
processes the prompt, and that is not something a checkbox can express.

---

## Catalogue

| Command | What it does |
|---|---|
| `/image <prompt>` | Generate an image. Routes to the image-gen pipeline (K=1, no fusion). On `/api/chat` this returns a 400 telling the client to use `POST /api/images`. |
| `/speech` | Speech-to-text + diarisation only. Requires an audio attachment via `POST /api/chat/multimodal`. Bypasses the LLM pipeline. |
| `/personal <prompt>` | Inject your personal knowledge base (`data/personal/wiki/`) into the prompt. See [PERSONAL_MODE.md](PERSONAL_MODE.md). |
| `/help` | Returns the catalogue as the assistant message. No fan-out, no Elo update, no telemetry. |

Aliases: `/image-gen` and `/img` for `/image`; `/stt` and `/transcribe`
for `/speech`.

> Parsing is **leading-only** and **fail-soft**: only one slash at the
> start of the prompt is recognised, and unknown commands pass through
> as literal text (a typo never silently changes behaviour). The retired
> per-turn slashes — `/local`, `/cheap`, `/k`, `/pseudo`, `/nopseudo` —
> now pass through verbatim so muscle memory doesn't trigger silent
> behaviour changes.

---

## Examples

### Image generation

```bash
# Wrong endpoint → 400 with a pointer:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "/image a wren in oil paint"}'
# → {"detail": {"error": "wrong_endpoint", ..., "route_to": "image"}}

# Right endpoint:
curl -X POST http://localhost:8000/api/images \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a wren in oil paint", "size": "1024x1024"}'
```

### Per-turn preferences (no slash)

```bash
# Native API — set the booleans on `preferences`:
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Refactor this module.",
    "preferences": {"independence": true, "pseudonymize": true, "max_cost_usd": 0.001},
    "top_k": 5
  }'
```

```bash
# CLI — flags are the visible-state equivalent:
roitelet ask --independence --pseudonymize --max-cost-usd 0.001 --top-k 5 \
  "Refactor this module."
```

```text
# Web — open the sliders popover next to the send button and tick the
# matching boxes. A blue dot appears on the icon while any non-default
# is set; "New chat" resets per-turn prefs to your persisted defaults.
```

### Help

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "/help"}'
# Returns the catalogue inline as the synthesis content.
```

---

## Why slashes are routes only

- **Visible state beats invisible state.** A user who flipped `/local`
  three turns ago shouldn't have to scroll up to remember. A checkbox
  with a blue dot says it at a glance.
- **No catalogue inflation.** Every new preference used to need a slash
  *and* an "off" slash (`/pseudo` / `/nopseudo`). The popover scales by
  adding a row; the slash catalogue stayed at four.
- **Discoverability.** A new user finds a sliders icon by looking at
  the composer; they would not find `/cheap 0.001` without reading
  the docs.

The slash catalogue lives in `core/commands.py::HELP_LINES` and the
preference surface mapping in `core/commands.py::PREFERENCE_SURFACES`.
