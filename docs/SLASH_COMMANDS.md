# Slash Commands

Roitelet recognises a small set of leading slash commands that act as
**per-turn overrides** without bloating `ChatRequest` with new fields.
Type them at the start of a prompt; the rest of the prompt is forwarded
to the right pipeline with the overrides applied.

> Parsing is **leading-only** and **fail-soft**: only commands at the
> start of the prompt are recognised, and unknown commands are passed
> through as literal text (so a typo never silently changes behaviour).

---

## Catalogue

| Command | What it does |
|---|---|
| `/image <prompt>` | Generate an image. Routes to the image-gen pipeline (K=1, no fusion). On `/api/chat` this returns a 400 telling the client to use `POST /api/images`. |
| `/speech` | Speech-to-text + diarisation only. Requires an audio attachment via `POST /api/chat/multimodal`. Bypasses the LLM pipeline. |
| `/local <prompt>` | Force independence mode (local OSS models only) for this turn. Equivalent to `preferences.independence=true`. |
| `/cheap <usd> <prompt>` | Set `max_cost_usd` for this turn. Filters paid candidates above the budget *before* scoring. |
| `/k <n> <prompt>` | Override the top-K fan-out. Clamped to `[1, 8]`. |
| `/help` | Returns the catalogue as the assistant message. No fan-out, no Elo update, no telemetry. |

Aliases: `/image-gen` and `/img` for `/image`; `/stt` and `/transcribe`
for `/speech`.

---

## Composition

Override commands chain. The parser peels one command at a time
left-to-right; routing commands (`/image`, `/speech`, `/help`)
short-circuit further parsing.

```
/local /cheap 0.001 refactor this module
   ↓
  independence=true, max_cost_usd=0.001, prompt="refactor this module"
```

```
/local /image a sunset
   ↓
  independence=true (still set), route=image, prompt="a sunset"
```

```
/cheap rent in this neighborhood
   ↓
  fail-soft: no numeric argument → command is left in the prompt as literal text
```

---

## Examples

### Native chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "/local /k 5 explain the GIL"}'
```

The router runs locally with K=5; the request body itself doesn't need
`preferences.independence` or `top_k=5` set explicitly.

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

### Help

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "/help"}'
# Returns the catalogue inline as the synthesis content.
```

---

## Why slash commands?

- Per-turn settings without expanding the API contract.
- Discoverable (`/help` lists everything).
- Composable for power users (`/local /cheap 0.001 ...`).
- Familiar UX (Slack, Discord, ChatGPT all use it).
- Fail-soft — typos are inert.

The full catalogue lives in `core/commands.py`; extending it is a
one-method addition plus a help-line tuple update.
