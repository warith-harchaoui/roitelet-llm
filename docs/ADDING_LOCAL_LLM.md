# Adding a local LLM to Roitelet (GGUF and beyond)

Roitelet's local-first stance means the router *prefers* local models
when ``independence`` mode is on or when the cost-budget regime kicks
in. This guide covers three ways to add a local model — from "easiest"
to "most control over the runtime."

| Path | Server | Best for |
|---|---|---|
| **A — Ollama Modelfile** | Ollama | Any GGUF, painless registration, live-discovery. **Recommended.** |
| **B — Ollama Hub pull** | Ollama | Curated models already on `ollama.com`. |
| **C — `llama-server` + OpenAI-compat** | `llama.cpp` directly | When you need tight control over context size, KV cache, GPU offload, or a Roitelet deployment that won't run Ollama. |

All three are valid. Pick the one that fits your tooling.

---

## Path A — bring your own GGUF via Ollama (recommended)

Ollama runs any GGUF file via a Modelfile. This is the path most users
want: zero new dependencies in Roitelet, automatic discovery, and the
same provider client (`OllamaClient`) the OSS bundle uses.

### Step 1 — Put the GGUF on disk

Download from Hugging Face (or wherever):

```bash
curl -L -o /models/my-model-q4.gguf \
  https://huggingface.co/<author>/<repo>/resolve/main/<file>.gguf
```

### Step 2 — Write a Modelfile

`Modelfile` (no extension):

```
FROM /models/my-model-q4.gguf

# Optional: pin context length, sampling defaults, system prompt.
PARAMETER num_ctx 8192
PARAMETER temperature 0.2

# Optional: hint Ollama at the chat template if the GGUF metadata isn't enough.
# TEMPLATE """{{ .System }} {{ .Prompt }}"""
```

### Step 3 — Register with Ollama

```bash
ollama create my-model -f Modelfile
ollama list      # should show "my-model"
```

### Step 4 — That's it

Roitelet's live-discovery loop polls `/api/tags` every 60 s. Within
that window the new model appears in the routing pool as
`ollama/my-model`. No restart, no settings edit, no bootstrap change.

If you want the router to *trust* it immediately (rather than the
default 0.65 priors), add a bootstrap entry — see
[`ADDING_PAID_LLM.md`](ADDING_PAID_LLM.md#adding-richer-priors-for-a-new-model)
for the exact shape (pricing fields stay at 0 for local).

---

## Path B — pull a curated model from Ollama Hub

For models already in <https://ollama.com/library>:

```bash
ollama pull llama3.3:70b-instruct
ollama pull qwen3:8b
ollama pull phi4:latest
```

Same auto-discovery as Path A — the model shows up within the next 60 s
TTL window. The OSS quick-start bundle is exactly this:
`scripts/pull_defaults.sh` runs five `ollama pull`s.

---

## Path C — `llama-server` for direct control over the GGUF runtime

When Ollama's abstractions get in the way (custom context length per
model, fine-grained GPU offload, custom rope scaling, multiple GGUFs
of the same model with different quantisations served side-by-side),
talk to `llama.cpp` directly. `llama-server` ships an
OpenAI-compatible HTTP API on `localhost:8080` so Roitelet treats it
like any other paid provider — just one that happens to be free and
local.

### Step 1 — Start `llama-server`

```bash
# Assuming you've built llama.cpp (see https://github.com/ggerganov/llama.cpp)
llama-server -m /models/my-model-q4.gguf \
    -c 8192 \
    --port 8080 \
    --host 127.0.0.1
```

The server exposes `http://localhost:8080/v1/chat/completions` in
standard OpenAI shape.

### Step 2 — Point Roitelet at it

In the web UI's Settings sheet (or via `.env`):

```env
OPENAI_COMPATIBLE_BASE_URL=http://localhost:8080/v1
OPENAI_COMPATIBLE_API_KEY=any-non-empty-string
```

`llama-server` ignores the API key but the field must be non-empty —
the Roitelet registry skips registering candidates whose key is unset
(`registry._prune_unauthorized_remotes`).

### Step 3 — Register the model name

In Settings → "Paid OpenAI-compatible models", or via POST to
`/api/settings`:

```json
{
  "paid_openai_compatible_models": ["my-model"]
}
```

The router now considers `openai-compatible/my-model` on the next
prompt. To force-prefer it on cost (free, after all), set
`max_cost_usd=0` on the request preferences — the cost-budget regime
will exclude paid candidates and the local GGUF wins.

### Why Path C *isn't* the default

The Ollama path is dramatically simpler and supports the same GGUFs.
Path C exists for cases where:

- you can't run Ollama (some container environments, weird Linux
  distros),
- you need to serve a quantisation Ollama doesn't ship (very high or
  very low bit-widths),
- you want per-model context lengths that Ollama's templating layer
  collapses,
- you're embedding Roitelet inside a stack that already runs
  `llama-server`.

If none of those apply, Path A is strictly easier.

---

## Architecture map

The factory in `core/providers/factory.py` dispatches by `model_id` prefix:

```
ollama/<tag>              → OllamaClient → OLLAMA_BASE_URL
openai-compatible/<name>  → OpenAICompatibleClient → OPENAI_COMPATIBLE_BASE_URL
```

A local GGUF served by `llama-server` flows through the
`openai-compatible/...` branch — there is no separate "local
OpenAI-compat" path because the *protocol* is identical to a paid
provider. The only practical difference is the base URL.

---

## See also

- **[ADDING_PAID_LLM.md](ADDING_PAID_LLM.md)** — same pattern, applied to
  paid LLMs.
- **[MECHANISM.md](../MECHANISM.md)** — full architecture overview, including
  the model-source merge order in §4.
