# Adding models to Roitelet

Roitelet's routing pool is the union of:

1. **Bootstrap priors** — `data/bootstrap/model_priors.json` (shipped).
2. **User-configured models** in `AppSettingsPayload`:
   * `selected_ollama_models` (local Ollama),
   * `paid_openrouter_models` (any model OpenRouter relays),
   * `custom_engines[*].models` (any OpenAI-compatible endpoint).
3. **Live Ollama discovery** — auto-detected via `/api/tags` every
   60 s.

The factory in `core/providers/factory.py` dispatches by `model_id`
prefix:

| Prefix | Client | Endpoint |
|---|---|---|
| `ollama/...` | `OllamaClient` | `OLLAMA_BASE_URL` |
| `openai/...` | `OpenAICompatibleClient` | `https://api.openai.com/v1` |
| `openrouter/...` | `OpenAICompatibleClient` | `OPENROUTER_BASE_URL` |
| `openai-compatible/...` | `OpenAICompatibleClient` | custom-engine `base_url` |

A registered model is only invoked when its API key is set. Missing
keys cause `registry._prune_unauthorized_remotes` to drop the
candidate so the router never crowns a model whose provider call
will 401.

---

## Part 1 — Local models

Roitelet prefers local models when `independence` mode is on or
when a tight cost budget filters paid candidates out. Three paths,
from easiest to most controlled:

### Path A — Ollama Hub pull *(easiest, recommended)*

For models already in <https://ollama.com/library>:

```bash
ollama pull llama3.3:70b-instruct
ollama pull qwen3:8b
ollama pull phi4:latest
```

Roitelet's live-discovery loop polls `/api/tags` every 60 s; the
new model appears in the routing pool as `ollama/<name>` within
that window. No restart, no settings edit. The OSS quick-start
bundle (`scripts/pull_defaults.sh`) is exactly this — five
`ollama pull`s.

### Path B — bring your own GGUF via Ollama Modelfile

When the model is a GGUF not in the Ollama Hub.

1. **Put the GGUF on disk:**
   ```bash
   curl -L -o /models/my-model-q4.gguf \
     https://huggingface.co/<author>/<repo>/resolve/main/<file>.gguf
   ```
2. **Write a `Modelfile`:**
   ```
   FROM /models/my-model-q4.gguf
   PARAMETER num_ctx 8192
   PARAMETER temperature 0.2
   # TEMPLATE """{{ .System }} {{ .Prompt }}"""  # only if metadata is insufficient
   ```
3. **Register with Ollama:**
   ```bash
   ollama create my-model -f Modelfile
   ollama list  # should show "my-model"
   ```
4. **Done.** Discovery picks it up within 60 s as `ollama/my-model`.

If you want the router to *trust* it immediately (instead of the
default 0.65 priors), append a bootstrap entry — see [§3](#part-3--adding-richer-priors-for-a-new-model)
below.

### Path C — `llama-server` directly (full control)

When Ollama's abstractions get in the way (custom context length
per model, fine-grained GPU offload, custom rope scaling, multiple
GGUFs of the same model with different quantisations side-by-side),
talk to `llama.cpp` directly.

```bash
# Assumes llama.cpp is built (https://github.com/ggerganov/llama.cpp).
llama-server -m /models/my-model-q4.gguf \
    -c 8192 \
    --port 8080 \
    --host 127.0.0.1
```

`llama-server` ships an OpenAI-compatible HTTP API on
`localhost:8080`, so Roitelet treats it as any other custom engine.
Add it via the web UI → Settings → **+ Add engine**:

| Field | Value |
|---|---|
| Label | `llama-server` |
| Base URL | `http://localhost:8080/v1` |
| API key | `any-non-empty-string` (`llama-server` ignores it but the field must be non-empty, else the registry drops the candidate) |
| Models | `my-model` (comma-separated for multiple) |

The router now considers `openai-compatible/llama-server/my-model`
on the next prompt. To force-prefer it on cost (free, after all),
set `max_cost_usd=0` on the request — the cost-budget regime
excludes paid candidates and the local GGUF wins.

**Why Path C isn't the default:** Path A is dramatically simpler
and supports the same GGUFs. Path C exists for environments where
Ollama doesn't run, for quantisations Ollama doesn't ship, or for
existing stacks already running `llama-server`.

---

## Part 2 — Paid / cloud models

Two paths, mirroring the local section:

### Path A — any OpenAI-compatible provider *(universal)*

Mistral, Together, Groq, Anyscale, DeepInfra, Fireworks, Perplexity,
direct OpenAI — all expose `/v1/chat/completions`. Add via Settings
sheet → **+ Add engine**. Equivalent `/api/settings` payload (Mistral
as the worked example):

```json
{
  "custom_engines": [
    {
      "label": "mistral",
      "base_url": "https://api.mistral.ai/v1",
      "api_key": "...",
      "models": ["mistral-large-latest", "mistral-medium"]
    }
  ]
}
```

Each entry becomes a routable model id
`openai-compatible/<label>/<model>`. Two engines can serve the
same model name (e.g. Together and Anyscale both serving
`mistralai/Mistral-7B`) without collision because of the label
namespace. Conservative default priors apply
(`coding=writing=…=0.65`, `input_per_1k=$0.002`) until the rolling
Elo adjusts them through use, or you hard-code richer ones (§3).

#### Worked example — direct OpenAI

Direct OpenAI is a special case and ships pre-configured:

1. `OPENAI_API_KEY=sk-proj-...` (env var or web UI Settings).
2. Restart.
3. `openai/gpt-4.1`, `openai/gpt-4o`, `openai/gpt-4o-mini` are
   already in `data/bootstrap/model_priors.json`.

### Path B — route through OpenRouter

For native-API-only providers (Anthropic, Gemini, Cohere when
talking through their own SDKs), use OpenRouter as a relay.

1. Get a key — <https://openrouter.ai/keys>.
2. Set `OPENROUTER_API_KEY=sk-or-v1-...`.
3. Roitelet bootstrap already includes
   `openrouter/anthropic/claude-3.7-sonnet`,
   `openrouter/google/gemini-2.5-pro`,
   `openrouter/deepseek/deepseek-r1`,
   `openrouter/meta-llama/llama-3.3-70b-instruct`. Add more via
   Settings sheet → "Paid OpenRouter models".

OpenRouter takes a relay cut on top of provider pricing, but it's
the lowest-effort way to reach any frontier model from Roitelet
today.

---

## Part 3 — Adding richer priors for a new model

Default priors are conservative (every capability at 0.65) so an
unknown model can't immediately win every routing call. Once you
know the model is strong on, say, math, append a bootstrap entry
with real numbers:

```json
"openai-compatible/mistral-large-latest": {
  "provider": "openai-compatible",
  "local": false,
  "vlm": false,
  "pricing": {"input_per_1k": 0.002, "output_per_1k": 0.006},
  "latency_s": 3.6,
  "energy_kwh": 0.00055,
  "capabilities": {
    "coding": 0.88, "math": 0.89, "reasoning": 0.91,
    "writing": 0.90, "analysis": 0.87,
    "multilingual": 0.92, "long_context": 0.85
  }
}
```

Append to `data/bootstrap/model_priors.json`, restart, done. The
rolling Elo loop adjusts each capability per actual win/loss, so
even a wrong prior self-corrects after ~50 turns.

For **local** models, set `provider: "ollama"`, `local: true`, and
keep pricing at zero.

---

## Sanity-checking

After adding a key + restarting, verify the model is in the
registered pool:

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

To inspect the actual candidate pool used for a real prompt:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quicksort in two sentences."}' | jq '.responses[].model_id'
```

If you see an `openai-compatible/...` (or `openai/...`,
`openrouter/...`) id, your key is wired correctly and the router
picked the model. If you don't, check:

1. The relevant API key is set (`GET /api/settings` shows
   `••••••••` for stored keys — the mask is intentional and
   round-trips safely).
2. Your preferences don't force-filter the model:
   * `independence=true` removes all non-local candidates,
   * `max_cost_usd=X` removes candidates whose pricing exceeds X.

---

## Security note

API keys are masked when read back from `/api/settings` (web UI sees
`••••••••`, the on-disk value stays intact). Don't commit `.env` to
version control — `.gitignore` already excludes it. For LAN
deployments, set `ROITELET_API_TOKEN` so the settings endpoints
require a Bearer token; see `.env.example`.

---

## See also

- **[docs/ARCHITECTURE.md](ARCHITECTURE.md)** — full architecture deep-dive.
- **[docs/OPENAI_COMPAT.md](OPENAI_COMPAT.md)** — use Roitelet from
  existing OpenAI-shaped tooling (LiteLLM, Continue.dev, …).
- **[docs/EVALUATION.md](EVALUATION.md)** — measure whether your
  newly-added model actually helps the fusion judge.
