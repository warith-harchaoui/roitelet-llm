# Adding a paid LLM to Roitelet

Roitelet fuses K parallel answers — diversity beats raw capability of any
one model. Adding a paid provider to the routing pool gives the local
OSS bundle (Qwen, Llama, Gemma, Phi) a stronger sibling to fuse with.

There are **two paths** depending on the provider:

| If your provider… | Use this path |
|---|---|
| Has an `/v1/chat/completions` endpoint (OpenAI-compatible) | **Path A — generic OpenAI-compat** (this doc) |
| Native API only (no OpenAI-compat layer) | **Path B — route through OpenRouter** (this doc) |

The OpenAI-compatible path is the universal one: Mistral, Together,
Groq, Anyscale, DeepInfra, Fireworks, Perplexity, `llama-server` (from
`llama.cpp`), and direct OpenAI all expose this shape. If your provider
ships an OpenAI-compatible URL, prefer Path A — it's zero-bootstrap,
zero-restart once configured.

---

## Path A — any OpenAI-compatible provider (universal)

### Step 1 — Configure the endpoint

In the web UI's Settings sheet, click **+ Add engine** under the
"OpenAI-compatible engines" card and fill it in. Equivalent
`/api/settings` payload (Mistral as the worked example — substitute
your provider):

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
`openai-compatible/<label>/<model>` — the label namespacing means two
engines can serve the same model name (e.g. Together and Anyscale
both serving `mistralai/Mistral-7B`) without collision. The router
considers them on the next prompt — no restart needed. Conservative
default priors (`coding=writing=…=0.65`, `input_per_1k=$0.002`)
apply until you either (a) let the rolling-Elo loop adjust them
through real use, or (b) hard-code richer priors in
`data/bootstrap/model_priors.json`.

That's it. The factory wires
`openai-compatible/<label>/<model>` requests to the configured
base URL + key via `core/providers/openai_compatible.py` and the
multi-engine dispatch in `core/providers/factory.py`.

### Worked example — direct OpenAI

Direct OpenAI is a special case of Path A and ships pre-configured:

1. `OPENAI_API_KEY=sk-proj-...` (env or web UI Settings).
2. Restart Roitelet.
3. `openai/gpt-4.1`, `openai/gpt-4o`, `openai/gpt-4o-mini` already live
   in `data/bootstrap/model_priors.json` — the router considers them on
   the next prompt.

To add a brand-new OpenAI model that isn't in the bootstrap yet, either
edit the bootstrap entry (best — has accurate priors) or add it via the
generic OpenAI-compat list (works without an edit, uses defaults).

---

## Path B — route any frontier model through OpenRouter

For native-API-only providers (today: Anthropic, Gemini, Cohere when
talking through their own SDKs), Roitelet doesn't ship dedicated
clients. The frictionless path is **OpenRouter**, which exposes every
mainstream model behind an OpenAI-compatible relay:

1. Get an OpenRouter key — <https://openrouter.ai/keys>.
2. Set `OPENROUTER_API_KEY=sk-or-v1-...`.
3. Roitelet bootstrap already includes
   `openrouter/anthropic/claude-3.7-sonnet`,
   `openrouter/google/gemini-2.5-pro`,
   `openrouter/deepseek/deepseek-r1`, and
   `openrouter/meta-llama/llama-3.3-70b-instruct`. Add more via the
   Settings sheet → "Paid OpenRouter models".

This costs slightly more per token (OpenRouter takes a relay cut) but
is the lowest-effort way to reach any frontier model from Roitelet
today.

---

## Architecture map

Roitelet's routing pool is the union of these sources (see
[docs/ARCHITECTURE.md](ARCHITECTURE.md) §4):

1. **Bootstrap priors** — `data/bootstrap/model_priors.json`.
2. **User-configured models** — three sources in `AppSettingsPayload`:
   * `selected_ollama_models` (local),
   * `paid_openrouter_models` (Path B),
   * `custom_engines[*].models` (Path A — one entry per OpenAI-compatible engine, each carrying its own base URL and API key).
3. **Live Ollama discovery** — auto-detected via `/api/tags` every 60 s.

The mapping from `model_id` prefix → provider client lives in
[`core/providers/factory.py`](../core/providers/factory.py):

| Prefix | Provider client | Endpoint source |
|---|---|---|
| `ollama/...` | `OllamaClient` | `OLLAMA_BASE_URL` |
| `openai/...` | `OpenAICompatibleClient` | OpenAI direct (`https://api.openai.com/v1`) |
| `openrouter/...` | `OpenAICompatibleClient` | OpenRouter (`OPENROUTER_BASE_URL`) |
| `openai-compatible/...` | `OpenAICompatibleClient` | **whatever** `OPENAI_COMPATIBLE_BASE_URL` points at |

A registered model is *only* invoked when the matching API key is set.
Unset keys cause the registry to auto-prune the corresponding entries
(`registry._prune_unauthorized_remotes`) so the router never crowns a
candidate whose provider call will 401.

---

## Adding richer priors for a new model

Default priors are deliberately conservative (every capability at 0.65)
so an unknown model can't immediately win every routing call. Once you
know the model is actually strong on, say, math, give it a bootstrap
entry with real numbers:

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
rolling Elo loop adjusts each capability per actual win/loss, so even
a wrong prior self-corrects after ~50 turns.

---

## Sanity-checking

After adding a key + restarting, verify the model is in the registered
pool:

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

You'll see `roitelet-llm` (the routed virtual model). To inspect the
*actual* candidate pool used for a real prompt, send one and look at
the telemetry:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quicksort in two sentences."}' | jq '.responses[].model_id'
```

If you see an `openai-compatible/...` (or `openai/...`, `openrouter/...`)
id, your key is wired correctly and the router picked the model. If you
don't, check that:

1. The relevant API key is set (`GET /api/settings` shows `••••••••`
   for stored keys — the mask is intentional and round-trips safely).
2. Your preferences don't force-filter the model:
   * `independence=true` removes all non-local candidates,
   * `max_cost_usd=X` removes candidates whose pricing exceeds `X`.

---

## Security note

API keys are masked when read back from `/api/settings` (web UI sees
`••••••••`, the on-disk value stays intact). Don't commit `.env` to
version control — `.gitignore` already excludes it. For LAN deployments,
set `ROITELET_API_TOKEN` so the settings endpoints require a Bearer
token; see `.env.example`.

---

## See also

- **[ADDING_LOCAL_LLM.md](ADDING_LOCAL_LLM.md)** — bring your own GGUF
  file, either through Ollama or through `llama-server`.
- **[docs/ARCHITECTURE.md](ARCHITECTURE.md)** — the full architecture deep-dive.
