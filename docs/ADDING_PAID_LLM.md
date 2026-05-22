# Adding a paid LLM to Roitelet

Roitelet fuses K parallel answers — diversity beats raw capability of any
one model. Adding a paid provider to the routing pool gives the local
OSS bundle (Qwen, Llama, Gemma, Phi) a stronger sibling to fuse with.

This guide walks through **adding ChatGPT** as the worked example. The
same pattern works for any provider with an OpenAI-compatible HTTP API.

---

## TL;DR — three steps

1. **Get an API key** from the provider (here: <https://platform.openai.com/api-keys>).
2. **Tell Roitelet about it** — paste the key into the Settings sheet
   in the web UI, OR put it in `.env`:
   ```env
   OPENAI_API_KEY=sk-proj-...
   ```
3. **Restart Roitelet.** That's it — `openai/gpt-4.1`, `openai/gpt-4o`,
   and `openai/gpt-4o-mini` are already in `data/bootstrap/model_priors.json`,
   so the router will start considering them on the next prompt.

---

## What's happening under the hood

Roitelet's routing pool is the union of three sources (see
[MECHANISM.md](../MECHANISM.md) §4):

1. **Bootstrap priors** — `data/bootstrap/model_priors.json`. Curated
   capability scores per model. The three GPT-4* entries above are there
   out of the box.
2. **User-configured models** — `selected_ollama_models` /
   `paid_openrouter_models` saved from the web UI.
3. **Live Ollama discovery** — auto-detected via `/api/tags`.

When the router rebuilds candidates for each prompt, it pulls everything
from those three sources, then filters by your preferences (raw power vs.
frugality vs. independence). A registered model is *only* invoked when
its provider's API key is set; an unset key means the provider's clients
fail closed and the candidate falls back to the next-best.

The mapping from `model_id` prefix → provider client lives in
[`core/providers/factory.py`](../core/providers/factory.py). The current
branches:

| Prefix | Provider | API key |
|---|---|---|
| `ollama/...` | Local Ollama | none |
| `openai/...` | Direct OpenAI | `OPENAI_API_KEY` |
| `openrouter/...` | OpenRouter relay | `OPENROUTER_API_KEY` |
| `openai-compatible/...` | Any OpenAI-compatible endpoint | `OPENAI_COMPATIBLE_API_KEY` + `_BASE_URL` |

---

## Adding a model that isn't in the bootstrap yet

Suppose OpenAI releases `gpt-5` next month and you want it in your
routing pool *before* the maintainers update `model_priors.json`.

Two options:

### Option 1 — quick (no bootstrap edit needed)

The router's "user-configured models" source picks up anything you stash
in `paid_openrouter_models` via the web UI. For non-OpenRouter providers
that path doesn't exist yet (see decision point in
[`ASSESSMENTS.md`](../.private/ASSESSMENTS.md)) — for now, use Option 2.

### Option 2 — add a bootstrap entry

Edit `data/bootstrap/model_priors.json` and append (note the leading
comma to extend the dict):

```json
"openai/gpt-5": {
  "provider": "openai",
  "local": false,
  "vlm": true,
  "pricing": {"input_per_1k": 0.01, "output_per_1k": 0.03},
  "latency_s": 4.0,
  "energy_kwh": 0.0007,
  "capabilities": {
    "coding": 0.96, "math": 0.92, "reasoning": 0.95,
    "writing": 0.92, "analysis": 0.93, "vision": 0.90,
    "multilingual": 0.90, "long_context": 0.95
  }
}
```

The capability numbers are priors — best guesses based on the model's
public benchmark scores. They're not load-bearing: Roitelet's rolling
Elo loop adjusts each capability per actual win/loss, so a wrong prior
self-corrects after ~50 turns.

Restart and the new model joins the candidate pool on the next prompt.

---

## Adding a provider that *isn't* OpenAI-compatible

The Anthropic and Gemini native APIs use different request/response
shapes than OpenAI. Until Roitelet ships dedicated clients for those
(tracked in [`ASSESSMENTS.md`](../.private/ASSESSMENTS.md)), the easiest
path for any non-OpenAI-compatible provider is **route through OpenRouter**:

1. Get an OpenRouter key — <https://openrouter.ai/keys>.
2. Set `OPENROUTER_API_KEY=sk-or-v1-...`.
3. Roitelet bootstrap already includes `openrouter/anthropic/claude-3.7-sonnet`,
   `openrouter/google/gemini-2.5-pro`, `openrouter/deepseek/deepseek-r1`,
   and `openrouter/meta-llama/llama-3.3-70b-instruct`.

This costs slightly more per token (OpenRouter takes a cut) but is the
zero-effort path to use any frontier model in Roitelet today.

---

## Sanity-checking

After adding a key + restarting, verify the model is in the registered
pool:

```bash
curl -s http://localhost:8000/v1/models | jq '.data[].id'
```

You should see `roitelet-llm` (the routed virtual model) — the actual
candidate pool is hidden behind the router, but you can confirm a
specific paid candidate is being considered by sending a prompt and
inspecting `/api/telemetry`:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quicksort in two sentences."}' | jq '.responses[].model_id'
```

If you see an `openai/...` id in the responses list, your key is wired
correctly and the router picked the model. If you don't, check that:

1. `OPENAI_API_KEY` is set (the GET `/api/settings` response will show
   `••••••••` if the key is stored — the mask is intentional).
2. Your preferences don't have `independence_local_only=true` (that
   filters out non-local models).

---

## Security note

API keys are masked when read back from `/api/settings` (the web UI sees
`••••••••`, the on-disk value stays intact). Don't commit `.env` to
version control — `.gitignore` already excludes it. For LAN deployments,
also set `ROITELET_API_TOKEN` so the settings endpoints require a Bearer
token; see `.env.example` for the variable.
