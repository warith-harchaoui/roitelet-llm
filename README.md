# Roitelet LLM

> **The Universal LLM:** The best Large Language Model for your query, no matter what.

Every week, a new frontier model is released by an AI giant. Evaluating, benchmarking, and maintaining integrations with each of them is exhausting. 

**Roitelet** (the "Wren") is a local-first adaptive router designed to abstract away this chaos. Instead of choosing a specific model, you prompt Roitelet. It dynamically selects the three most capable models for that specific prompt, queries them in parallel, and uses a local open-source model to synthesize the final, definitive answer. 

![Roitelet](assets/roitelet.jpg)

---

## The "Wren" Metaphor

Once upon a time in the great forest of Artificial Intelligences, a tiny wren dreamed of soaring higher than the majestic royal eagles. Its wings were short, so it struggled to clear the canopy! The clever little bird hid in an eagle's feathers, rode up to the roof of the sky, and at the last moment, flapped its own wings to soar past them all.

True power isn’t in server racks, billion-dollar budgets, or raw parameter counts—it’s in cunning. Our Roitelet LLM channels this with every prompt.

### How Does “Flapping Wings” Work?

Roitelet replaces the standard single-API-call with a three-step flight pattern:

1. 🦅 **Wing Flap 1 — Clever Discovery:** Our local router scores every registered model on capability priors + rolling Elo and picks the top K (default 3) from the global pool (GPT-4.1, Claude 3.7, Gemini 2.5, plus local OSS — Qwen, Llama, Gemma, Phi).
2. 🦅 **Wing Flap 2 — Aerial Triumvirate:** The K selected models answer in parallel. The point isn't to pick a winner — it's to gather diverse viewpoints from different model families.
3. 🦅 **Wing Flap 3 — Coronation:** A trusted, local open-source model (Qwen 3 by default) reads the K responses and **fuses** them into one comprehensive answer. Fusion, not selection — the synthesised answer can combine insights none of the candidates produced alone.

From your perspective, it feels like using one unified super-brain API. The rest is just show and feathers.

---

## Features

- 🧠 **Dynamic Routing:** No manual model selection needed. Hybrid regime-aware math (cost-budget, trivial, ambiguous, long-context) layers on top of the heuristic linear blend.
- 🌐 **Cross-family Fusion:** The synthesis judge fuses K parallel answers from *different* OSS families (Qwen + Llama + Gemma + Phi by default), not three flavours of one provider — better answers than any single model.
- ⚡ **Local Synthesis:** The fusing judge is a local LLM via Ollama, keeping the final pass private and free.
- 🌍 **Frontier Integrations:** Optional paid candidates through OpenRouter, direct OpenAI-compatible endpoints, Anthropic, Gemini, Perplexity.
- 🖼️ **Multimodal Attachments:** Drop images, PDFs, or audio into the chat — extracted locally (Ollama VLM caption, kreuzberg PDF text, whisper.cpp + NeMo diarization) before the text pipeline runs.
- 🎨 **Image Generation:** Route image prompts to the strongest registered image-gen model. K=1 because image fusion isn't a thing. OpenAI Images, OpenRouter relays, or local SD via the OpenAI-compatible shape. See [docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md).
- 🔬 **Two routers, one pipeline.** Default heuristic + opt-in `ROITELET_ROUTER=mf` learned matrix-factorisation router that trains on accumulated telemetry. Hybrid regimes (`trivial`, `budget_constrained`, `ambiguous`, …) adjust the math per turn.
- 🌐 **Two capability detectors.** Default keyword scan + opt-in `ROITELET_CAPABILITY_DETECTOR=embedding` classifier on top of a local Ollama embedding model (`nomic-embed-text`). Falls back transparently when offline.
- 📊 **Local Telemetry & Cost Tracking:** Dashboard monitoring for token costs, latency, simulated energy (kWh), and carbon footprints (gCO₂e).
- 🔄 **Self-Learning:** Capability-conditioned rolling Elo update loop automatically prioritises models that perform better over time.
- 🔌 **Standardized Endpoints:** OpenAI-compatible `/v1/chat/completions` + native FastAPI + MCP JSON-RPC. Image generation at `/api/images` (and OpenAI-compatible `/v1/images/generations`).
- 💬 **Slash commands:** `/image`, `/speech`, `/local`, `/cheap <usd>`, `/k <n>`, `/help` — per-turn overrides parsed at the prompt boundary. See [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md).
- 🔐 **Optional Bearer-Token Gate:** Set `ROITELET_API_TOKEN` to lock down every mutating + listing endpoint for LAN deployments. Defaults to a no-op for local-only single-user UX.

---

## User Interface & Control Room

Roitelet ships with a web-based control room (vanilla JS, served by the API at `/`) that provides a transparent view into your LLM fleet:

* **Configuration:** Inject your API keys, tune local model selection, and set routing parameters (Raw Power vs. Frugality vs. Independence).
* **Usage & Monitoring:** Monitor how models are routing and verify energy estimations and carbon intensity.
* **Auto-Discovery:** Plug in your local Ollama instance, and Roitelet will automatically live-discover all models you have pulled (e.g. `ollama pull llama3.3:70b-instruct`) and inject them into the routing pool within 60 seconds.

---

## Adding more LLMs

Roitelet treats every provider with an OpenAI-compatible
`/v1/chat/completions` endpoint as a first-class extension point. The
same path works for paid APIs, frontier-via-OpenRouter, and local GGUF
files served by `llama.cpp`'s `llama-server`.

- **Any paid LLM (ChatGPT, Mistral, Together, Groq, …)** — set the
  endpoint + key, list the model names. Done. Full walkthrough in
  [docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md).
- **Any local GGUF file** — either drop it into Ollama via a
  `Modelfile` (recommended, zero settings edits) or serve it with
  `llama-server` and treat it as an OpenAI-compatible endpoint. Walked
  through in [docs/ADDING_LOCAL_LLM.md](docs/ADDING_LOCAL_LLM.md).
- **Direct OpenAI** — special case of the first: set
  `OPENAI_API_KEY` and restart; `openai/gpt-4.1`, `openai/gpt-4o`, and
  `openai/gpt-4o-mini` are already in `data/bootstrap/model_priors.json`.

---

## Installation & Setup

> **Complete Installation Guide:** See [INSTALL.md](INSTALL.md) for full instructions covering conda, venv, and Docker deployment.

### Quick Start (Conda)

```bash
# 1. Create and isolate environment
conda env create -f environment.yaml
conda activate roitelet-llm

# 2. Configure credentials
cp .env.example .env
# Edit .env to add your API keys (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.)

# 3. Pull the OSS default bundle (Qwen + Llama + Gemma + Phi + VLM)
chmod +x scripts/pull_defaults.sh
./scripts/pull_defaults.sh

# 4. Start the application
chmod +x start.sh
./start.sh
```

- **API Base URL:** `http://localhost:8000`
- **Web Control Room:** `http://localhost:8000/` (served by the API)

---

## Folder Layout

```text
roitelet-llm/
├── core/               # Shared backend logic, router, storage, capabilities
│   ├── pipeline.py     # End-to-end orchestration (router → fan-out → judge → Elo)
│   ├── router.py       # Capability-weighted scoring + top-K selection
│   ├── registry.py     # Bootstrap + user + live-Ollama model pool, rolling Elo
│   ├── judge.py        # Anonymized synthesis with sentinel-delimited winners
│   ├── capabilities.py # Lexical capability detection
│   ├── providers/      # Ollama + OpenAI-compatible clients (OpenRouter, OpenAI, ...)
│   └── multimodal/     # Local audio / image / PDF extractors
├── api/                # FastAPI application (native, OpenAI-compatible, MCP)
├── web/                # Vanilla-JS control room served at `/` by the API
├── cli/                # Command-line interface and terminal REPL
├── docs/               # Topic-specific guides (e.g. ADDING_PAID_LLM.md)
├── data/
│   └── bootstrap/model_priors.json   # Benchmark-inspired default Elo priors
├── scripts/            # Crawler tooling, asset vendor, pull_defaults.sh
├── tests/              # Pytest suite (core, api, pipeline, cli, eval)
├── assets/             # Branding (logo)
├── start.sh            # Launcher script
├── Dockerfile          # Multi-stage container build
├── docker-compose.yml  # Deploy stack definition
├── environment.yaml    # Conda environment manifest
├── requirements.txt    # Pip dependencies
├── MECHANISM.md        # Architecture deep-dive (Mermaid diagrams)
├── INSTALL.md          # English install guide
├── MANUAL.md           # English usage guide
├── INSTALLER.md        # French install guide
├── MODEDEMPLOI.md      # French usage guide
├── LISEZMOI.md         # French README mirror
└── .env.example
```

---
© 2025 deraison.ai | `warithmetics@deraison.ai`
