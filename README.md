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

- 🧠 **Dynamic Routing:** No manual model selection needed.
- 🌐 **Cross-family Fusion:** The synthesis judge fuses K parallel answers from *different* OSS families (Qwen + Llama + Gemma + Phi by default), not three flavours of one provider — better answers than any single model.
- ⚡ **Local Synthesis:** The fusing judge is a local LLM via Ollama, keeping the final pass private and free.
- 🌍 **Frontier Integrations:** Optional paid candidates through OpenRouter, direct OpenAI-compatible endpoints, Anthropic, Gemini, Perplexity.
- 📊 **Local Telemetry & Cost Tracking:** Dashboard monitoring for token costs, latency, simulated energy (kWh), and carbon footprints (gCO₂e).
- 🔄 **Self-Learning:** Capability-conditioned rolling Elo update loop automatically prioritises models that perform better over time.
- 🔌 **Standardized Endpoints:** OpenAI-compatible `/v1/chat/completions` + native FastAPI + MCP JSON-RPC.

---

## User Interface & Control Room

Roitelet ships with a web-based control room (vanilla JS, served by the API at `/`) that provides a transparent view into your LLM fleet:

* **Configuration:** Inject your API keys, tune local model selection, and set routing parameters (Raw Power vs. Frugality vs. Independence).
* **Usage & Monitoring:** Monitor how models are routing and verify energy estimations and carbon intensity.
* **Auto-Discovery:** Plug in your local Ollama instance, and Roitelet will automatically live-discover all models you have pulled (e.g. `ollama pull llama3.3:70b-instruct`) and inject them into the routing pool within 60 seconds.

---

## Adding a paid LLM (e.g. ChatGPT)

Roitelet ships ready to route to OpenAI's models — set one env var and
restart:

```env
OPENAI_API_KEY=sk-proj-...
```

`openai/gpt-4.1`, `openai/gpt-4o`, and `openai/gpt-4o-mini` are already
in `data/bootstrap/model_priors.json`, so the router considers them on
the next prompt. See [docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md)
for the full walkthrough, including how to add models that aren't in
the bootstrap yet and how to route to non-OpenAI providers via
OpenRouter.

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
├── api/                # FastAPI application (OpenAI-compatible & MCP endpoints)
├── web/                # Vanilla-JS control room served at `/` by the API
├── cli/                # Command-line interface and terminal REPL
├── data/
│   └── bootstrap/model_priors.json   # Benchmark-inspired default Elo priors
├── scripts/            # Crawler tooling and autonomous updates
├── tests/
│   ├── test_core.py    # Pytest for core engine
│   ├── test_api.py     # Pytest for API layer
│   ├── test_pipeline.py# Pytest for the end-to-end pipeline
│   └── test_cli.py     # Pytest for CLI tools
├── start.sh            # Launcher script
├── Dockerfile          # Containerization multi-stage build
├── docker-compose.yml  # Deploy stack definitions
├── environment.yaml    # Conda environment manifest
├── requirements.txt    # Pip dependencies
├── INSTALL.md          # Comprehensive English install guide
├── INSTALLER.md        # Comprehensive French install guide
└── .env.example
```

---
© 2025 deraison.ai | `warithmetics@deraison.ai`
