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

1. 🦅 **Wing Flap 1 — Clever Discovery:** Our local router predicts which 3 LLMs from the global pool (like GPT-4o, Claude 3.7, Gemini 2.5) are most likely to excel at your specific question based on capability priors and historical Elo tracking.
2. 🦅 **Wing Flap 2 — Aerial Triumvirate:** The three selected models generate answers in parallel. We avoid relying on just one provider.
3. 🦅 **Wing Flap 3 — Coronation:** A trusted, local open-source model (like Qwen2.5 running on Ollama) reads the three responses and synthesizes them into a single, comprehensive, highly-accurate final answer.

From your perspective, it feels like using one unified super-brain API. The rest is just show and feathers.

---

## Features

- 🧠 **Dynamic Routing:** No manual model selection needed.
- ⚡ **Local Synthesis:** The final judge is a local LLM running via Ollama, maintaining privacy and control over the final synthesis.
- 🌍 **Frontier Integrations:** Built-in support for OpenRouter, direct OpenAI-compatible endpoints, Anthropic, Gemini, Perplexity, and more.
- 📊 **Local Telemetry & Cost Tracking:** Dashboard monitoring for token costs, latency, simulated energy (kWh), and carbon footprints (gCO₂e).
- 🔄 **Self-Learning:** Implements a rolling, capability-conditioned Elo update loop to automatically prioritize models that perform better over time.
- 🔌 **Standardized Endpoints:** Serves an OpenAI-compatible `/v1/chat/completions` API alongside a native FastAPI backend and an MCP JSON-RPC Server.

---

## User Interface & Control Room

Roitelet ships with a **Streamlit** control room that provides a transparent view into your LLM fleet:

* **Configuration:** Inject your API keys, tune local model selection, and set routing parameters (Raw Power vs. Frugality vs. Independence).
* **Usage & Monitoring:** Monitor how models are routing and verify energy estimations and carbon intensity.
* **Auto-Discovery:** Plug in your local Ollama instance, and Roitelet will automatically live-discover all models you have pulled (e.g. `ollama pull llama3.3:70b-instruct`) and inject them into the routing pool within 60 seconds.

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

# 3. Pull a local synthesis model for the coronation phase
ollama pull qwen2.5:14b-instruct

# 4. Start the Application
chmod +x start.sh
./start.sh
```

- **API Base URL:** `http://localhost:8000`
- **Streamlit Control Room:** `http://localhost:8501`

---

## Folder Layout

```text
roitelet-llm/
├── app/
│   ├── core/           # router, registry, judge, pipeline, capabilities
│   ├── providers/      # Ollama, generic clients, and integrations
│   ├── config.py       # pydantic-settings
│   ├── main.py         # FastAPI application entrypoint
│   ├── schemas.py      # shared API/Internal data models
│   └── storage.py      # Local JSON data persistence layer
├── data/
│   └── bootstrap/model_priors.json   # Benchmark-inspired default Elo priors
├── scripts/            # Crawler tooling and autonomous updates
├── tests/
│   └── test_roitelet.py              # Pytest battery
├── streamlit_app.py    # Streamlit dashboard
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
