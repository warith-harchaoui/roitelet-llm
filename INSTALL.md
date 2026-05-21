# INSTALL — Roitelet LLM

Complete installation guide for every supported setup path.

---

## Prerequisites

| Tool | Minimum version | Purpose |
|---|---|---|
| [Ollama](https://ollama.com) | 0.3+ | Local synthesis / judge model |
| Python | 3.11+ | Runtime |
| [conda](https://docs.conda.io) **or** venv | any | Environment isolation |
| [Docker](https://docs.docker.com/get-docker/) | 24+ | Container deployment (optional) |

> **Recommended first model to pull with Ollama:**
> ```bash
> ollama pull qwen2.5:14b-instruct
> ```
> This is the default local synthesis / judge model. Pull it before starting Roitelet.

---

## Option A — Conda (recommended)

### A1. One-command environment creation

```bash
conda env create -f environment.yaml
conda activate roitelet-llm
```

The `environment.yaml` file pins Python 3.11 and delegates all package
installation to `requirements.txt` via pip.

### A2. Manual creation (equivalent)

```bash
conda create -n roitelet-llm python=3.11 -y
conda activate roitelet-llm
pip install -r requirements.txt
```

---

## Option B — pip + venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Option C — Docker

### C1. Build and start

```bash
cp .env.example .env          # then edit .env with your credentials
docker compose up --build -d
```

The container exposes:
- **API + Web UI**: `http://localhost:8000` (the FastAPI process serves the JSON API and the static web client at the same origin)

> **Ollama on the host machine**
> The compose file pre-configures `LOCAL_LLM_BASE_URL=http://host.docker.internal:11434`
> so Roitelet inside Docker can reach Ollama running natively on your machine
> (macOS, Windows, and Linux with Docker 20.10+).

### C2. Persistent data

Conversations, telemetry, Elo state, and settings are written to the Docker
named volume `roitelet_data`. To inspect or back up:

```bash
docker volume inspect roitelet_data
```

### C3. Useful commands

```bash
docker compose logs -f                  # live logs
docker compose ps                       # check health status
docker compose down                     # stop
docker compose down -v                  # stop + delete volume
docker compose pull && docker compose up -d   # update image
```

---

## Configuration

### 1. Copy the env template

```bash
cp .env.example .env
```

### 2. Minimum recommended settings

```env
# Paid frontier models via OpenRouter
OPENROUTER_API_KEY=sk-or-...

# Local synthesis / judge model
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen2.5:14b-instruct

```

> **Local-only mode (zero cost)**
> You can run entirely offline with no API keys. Set
> `ROITELET_CANDIDATE_POOL_SIZE=4` and add local Ollama models
> through the web configuration page.

### 3. Full variable reference

See [`.env.example`](.env.example) for all available variables and their defaults.

---

## Starting the service

### Direct (conda or venv)

```bash
chmod +x start.sh
./start.sh
```

This launches a single uvicorn process on `http://localhost:8000` which serves both the JSON API and the static web client at `/`.

### Manual

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## First-run verification

```bash
# 1. API health check
curl http://localhost:8000/

# 2. List registered models
curl http://localhost:8000/v1/models

# 3. Send a test prompt (requires Ollama running)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "top_k": 1}'
```

Expected health response:
```json
{"status": "ok", "service": "roitelet-llm", "base_url": "http://localhost:8000"}
```

---

## Running the test suite

```bash
# Install dev dependencies first
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -q
```

All 21 tests are network-free and complete in under a second.

---

## Updating

### pip / conda

```bash
git pull
pip install -r requirements.txt   # pick up any new packages
./start.sh
```

### Docker

```bash
git pull
docker compose build --no-cache
docker compose up -d
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Bootstrap priors not found` | Corrupted clone | Re-clone the repo |
| `Connection refused` on port 8000 | API not started | Run `./start.sh` |
| Synthesis always returns empty | Ollama not running | `ollama serve` |
| `401 Unauthorized` from OpenRouter | Wrong key | Update `OPENROUTER_API_KEY` in `.env` |
| Models don't appear in router despite `ollama pull` | Cache TTL | Wait up to 60 s or restart API |

---

## Folder layout

```text
roitelet-llm/
roitelet-llm/
├── core/               # router, registry, judge, pipeline, capabilities
├── api/                # FastAPI application (OpenAI-compatible & MCP)
├── gui/                # Streamlit control room
├── cli/                # Command-line interface and terminal REPL
├── data/
│   └── bootstrap/model_priors.json   # benchmark-inspired prior scores
├── tests/
│   ├── test_core.py    # Pytest for core engine
│   ├── test_api.py     # Pytest for API layer
│   ├── test_gui.py     # Pytest for Streamlit helpers
│   └── test_cli.py     # Pytest for CLI tools
├── start.sh            # launcher script
├── Dockerfile          # multi-stage build
├── docker-compose.yml  # compose stack
├── environment.yaml    # conda environment
├── requirements.txt    # pip dependencies
└── .env.example        # environment variable template
```
