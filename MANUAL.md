# MANUAL — Installation and settings

> **Complete installation guide**: [INSTALL.md](INSTALL.md)

## 1. Conda installation

```bash
# Preferred — one command using environment.yaml
conda env create -f environment.yaml
conda activate roitelet-llm

# Alternative — manual
conda create -n roitelet-llm python=3.11 -y
conda activate roitelet-llm
pip install -r requirements.txt
```

## 2. Environment file

```bash
cp .env.example .env
```

### Minimum recommended setup

```env
OPENROUTER_API_KEY=...
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b
LOCAL_VLM_MODEL=qwen2.5vl:7b
```

## 3. Start the project

```bash
./start.sh
```

This launches a single FastAPI process on port `8000` that serves the JSON API and the web control room at `/`.

## 4. Docker setup

```bash
docker compose up --build
```

## 5. Web control room

### Configuration page
Use it to:
- store OpenRouter credentials,
- point to an Ollama server,
- choose the local synthesis model,
- authorize or disable VLMs,
- tune:
  - **Raw Power**,
  - **Frugality**,
  - **Independence (local only)**.

### Monitoring page
The monitoring page aggregates:
- number of calls,
- energy usage estimate,
- carbon estimate,
- cost estimate,
- top detected prompt capability.

## 6. Command-line interface (CLI)

Run an interactive terminal chat (inspired by Gemini CLI):
```bash
python -m cli chat
```

Single-shot mode:
```bash
python -m cli ask "What is the capital of France?"
```

## 7. OpenAI-compatible serving

Endpoint:

```text
POST /v1/chat/completions
```

Model name:

```text
roitelet-llm
```

## 7. MCP access

Endpoint:

```text
POST /mcp
```

Methods supported:
- `initialize`
- `tools/list`
- `tools/call`

Main tool:
- `roitelet.chat`

## 8. Data written to disk

The project stores local JSON files under `data/`:
- `conversations/`
- `telemetry/`
- `runtime/settings.json`
- `runtime/elo_state.json`

## 9. Notes on learning

The online learner currently uses **capability-conditioned rolling Elo updates**.
This is the intentionally simple first implementation before introducing a more advanced classifier or contextual bandit.
