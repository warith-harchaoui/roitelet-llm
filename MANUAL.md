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

### Attachments (multimodal)
The chat input accepts file attachments. Each attachment is converted to text
**locally** before the standard pipeline runs — the router, candidate fan-out,
and judge stay text-only:

| Format | Pipeline | Notes |
|---|---|---|
| Image (`.png`, `.jpg`, `.webp`, ...) | Local Ollama VLM caption (`qwen2.5vl:7b` by default) | Gated by the **Allow VLMs** switch on the Config page |
| PDF (`.pdf`) | `kreuzberg` (pdfium text layer, Tesseract OCR fallback) | Requires `pip install -e .[multimodal]` |
| Audio (`.wav`, `.mp3`, `.m4a`, ...) | `pywhispercpp` transcription + NeMo Sortformer diarization | Requires `pip install -e .[multimodal]` |

Files of unknown type are skipped with a `[Note] Skipped: ...` line appended
to the prompt so you can see what was ignored.

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

## 10. Locking down the API (LAN / multi-user deployments)

The single-user local-first default leaves every endpoint open. For shared
deployments set `ROITELET_API_TOKEN` in `.env`:

```env
ROITELET_API_TOKEN=please-change-me
```

When set, every chat, settings, conversation, and telemetry endpoint requires
`Authorization: Bearer <token>`. `/healthz`, `/v1/models`, and the static SPA
assets stay public so basic liveness checks keep working.

> **Note** — the bundled web UI does not yet send the bearer header. With a
> token configured, prefer the CLI, the OpenAI-compatible endpoint, or a
> reverse proxy that injects the header until the UI gains token support.
