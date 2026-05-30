# Roitelet LLM

> **Ask one question — several AI models answer at the same time — a
> small model on your computer picks the best parts of each answer
> and gives you one response.**

Roitelet runs on your machine. You can use the AI on your laptop, or
plug in cloud models (ChatGPT, Claude, Gemini via OpenRouter, …) and
let them compete on each question. Optionally, Roitelet hides your
personal information before anything goes to the cloud and puts it
back in the answer.

![Roitelet](assets/roitelet.jpg)

---

## The wren

Once upon a time, the birds of the forest agreed that whoever flew
highest would be crowned king. The eagle climbed effortlessly past
every other bird. But a tiny wren had hidden in the eagle's feathers,
rode all the way up, and at the very top fluttered a few wingbeats
higher to take the crown.

The point isn't that the wren is the strongest bird — it isn't. The
point is what small, well-placed local moves can do on top of much
larger external forces. Roitelet (*roitelet* is French for "wren") is
shaped around the same idea: a small local pipeline that rides on top
of large language models — composing them, comparing their answers,
running its own local synthesis pass on top.

---

## Which doc should I read?

Pick the row that matches what you came here to do.

| You are… | …and you want to | Start here |
|---|---|---|
| 🧑 **A curious user** | Try Roitelet on your laptop, ask it a question, see it work | [Quick start](#quick-start) (below) |
| 🧰 **A user with files** | Drop in PDFs, audio, images, or a website URL and ask about them | [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md) |
| 🔐 **A privacy-conscious user** | Understand what stays local and how PII hiding works | [docs/PRIVACY.md](docs/PRIVACY.md) → [docs/PSEUDO.md](docs/PSEUDO.md) |
| 🧑‍💻 **A developer with existing OpenAI compatible tooling** | Point your `openai` SDK / LiteLLM / Continue.dev at Roitelet | [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md) |
| 🏗️ **A developer wiring local models** | Add a local GGUF, OpenAI, Mistral, Together, etc. | [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) |
| 🎛️ **A power user** | Use slash routes (`/image`, `/personal`, …) and per-turn controls | [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md) |
| 🖥️ **A sysadmin / installer** | Install on Linux/Mac/Windows with conda, venv, or Docker | [INSTALL.md](INSTALL.md) ([Français](INSTALLER.md)) |
| 🔬 **A researcher / honest skeptic** | See the numbers — does fusion actually help? | [docs/EVALUATION.md](docs/EVALUATION.md) |
| 🛠️ **A contributor / forker** | Understand the internals: router, regimes, Elo loop | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 🇫🇷 **Francophone** | Tout ce qui précède, en français | [LISEZMOI.md](LISEZMOI.md) |

---

## Quick start

Five minutes to a running Roitelet on your laptop:

```bash
# 1. Install Ollama (one-time).
#    macOS: brew install ollama
#    Linux: curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a small local model so Roitelet has something to talk to.
ollama pull qwen3:8b
ollama pull nomic-embed-text     # tiny — used by personal mode

# 3. Install Roitelet.
git clone https://github.com/<your-fork>/roitelet-llm.git
cd roitelet-llm
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 4. Run.
./start.sh                       # opens http://localhost:8000
```

Open `http://localhost:8000` and ask anything. The web UI is
self-explanatory — there's a language toggle (EN/FR) in the
sidebar header, a "sliders" icon next to the send button for
per-message options, and a Settings sheet behind the gear at the
bottom of the sidebar.

Prefer the terminal? Same operations, different surface:

```bash
roitelet ask "Explain quicksort in one paragraph."
roitelet ask --pseudonymize "Email Marie Dupont at marie@orange.fr about Q3."
roitelet ask --url https://docs.python.org/3/library/asyncio.html "Summarise."
roitelet chat --independence     # interactive REPL, local-only
roitelet settings get            # see what's persisted
```

For installer deep-dives (Docker, model bundles, profile comparison):
[INSTALL.md](INSTALL.md) (English), [INSTALLER.md](INSTALLER.md)
(French).

---

## What it does (plain language)

| Feature | What it means for you |
|---|---|
| **Compare AI models** | One prompt, several answers (e.g. Claude + Llama + Gemma), one final response. |
| **Local-first** | If you only configure local models, **nothing leaves your machine**. |
| **Hide personal info** | Toggle "Pseudonymize" — names, addresses, IDs are swapped for plausible fakes before sending, restored in the answer. |
| **Attach files** | Audio (transcribed), images (read by a vision model), PDFs (text-extracted) — all locally. |
| **Attach websites** | Paste a URL — Roitelet scrapes the page (Firecrawl) and includes it in the prompt. |
| **Personal RAG** | Drop your own notes into a folder; Roitelet uses them to answer your questions. |
| **Image generation** | If you've configured DALL-E / Stable Diffusion / Imagen, ask with `/image` to draw. |
| **Same operations in CLI and API** | Anything you can do in the GUI you can do from the terminal or an HTTP call. |

---

## How it works (one diagram)

```mermaid
flowchart LR
    U[User prompt] --> P{Pseudonymize?<br>(opt-in)}
    P -- yes --> PFW[Local LLM<br>strips PII] --> R
    P -- no --> R
    R[Router<br>capability priors<br>+ rolling Elo<br>+ regimes] --> SEL[Top-K<br>candidates]
    SEL -.parallel.-> C1[Candidate 1]
    SEL -.parallel.-> C2[Candidate 2]
    SEL -.parallel.-> CN[Candidate K]
    C1 --> J[Local judge<br>anonymized<br>+ shuffled]
    C2 --> J
    CN --> J
    J --> REV{Pseudo on?}
    REV -- yes --> PREV[Local LLM<br>restores PII] --> A
    REV -- no --> A
    A[Fused answer] --> USER[User]
    J -.winners.-> ELO[(Per-capability<br>rolling Elo)]
    ELO -.next turn.-> R
    style P fill:#fef3c7,stroke:#f59e0b
    style REV fill:#fef3c7,stroke:#f59e0b
    style PFW fill:#fef3c7,stroke:#f59e0b
    style PREV fill:#fef3c7,stroke:#f59e0b
    style J fill:#dbeafe,stroke:#3b82f6
    style ELO fill:#f3e8ff,stroke:#a855f7
```

Per turn:

1. **Router** picks the top-K models for the question (default K=2).
2. **Fan-out** — the K models answer in parallel.
3. **Judge** — a model on your machine reads the K answers
   (anonymised and shuffled), then fuses them into one.
4. **Elo update** — winners gain points on the question's topic;
   losers lose. The next turn benefits from this signal.

Full internals (the math, the regime detectors, the matrix-fac
router variant) live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## What's actually in here (the honest pitch)

Roitelet packages a few primitives together that I don't see
combined elsewhere:

- **Per-capability rolling Elo updated from the judge's own
  signal** — an online preference loop, not a fixed scorecard or
  an offline-trained classifier.
- **Calibrated `quality_threshold` knob** — a single number in
  [0, 1] traces the cost/quality Pareto frontier of Roitelet's
  router (shape-equivalent to RouteLLM's threshold knob, derived
  from the rolling-Elo blend).
- **Ecofrugality as a first-class router input** — cost (USD) +
  energy (kWh) + latency in one bonus term.
- **LLM-based pseudonymization** with a 19-category PII taxonomy
  and a literal+LLM repair reverse pass, fail-closed if any
  invariant breaks. Not a regex/NER proxy.
- **Regime-aware hybrid routing** — the linear blend composes
  with per-prompt regimes (trivial, budget-constrained,
  long-context, ambiguous, capability-dominant).
- **Firecrawl recursive crawling** as a fourth input modality,
  alongside audio / image / PDF.
- **Cross-surface parity** — every operation works in the GUI, on
  the CLI, and in the API (native + OpenAI-compatible + MCP), with
  full EN/FR i18n.

What it isn't:

- a new model architecture
- a calibrated `P(strong wins)` cost-router (that's RouteLLM)
- a hosted gateway (that's LiteLLM)
- a peer-reviewed result

---

## Three surfaces, same features

| Surface | How to access | Per-turn preferences |
|---|---|---|
| **Web** (GUI) | `http://localhost:8000/` after `./start.sh` | Sliders icon next to the send button |
| **CLI** | `roitelet ask "…"` / `roitelet chat` | `--top-k`, `--independence`, `--pseudonymize`, `--max-cost-usd`, `--quality-threshold`, `--url[ --url-recursive]`, `--verbose` |
| **API** | Native `POST /api/chat`, OpenAI-compatible `POST /v1/chat/completions`, MCP `POST /mcp` | `preferences.{independence, pseudonymize, top_k, max_cost_usd, quality_threshold}` in the JSON body |

For OpenAI clients (Python SDK, LiteLLM, Continue.dev, …) Roitelet
is a drop-in target — see [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md).
Roitelet-specific knobs ride on `metadata.roitelet.{…}` so you
don't lose anything by using the OpenAI shape.

---

## What works today (feature list)

- **Hybrid routing** — capability priors + rolling Elo + regimes
  (cost budget, trivial-prompt, long-context, ambiguous,
  capability-dominant). Optional learned matrix-fac router behind
  `ROITELET_ROUTER=mf`.
- **Parallel top-K fan-out** — bounded by the slowest candidate.
  K=2 default.
- **Local synthesis** — anonymised, shuffled candidates → fused
  answer; replaceable judge.
- **Pseudonymization** — opt-in PII swap before remote calls,
  restore after. Fail-closed; full audit.
  [docs/PSEUDO.md](docs/PSEUDO.md)
- **Personal mode** — drop files into `data/personal/inbox/`;
  small corpora inject inline, large ones switch to embedding
  retrieval. 2-D PCA scatter.
  [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md)
- **Multimodal attachments** — images, PDFs, audio extracted
  locally (Ollama VLM, kreuzberg, whisper.cpp + NeMo) before the
  text pipeline.
- **Website attachments** — Firecrawl-scraped markdown (single
  page or recursive crawl).
- **Image generation** — K=1 routing to the strongest registered
  image-gen model.
- **Universal extension** — any provider with an OpenAI-compatible
  endpoint registers in three settings fields.
- **Standardized endpoints** — native `/api/chat`,
  OpenAI-compatible `/v1/chat/completions` + `/v1/images/generations`,
  MCP JSON-RPC `/mcp`.
- **Slash commands** — routes only (`/image`, `/speech`,
  `/personal`, `/help`). Per-turn preferences are visible controls,
  not slashes.
- **Local telemetry** — per-turn JSON: router decision, every
  candidate (failures included), synthesis, winners.
- **Optional Bearer-token gate** — `ROITELET_API_TOKEN` locks
  every endpoint except `/healthz` and the static SPA.

---

## Latency, cost, when not to use it

A turn's wall-clock is `max(candidate_latencies) + judge_latency`
because the fan-out runs through `asyncio.gather`. Local models are
free at the marginal token but cost RAM/VRAM. Remote candidates
cost what their provider charges — Roitelet doesn't arbitrage, it
just calls them.

K=2 is the empirical sweet spot on the held-out dataset (see
[docs/EVALUATION.md](docs/EVALUATION.md)); K=1 leaves quality on
the table, K=3 doubles wall-clock for ~+1 pp.

**When not to use Roitelet:**

- **Very low-latency chat UX.** A single fast model beats fan-out
  + fusion. If your UI lives or dies by sub-second response,
  wrong tool.
- **Trivial prompts.** "What's 2+2?" doesn't need three opinions.
- **High-volume production traffic.** Cost is multiplicative
  (K models + a judge per turn). A single calibrated model +
  caching is cheaper.
- **You just want one provider gateway.** That's
  [LiteLLM](https://github.com/BerriAI/litellm)'s job.

---

## Security note

Roitelet ships **safe by default**: `start.sh` binds `127.0.0.1`
and `ROITELET_API_TOKEN` is empty. Localhost-only with no auth is
fine for a single-user laptop.

The Docker image binds `0.0.0.0` because container port-forwarding
requires it — exposure is then governed by your `docker-compose.yml`
port map.

**Before exposing to a LAN, the internet, ngrok, Tailscale, etc.:**

1. Set `ROITELET_API_TOKEN` to a non-empty value.
2. Either keep the service behind an auth-handling reverse proxy,
   or accept that the token is your only line of defence.

Without those, anyone who reaches the port can read your
conversations, your raw telemetry (which contains prompts and
provider responses), and trigger paid provider calls against your
keys. Threat model: [docs/PRIVACY.md](docs/PRIVACY.md).

---

## How Roitelet differs from neighbouring projects

| Project | Primary role | How Roitelet differs |
|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | OpenAI-compatible gateway over many APIs | Roitelet is narrower: local-first fan-out + local synthesis + inspectable Elo. LiteLLM is a candidate Roitelet could call. |
| [OpenRouter](https://openrouter.ai) | Hosted multi-model marketplace | Roitelet runs on your machine. OpenRouter is a candidate provider, not a replacement. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Cost-aware strong-vs-weak routing trained on preferences | Roitelet does top-K + fusion, not binary routing. Roitelet exposes a `quality_threshold` knob in the same *shape* (single scalar, monotonic), derived from rolling Elo rather than a preference-trained classifier. |
| [LangChain](https://www.langchain.com) / LangGraph | LLM-orchestration frameworks | Roitelet is an end-user system, not a framework. |
| [DSPy](https://github.com/stanfordnlp/dspy) | Prompt-pipeline optimisation | Roitelet routes and fuses at inference time. They can coexist. |
| Single-model chat clients | One model in, one answer out | Roitelet trades simplicity and latency for comparison + redundancy + synthesis. |

---

## Folder layout

```text
roitelet-llm/
├── core/               # Router, registry, pipeline, judge, pseudo, multimodal
├── api/                # FastAPI app (native + OpenAI-compat + MCP)
├── web/                # Vanilla-JS GUI (served at /)
├── cli/                # `roitelet` console-script entry point
├── docs/               # Topic guides — architecture, eval, features, privacy
├── data/bootstrap/     # Default Elo + capability priors
├── scripts/            # pull_defaults.sh, vendor_web_assets.sh
├── tests/              # Unit + opt-in DeepEval / MT-Bench suite
├── .demos/             # Reproducible demo bundle (screenshots, transcripts, video)
├── INSTALL.md          # English install
├── INSTALLER.md        # French install
├── LISEZMOI.md         # French README mirror
└── README.md           # This file
```

---

## License

Released under the **BSD 3-Clause License** — see [LICENSE](LICENSE).

## Author

[Warith HARCHAOUI](https://www.linkedin.com/in/warith-harchaoui/)
