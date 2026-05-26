# Roitelet LLM

> **A local-first LLM routing and fusion workbench.** Roitelet routes
> prompts across local and remote language models, compares their
> answers, synthesizes a final response locally, and learns routing
> preferences over time from its own judge signal.

It is **not** a "universal LLM" or a guarantee of the best answer for
every query. It is an experiment platform for the cost / latency /
privacy / quality tradeoffs that come up when you stop committing to
one model.

![Roitelet](assets/roitelet.jpg)

---

## What Roitelet does

Given a user prompt, Roitelet:

1. Scores every registered model (local + optional remote) against
   the prompt using a hybrid router — curated capability priors,
   rolling Elo, and a small set of regime-aware filters (cost budget,
   trivial-prompt, long-context, …).
2. Fans out to the top-K candidates in parallel (default K=3).
3. Passes the K answers, **anonymized and shuffled**, to a local
   synthesis judge that fuses them into a single answer.
4. Persists per-turn telemetry and nudges per-capability Elo scores
   so the next prompt can be routed slightly better.

Each step is inspectable. The router decision, the candidate replies,
the judge's reasoning, and the rolling Elo state all land as plain
JSON on disk; nothing is hidden behind an opaque service.

### What it is good for

- **Comparing model families** on the same prompt without juggling
  three SDKs.
- **Running a local synthesis pass** on top of remote candidate
  answers — useful when you want the final word to come from a model
  you control.
- **Experimenting with routing and fusion strategies** (cost-budget
  filters, learned matrix-factorisation router, embedding-based
  capability detector) under a single API.
- **Studying tradeoffs** between cost, latency, privacy, and answer
  quality, with the data trail to make those studies reproducible.

### What it does **not** claim

- That the fused answer is always better than the strongest single
  candidate. Whether fusion helps depends on the prompt class, the
  judge model, and the candidate diversity; this is exactly what the
  ablation roadmap in [docs/EVALUATION.md](docs/EVALUATION.md) is
  designed to measure.
- That the local synthesis judge is an objective oracle. Roitelet
  learns *judge-conditioned* preferences — different judges produce
  different rolling-Elo trajectories. The judge bias is a feature to
  inspect, not a bug to hide.
- That it is automatically "private". Roitelet is local-**first**, not
  local-**only**. Prompts may still go out to remote providers when
  they are selected as candidates. See
  [docs/PRIVACY.md](docs/PRIVACY.md) for the precise distinction.

---

## The wren

The project is named after the wren (*roitelet* in French): a tiny
bird that, in the fable, rides on an eagle's back and then flutters
slightly higher at the last moment. The metaphor is about composing
small local moves on top of large external models — not about the
wren being the best bird in the forest.

---

## Features

- **Hybrid routing.** Capability priors + rolling Elo + regime-aware
  adjustments (cost budget, trivial-prompt, long-context, ambiguous,
  capability-dominant). Optional learned matrix-factorisation router
  behind `ROITELET_ROUTER=mf`.
- **Parallel top-K fan-out.** Default K=3, configurable per turn.
  Wall-clock time is bounded by the slowest selected candidate
  (see [latency + cost tradeoffs](#latency-and-cost-tradeoffs) below).
- **Local synthesis pass.** Candidate answers are anonymized,
  shuffled, and handed to a local Ollama model that fuses them.
  The judge is replaceable.
- **Per-capability rolling Elo.** Each turn's judge winners gain Elo
  on the capabilities the prompt invoked; losers lose. Bounded
  updates; no feedback runaway.
- **Universal extension point.** Any paid LLM with an OpenAI-compatible
  `/v1/chat/completions` endpoint registers in three settings fields.
  Same for any local GGUF served by `llama-server`.
- **Multimodal attachments.** Drop images, PDFs, or audio — extracted
  locally (Ollama VLM, kreuzberg, whisper.cpp + NeMo) before the
  text pipeline runs.
- **Image generation.** K=1 routing to the strongest registered
  image-gen model (no fusion — image ensembling is not a defined
  operation).
- **Personal mode.** Drop your own files into a folder; small corpora
  inject inline (Karpathy LLM-wiki style), large ones switch to
  embedding retrieval. Includes a 2-D PCA scatter of the corpus.
  See [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md).
- **Two capability detectors.** Default keyword scan + opt-in
  embedding-based classifier on top of a local Ollama embedding model.
- **Slash commands.** `/image`, `/speech`, `/personal`, `/local`,
  `/cheap <usd>`, `/k <n>`, `/help`. See
  [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md).
- **Standardized endpoints.** OpenAI-compatible `/v1/chat/completions`
  + `/v1/images/generations`, native FastAPI, MCP JSON-RPC.
- **Local telemetry.** Per-turn JSON records of the router decision,
  every candidate response (including failures), the synthesis, and
  the winners. See [docs/PRIVACY.md](docs/PRIVACY.md) for what's
  recorded.
- **Optional Bearer-token gate.** `ROITELET_API_TOKEN` locks every
  mutating + listing endpoint. Off by default to preserve the
  single-user-localhost UX.

---

## How Roitelet differs from neighbouring projects

Roitelet lives in an active space. The table below positions it
against the closest neighbours, fairly and at a high level. None of
these projects "lose" — they solve different problems.

| Project | Primary role | Strengths | How Roitelet differs |
|---|---|---|---|
| [**LiteLLM**](https://github.com/BerriAI/litellm) | Provider gateway / OpenAI-compatible abstraction over many APIs. | Broad provider coverage, drop-in OpenAI client, server mode, robust SDK. | Roitelet is narrower and more opinionated: local-first by default, focused on multi-model fan-out + a local synthesis pass + an inspectable rolling-Elo loop. LiteLLM is one of the providers Roitelet could plug into. |
| [**OpenRouter**](https://openrouter.ai) | Hosted marketplace + routing for many remote models behind one billing surface. | Huge model catalogue, hosted convenience, single API key. | Roitelet runs on your machine and lets you inspect / modify the routing and fusion loop. OpenRouter is an excellent *candidate provider* for Roitelet, not a replacement. |
| [**RouteLLM**](https://github.com/lm-sys/RouteLLM) | Research framework for cost-aware routing between a strong and a weak model, trained on human preference data. | Principled `P(strong wins)` estimator, published cost-quality Pareto curves, calibrated threshold knob. | Roitelet does top-K fan-out + fusion rather than binary routing, and is set up as a personal workbench rather than a research benchmark. RouteLLM's `mf` router slots cleanly behind Roitelet's `Router` Protocol if you want both. |
| [**LangChain / LangGraph**](https://www.langchain.com) | General LLM-orchestration frameworks. | Composable graphs, broad ecosystem, agent patterns. | Roitelet is an end-user system, not a framework. It ships an opinionated pipeline (router → parallel candidates → local judge → telemetry) with HTTP, CLI and web entry points, instead of leaving the orchestration to you. |
| [**DSPy**](https://github.com/stanfordnlp/dspy) | Programming model for compiling prompt pipelines, optimising them against metrics. | Powerful abstractions for optimisation-driven prompting and retrieval. | Roitelet doesn't compile programs — it routes and fuses at inference time, with the rolling-Elo loop as its only online "optimisation". DSPy and Roitelet can coexist (DSPy could be the candidate; Roitelet could be the runner). |
| **Single-model chat clients** (OpenAI playground, Ollama desktop, etc.) | One model in, one answer out. | Simple, fast, low-latency, low-cost. | Roitelet deliberately trades simplicity and latency for comparison, redundancy and synthesis. For a trivial prompt to a familiar model, those clients win. For "I want three opinions and a local synthesis", Roitelet is the one. |

The honest summary: Roitelet is a **workbench**, not a gateway, not a
hosted marketplace, not a framework, and not a chat client. Pick the
tool whose primary role matches what you're actually trying to do.

---

## Why fusion can help — and where the judge bias sits

Judging and fusing K already-written candidate answers is a different
job from generating a strong answer from scratch. The judge does not
have to know everything: it has K drafts in front of it, and its job
is to compare them, find overlaps, drop contradictions, preserve
useful details, and emit a single fused answer. A relatively small
local model can sometimes do that well — comparing K versions is
easier than producing the first one.

**But this is not free magic.** The judge is not an objective oracle:

- Roitelet learns *judge-conditioned* preferences. If Qwen is your
  local judge, the rolling-Elo loop will quietly internalise what Qwen
  tends to prefer. Useful for routing under that judge; not a
  universal quality signal.
- A clueless or biased judge will fuse confidently in the wrong
  direction. The fail-closed parse on the winners marker
  (`core/judge.py`) limits how badly a broken judge can corrupt the
  Elo state, but the *content* of a bad fusion is still bad.
- Whether fusion of three OSS candidates beats the strongest single
  paid candidate depends on the prompt class, the candidate
  diversity, and the judge. The answer is empirical, not theoretical.

This is why ablation studies are first-class concerns in this project,
not a "future maybe" — see [docs/EVALUATION.md](docs/EVALUATION.md).
The matrix there proposes single-best vs top-K vs top-K+fusion vs
different judge models, against tasks that span coding, reasoning,
writing, multilingual, factual QA, and long-context summarisation.

---

## Latency and cost tradeoffs

Roitelet's design choices have measurable consequences. They are
worth understanding before you run the system in front of users.

**Latency.** K parallel calls are *not* K times slower than one — the
fan-out runs through `asyncio.gather`. But the wall-clock time of a
turn is bounded by **the slowest selected candidate**, plus the
fusion pass on the local judge. For three local OSS models running
side by side on a laptop CPU, that's a few tens of seconds; for a
mix of one frontier API + two locals, the frontier latency dominates.

**Fusion overhead.** The judge is one extra local generation, with
the system prompt + the K candidate answers as input. On a small
local judge (Qwen 3 8B by default), that adds roughly the same wall
time as one candidate. The result: total time is approximately
`max(candidate_latencies) + judge_latency`.

**Cost.** Local models are free at the marginal token but pay for
themselves in RAM/VRAM and disk. Remote candidates cost what their
provider charges — Roitelet does not arbitrage; it just calls them.
The cost-budget regime (`/cheap <usd>` slash command or
`max_cost_usd` in `RouterPreferences`) drops candidates above the
budget *before* scoring.

### When **not** to use Roitelet

- **Very low-latency chat UX.** A single fast model beats Roitelet's
  fan-out + fusion. If your UI lives or dies by sub-second response
  times, this is the wrong tool.
- **Trivial prompts.** "What's 2+2?" doesn't need three opinions and
  a synthesis. The `trivial` regime surfaces it in telemetry but
  doesn't auto-collapse K — that's a maintainer call.
- **High-volume production traffic where every token matters.**
  Roitelet calls K models and a judge for every turn; the cost is
  multiplicative. A single calibrated model + caching is cheaper.
- **Prompts that must never leave the local machine**, unless you
  explicitly enable local-only mode and use only local candidates.
  See [docs/PRIVACY.md](docs/PRIVACY.md).
- **You just want one provider gateway.** That is exactly LiteLLM's
  job; pick LiteLLM and stop here.

Use Roitelet when you value comparison, redundancy, model diversity,
local synthesis on top of remote answers, or the ability to study
how those tradeoffs play out in your data.

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
├── INSTALL.md          # English install guide
├── INSTALLER.md        # French install guide
├── LISEZMOI.md         # French README mirror
├── MECHANISM.md        # Architecture deep-dive (Mermaid diagrams) — contributors
└── .env.example
```

---

## Documentation map

The docs are split into three tiers — pick the one that matches what
you're trying to do.

### Tier 1 — Users (you want to *run* Roitelet)
- **[README.md](README.md)** / **[LISEZMOI.md](LISEZMOI.md)** — what
  Roitelet is, why it exists, 5-minute quickstart.
- **[INSTALL.md](INSTALL.md)** / **[INSTALLER.md](INSTALLER.md)** —
  full installation guide (conda, venv, Docker).

### Tier 2 — Tech (you want to *use* Roitelet's features)
- **[docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md)** — wire any
  OpenAI-compatible paid LLM (ChatGPT, Mistral, Together, …).
- **[docs/ADDING_LOCAL_LLM.md](docs/ADDING_LOCAL_LLM.md)** — bring
  your own GGUF via Ollama or `llama-server`.
- **[docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md)** — set up
  image generation (DALL-E, local Stable Diffusion, …).
- **[docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md)** — drop files,
  ingest, query your personal knowledge base. Includes the
  Karpathy-style 2-D embedding scatter.
- **[docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md)** — `/image`,
  `/personal`, `/local`, `/cheap`, `/k`, `/help` per-turn overrides.

### Tier 3 — Contributors (you want to *modify* Roitelet)
- **[MECHANISM.md](MECHANISM.md)** — full architectural walk-through
  with Mermaid diagrams. Routing math, regimes, Elo loop, the two
  routers, the two capability detectors, image-gen pipeline.

---

## License

Released under the **BSD 3-Clause License** — see [LICENSE](LICENSE).

## Author

[Warith HARCHAOUI](https://www.linkedin.com/in/warith-harchaoui/)
