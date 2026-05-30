# Roitelet LLM

> **A local-first LLM routing and fusion workbench.** Several models
> answer the same prompt; a local model fuses the best parts of each
> answer; the user sees one response.

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

## How it works

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

Per turn, in plain words:

1. **Router.** Score every registered model (local + optional remote)
   on capability priors, rolling Elo, regimes (cost budget, trivial,
   long-context). Take the top-K (default K=2 — empirical sweet spot,
   see [docs/EVALUATION.md §4.3](docs/EVALUATION.md)).
2. **Fan-out.** K candidates answer in parallel via `asyncio.gather`;
   one slow provider doesn't block the others.
3. **Judge.** A local Ollama model reads the K answers — anonymised
   and shuffled — and fuses them.
4. **Elo update.** Winners gain Elo on the prompt's capabilities;
   losers lose. Bounded; no runaway.

Optional **pseudonymization** wraps the whole thing: a local model
swaps PII (names, places, IDs, …) for plausible same-locale
substitutes before remote calls, then restores them on the way back.
Audit trail attached to every turn. See
[docs/PSEUDO.md](docs/PSEUDO.md).

Every step is inspectable. The router decision, the candidate
replies, the judge's reasoning, and the Elo state are plain JSON
files on disk — nothing is hidden behind an opaque service.

---

## What this is good for

- **Comparing model families** on the same prompt without juggling
  three SDKs.
- **Running a local synthesis pass** on top of remote candidate
  answers — final word from a model you control.
- **Hiding personal info** before sending to a cloud model
  (`pseudonymize`).
- **Studying tradeoffs** between cost, latency, privacy, and answer
  quality, with the data trail to make those studies reproducible.

Caveats up front: the fused answer is not guaranteed to beat the
strongest single candidate on every prompt class — that's exactly
what the ablation roadmap in
[docs/EVALUATION.md](docs/EVALUATION.md) is designed to measure. The
synthesis judge is not an objective oracle; Roitelet learns
*judge-conditioned* preferences. And Roitelet is local-**first**,
not local-**only** — see [docs/PRIVACY.md](docs/PRIVACY.md) for the
precise distinction and the local-only switch.

---

## Three surfaces, same features

The same operations work across all three surfaces:

| Surface | How to access | Per-turn preferences |
|---|---|---|
| **Web** (GUI) | `http://localhost:8000/` after `./start.sh` | Sliders icon next to the send button |
| **CLI** | `roitelet ask "…"` / `roitelet chat` | `--top-k`, `--independence`, `--pseudonymize`, `--max-cost-usd`, `--verbose` |
| **API** | Native `POST /api/chat`, OpenAI-compatible `POST /v1/chat/completions`, MCP `POST /mcp` | `preferences.{independence, pseudonymize, top_k, max_cost_usd, …}` in the JSON body |

For OpenAI clients (Python SDK, LiteLLM, Continue.dev, …) Roitelet
is a drop-in target — see [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md).
Roitelet-specific knobs ride on `metadata.roitelet.{…}` so you
don't lose anything by using the OpenAI shape.

---

## Features at a glance

- **Hybrid routing** — capability priors + rolling Elo + regimes
  (cost budget, trivial-prompt, long-context, ambiguous,
  capability-dominant). Optional learned matrix-fac router behind
  `ROITELET_ROUTER=mf`.
- **Parallel top-K fan-out** — bounded by the slowest candidate.
  K=2 default (`ROITELET_DEFAULT_TOP_K` overrides).
- **Local synthesis** — anonymised, shuffled candidates → fused
  answer; replaceable judge.
- **Pseudonymization** — opt-in PII swap before remote calls,
  restore after. Fail-closed; full audit. See
  [docs/PSEUDO.md](docs/PSEUDO.md).
- **Personal mode** — drop files into `data/personal/inbox/`; small
  corpora inject inline, large ones switch to embedding retrieval.
  2-D PCA scatter. See [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md).
- **Multimodal attachments** — images, PDFs, audio extracted
  locally (Ollama VLM, kreuzberg, whisper.cpp + NeMo) before the
  text pipeline.
- **Image generation** — K=1 routing to the strongest registered
  image-gen model (no fusion — image ensembling isn't defined).
- **Universal extension** — any provider with an OpenAI-compatible
  endpoint registers in three settings fields.
- **Standardized endpoints** — native `/api/chat`,
  OpenAI-compatible `/v1/chat/completions` + `/v1/images/generations`,
  MCP JSON-RPC `/mcp`.
- **Slash commands** — routes only (`/image`, `/speech`,
  `/personal`, `/help`). Per-turn preferences are visible controls,
  not slashes. See [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md).
- **Local telemetry** — per-turn JSON: router decision, every
  candidate (failures included), synthesis, winners.
- **Optional Bearer-token gate** — `ROITELET_API_TOKEN` locks every
  endpoint except `/healthz` and the static SPA. Off by default to
  preserve the single-user-localhost UX.

---

## Quick start

```bash
# Install runtime deps (conda or venv both work).
conda env create -f environment.yaml && conda activate roitelet-llm
# or:
python -m venv .venv && source .venv/bin/activate && pip install -e .

# Pull a model bundle (Minimal ~3 GB or Full local ~15 GB).
chmod +x scripts/pull_defaults.sh
./scripts/pull_defaults.sh --minimal

# Optional: configure remote providers.
cp .env.example .env       # then edit to add API keys

# Run.
./start.sh                 # binds 127.0.0.1:8000 by default
```

Then open `http://localhost:8000/` for the web UI, or use the CLI:

```bash
roitelet ask "Explain quicksort in one paragraph."
roitelet ask --pseudonymize "Email Marie Dupont at marie@orange.fr about Q3."
roitelet settings get
roitelet chat --independence    # interactive REPL, local-only
```

For installation deep-dives (Docker, venv variants, profile
comparison): [INSTALL.md](INSTALL.md) (English),
[INSTALLER.md](INSTALLER.md) (French).

---

## Latency and cost in one paragraph

A turn's wall-clock is `max(candidate_latencies) + judge_latency`
because the fan-out runs through `asyncio.gather`. Local models are
free at the marginal token but cost RAM/VRAM. Remote candidates cost
what their provider charges — Roitelet doesn't arbitrage, it just
calls them. The cost-budget regime (`--max-cost-usd` flag, composer
slider, or `preferences.max_cost_usd`) drops candidates above the
budget *before* scoring. **K=2 is the empirical sweet spot** on the
held-out dataset (see below); K=1 leaves quality on the table, K=3
doubles wall-clock for ~+1 pp.

### Has fusion been measured?

End-to-end ablation on the 25-prompt mixed dataset, local-only (3
small OSS candidates, `qwen3:8b` judge), graded by DeepEval
`GEval(correctness, threshold=0.6)`:

| K | mean correctness | pass (≥0.6) | mean wall-clock per prompt | judge share |
|---|---|---|---|---|
| 1 | 0.87 | 23 / 25 | 32.1 s | 70 % |
| 2 | **0.95** | **25 / 25** | 55.9 s | 74 % |
| 3 | 0.96 | **25 / 25** | 112.1 s | 73 % |

> "Mean wall-clock per prompt" is the average end-to-end latency
> for one user prompt: router + parallel candidate fan-out (bounded
> by the slowest candidate) + judge + telemetry. Averaged across
> the 25 prompts.

K=1 → K=2 is **+8 pp mean correctness** for +24 s of wall-clock.
K=2 → K=3 hits a quality ceiling. Full per-category breakdown,
caveats, judge-swap ablation (3B vs 4B vs 8B), and the open
follow-ups live in [docs/EVALUATION.md](docs/EVALUATION.md).

---

## When **not** to use Roitelet

- **Very low-latency chat UX.** A single fast model beats fan-out +
  fusion. If your UI lives or dies by sub-second response times,
  pick the wrong tool.
- **Trivial prompts.** "What's 2+2?" doesn't need three opinions.
- **High-volume production traffic.** Roitelet calls K models + a
  judge per turn; cost is multiplicative. A single calibrated model
  + caching is cheaper.
- **You just want one provider gateway.** That's
  [LiteLLM](https://github.com/BerriAI/litellm)'s job.

Use Roitelet when you value comparison, redundancy, model diversity,
local synthesis on top of remote answers, or auditable studies of
the tradeoffs.

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

## Documentation

Three tiers — pick by intent:

### Run it
- [README.md](README.md) / [LISEZMOI.md](LISEZMOI.md) — what it is,
  quickstart (this file).
- [INSTALL.md](INSTALL.md) / [INSTALLER.md](INSTALLER.md) — full
  installation guide (conda, venv, Docker).

### Use a feature
- [docs/OPENAI_COMPAT.md](docs/OPENAI_COMPAT.md) — drop Roitelet
  into existing OpenAI tooling.
- [docs/SLASH_COMMANDS.md](docs/SLASH_COMMANDS.md) — route slashes
  + the visible-control matrix.
- [docs/PSEUDO.md](docs/PSEUDO.md) — pseudonymization (PII
  taxonomy, fail-closed contract, audit).
- [docs/PERSONAL_MODE.md](docs/PERSONAL_MODE.md) — personal-RAG
  workflow + the embedding viz.
- [docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md) — wiring
  image-gen providers.
- [docs/PRIVACY.md](docs/PRIVACY.md) — what's stored on disk,
  what goes over the network, the four privacy modes.
- [docs/EVALUATION.md](docs/EVALUATION.md) — standing ablation
  roadmap with results.

### Extend it
- [docs/ADDING_LOCAL_LLM.md](docs/ADDING_LOCAL_LLM.md) — bring your
  own GGUF.
- [docs/ADDING_PAID_LLM.md](docs/ADDING_PAID_LLM.md) — wire any
  OpenAI-compatible paid provider.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — internals
  deep-dive (Mermaid diagrams, routing math, Elo loop, regimes,
  matrix-fac router).

---

## How Roitelet differs from neighbouring projects

| Project | Primary role | How Roitelet differs |
|---|---|---|
| [LiteLLM](https://github.com/BerriAI/litellm) | OpenAI-compatible gateway over many APIs | Roitelet is narrower: local-first fan-out + local synthesis + inspectable Elo. LiteLLM is a candidate Roitelet could call. |
| [OpenRouter](https://openrouter.ai) | Hosted multi-model marketplace | Roitelet runs on your machine. OpenRouter is a candidate provider, not a replacement. |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | Cost-aware strong-vs-weak routing trained on preferences | Roitelet does top-K + fusion, not binary routing. RouteLLM's `mf` slots behind Roitelet's Router Protocol. |
| [LangChain](https://www.langchain.com) / LangGraph | LLM-orchestration frameworks | Roitelet is an end-user system, not a framework. |
| [DSPy](https://github.com/stanfordnlp/dspy) | Programming model for prompt-pipeline optimisation | Roitelet routes and fuses at inference time; rolling Elo is its only online "optimisation". They can coexist. |
| Single-model chat clients | One model in, one answer out | Roitelet trades simplicity and latency for comparison + redundancy + synthesis. |

Roitelet is a **workbench**, not a gateway, marketplace, framework,
or chat client.

---

## Folder layout

```text
roitelet-llm/
├── core/               # Router, registry, pipeline, judge, pseudo, multimodal
├── api/                # FastAPI app (native + OpenAI-compat + MCP)
├── web/                # Vanilla-JS GUI (served at /)
├── cli/                # `roitelet` console-script entry point
├── docs/               # Topic guides — architecture, eval, features
├── data/bootstrap/     # Default Elo + capability priors
├── scripts/            # pull_defaults.sh, vendor_web_assets.sh
├── tests/              # Unit + opt-in DeepEval suite
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
