# How Roitelet Works

A field guide to what happens inside Roitelet when you send it a prompt. This
document complements the README (which explains *what* and *why*) by walking
through the *how* — the modules, the data flow, the scoring, and the feedback
loop. Diagrams are authored in Mermaid and render directly on GitHub.

---

## 1. Component map

Three frontends share the same brain. Only the API talks to the pipeline; the
GUI sits on top of the API over HTTP. The CLI imports the pipeline directly.

```mermaid
flowchart LR
    CLI["cli/main.py<br/>roitelet ask / chat"]
    WEB["web/index.html + web/app.js<br/>served by the API at /"]
    EXT["External clients<br/>OpenAI-compat · MCP"]

    subgraph API["api/main.py — FastAPI :8000"]
        EP1["/api/chat — native"]
        EP2["/v1/chat/completions"]
        EP3["/mcp — JSON-RPC"]
        EP4["/api/conversations<br/>/api/telemetry<br/>/api/settings"]
    end

    subgraph Core["core/"]
        PIPE["pipeline.run_roitelet_chat"]
        ROUTER["router.RoiteletRouter"]
        CAP["capabilities.detect_capabilities"]
        REG["registry.ModelRegistry<br/>(+ rolling Elo)"]
        JUDGE["judge.judge_and_synthesize"]
        FACT["providers.factory.get_provider_client"]
    end

    subgraph Providers["core/providers"]
        OLL["OllamaClient"]
        OAI["OpenAICompatibleClient<br/>(OpenRouter, OpenAI, ...)"]
    end

    subgraph Storage["core/storage.StorageManager"]
        DB1[("data/conversations/")]
        DB2[("data/telemetry/")]
        DB3[("data/runtime/<br/>elo_state.json<br/>settings.json")]
        DB4[("data/bootstrap/<br/>model_priors.json")]
    end

    CLI --> PIPE
    WEB -- "HTTP" --> API
    EXT -- "HTTP" --> API
    API --> PIPE

    PIPE --> ROUTER
    ROUTER --> CAP
    ROUTER --> REG
    PIPE --> FACT
    FACT --> OLL
    FACT --> OAI
    PIPE --> JUDGE
    JUDGE --> FACT
    PIPE --> Storage
    REG -. "loads bootstrap" .-> DB4
    REG -. "60 s live cache" .-> OLL
```

**Key idea**: the *router* picks which models to ask, *providers* dispatch in
parallel, the *judge* (a local model) synthesises one answer, and the
*registry* updates rolling Elo scores so future routing improves.

---

## 2. Single-turn request lifecycle

The whole pipeline lives in `core/core/pipeline.py:run_roitelet_chat`. Every
frontend ends up calling this one function. Below is exactly what happens for
one prompt.

```mermaid
sequenceDiagram
    autonumber
    participant U as Caller
    participant P as pipeline.run_roitelet_chat
    participant S as storage
    participant R as RoiteletRouter
    participant Reg as registry (singleton)
    participant Prov as Providers (3×)
    participant J as judge.judge_and_synthesize

    U->>P: ChatRequest(prompt, preferences, top_k=3)
    P->>S: get_or_create_conversation
    P->>S: append user message
    P->>R: route(prompt, preferences, top_k)
    R->>R: detect_capabilities(prompt)
    R->>Reg: capability_score(model, cap) for each
    R-->>P: RouterDecision (top-K selected)

    par parallel inference (asyncio.gather)
        P->>Prov: client.generate(model_1)
        P->>Prov: client.generate(model_2)
        P->>Prov: client.generate(model_3)
    end
    Prov-->>P: ModelResponse × K

    P->>P: drop errored / empty responses<br/>(fallback to raw set if all failed)
    P->>J: judge_and_synthesize(prompt, valid)
    J->>Prov: local synthesis model (Ollama)
    J-->>P: SynthesisResult + winning_model_ids

    P->>Reg: update_elo(winners, losers, capabilities)
    P->>S: append assistant message with full metadata
    P->>S: save TelemetryRecord
    P-->>U: ChatResponse
```

A few details worth pinning down:

- **Parallel fan-out** uses `asyncio.gather` — the slowest of the K calls sets
  the wall-clock latency, not the sum.
- **Partial failure is tolerated**: if one provider errors, the judge only
  sees the survivors. If *all* fail, the judge still runs (with empty content)
  to keep the response shape consistent.
- **Telemetry records every response**, including failed ones — failures must
  be visible in the audit trail, not hidden.

---

## 3. Capability detection and router scoring

`RoiteletRouter.route` is pure Python — no model calls. It scores every
registered candidate on a blend of *quality* (Elo-adjusted priors weighted by
detected capabilities) and *frugality* (cost + energy + latency), modulated by
user preferences.

```mermaid
flowchart TD
    P[Prompt text] --> CD["detect_capabilities<br/>(lexical keyword scan +<br/>backtick / length heuristics)"]
    CD --> CW["Normalised weights<br/>Σ = 1.0<br/>e.g. coding 0.7, reasoning 0.3"]

    REG[(ModelRegistry)] --> LIST[list_models]
    LIST --> FLT{Filters}
    FLT -->|"independence: drop non-local"| KEEP[Remaining candidates]
    FLT -->|"!allow_vlms ∧ no vision: drop VLMs"| KEEP

    CW --> Q
    KEEP --> Q
    Q["quality = Σ weight_c · capability_score(model, c)"]
    Q --> F["frugality_bonus =<br/>1 / (1 + output_price·100<br/>+ energy_kwh·1000 + latency_s)"]
    F --> FIN["final_score =<br/>raw_power · quality<br/>+ frugality · frugality_bonus<br/>+ indep · local_bonus (0.15)"]
    FIN --> SORT[Sort candidates by final_score]
    SORT --> TOPK[Take top-K → selected_model_ids]
```

**Why per-capability scores?** A model's coding ability and its translation
ability are decoupled. A global Elo would average them away. Roitelet keeps
one rolling adjustment *per capability per model* (`registry._load_elo_state`),
plus a `global` term that contributes at half-weight. The final score for a
prompt is therefore tilted toward models that are strong on the *specific*
capabilities the prompt demands.

---

## 4. The model registry: three sources, in priority order

The registry is rebuilt on every call to `router.route` to pick up new models
without a restart. Sources are merged in order of *decreasing* authority —
earlier sources win on conflict.

```mermaid
flowchart LR
    subgraph S1["1 · Bootstrap (most authoritative)"]
        B["data/bootstrap/model_priors.json<br/>curated benchmark-inspired priors:<br/>capabilities · pricing · latency · energy"]
    end
    subgraph S2["2 · User configuration"]
        U["AppSettingsPayload<br/>selected_ollama_models<br/>paid_openrouter_models<br/>(edited from the web UI)"]
    end
    subgraph S3["3 · Live discovery"]
        L["GET http://&lt;ollama&gt;/api/tags<br/>60 s TTL cache<br/>(warmed at API startup)"]
    end

    B --> M
    U -- "skip if already present" --> M
    L -- "skip if already present" --> M
    M["ModelRegistry.models<br/>model_id → ModelSpec"]

    M --> ES["elo_state[model][capability]"]
    ES --> SCORE["capability_score =<br/>clamp(prior + elo + 0.5·global, 0, 1.5)"]
```

**Why the priority**: bootstrap is curated and reflects real benchmark data,
so it should never be clobbered by a defaulted entry. User config wins over
live discovery because the user explicitly named those models. Live discovery
exists so a fresh `ollama pull foo` shows up in the router within one TTL
window without touching settings.

---

## 5. The Elo feedback loop

After each turn, the judge's winners gain Elo, the losers lose Elo — both
globally and on every capability the prompt invoked. The K-factor is small
(0.04) so individual turns nudge rather than swing.

```mermaid
sequenceDiagram
    participant P as pipeline
    participant J as judge
    participant Reg as registry
    participant D as runtime/elo_state.json

    P->>J: candidate responses + prompt
    J-->>P: SynthesisResult (WINNERS: 1, 3)
    P->>P: winners = winning_model_ids<br/>losers = selected − winners
    P->>Reg: update_elo(winners, losers, capabilities)

    loop for each winner w
        Reg->>Reg: state[w]['global'] += k / |W|
        Reg->>Reg: state[w][cap] += (k / |W|) · weight_cap<br/>(only for cap ∈ KNOWN_CAPABILITIES)
    end
    loop for each loser l
        Reg->>Reg: state[l]['global'] -= k / |L|
        Reg->>Reg: state[l][cap] -= (k / |L|) · weight_cap
    end

    Reg->>D: write elo_state.json
    Note over Reg,D: capability_score clamps to [0.0, 1.5]<br/>→ Elo cannot dominate the prior indefinitely
```

**Two safeguards in this loop**:

1. Only capabilities in `KNOWN_CAPABILITIES` are allowed into the state file
   (`registry.py`). A typo or a novel capability string cannot grow the file.
2. The final `capability_score` clamps to `[0.0, 1.5]`. Even with sustained
   wins, a model's effective score cannot run away — it asymptotes against
   the ceiling.

---

## 6. On-disk layout

Roitelet deliberately avoids a database. Everything is JSON, atomically
written, easy to inspect with `cat` and `jq`.

```mermaid
flowchart TD
    DD["data/ — ROITELET_DATA_DIR"]
    DD --> BOOT["bootstrap/<br/>model_priors.json<br/>(read-only, shipped with repo)"]
    DD --> CONV["conversations/<br/>&lt;uuid&gt;.json<br/>(one file per chat thread)"]
    DD --> TEL["telemetry/<br/>&lt;uuid&gt;.json<br/>(one file per turn)"]
    DD --> RUN["runtime/<br/>elo_state.json — rolling Elo<br/>settings.json — UI-edited config<br/>app_settings.json"]
    DD --> CACHE["cache/<br/>&lt;provider&gt;.jsonl — request cache"]
```

Writes go through `StorageManager._write_json` which uses the
write-temp-then-`os.replace` pattern, so a crash mid-write cannot corrupt an
existing file — readers either see the old version or the new one, never a
half-formed mix.

---

## 7. Where to read next

| Module | Lines | What to look for |
|---|---|---|
| `core/core/pipeline.py` | ~170 | The whole orchestration in one file — start here |
| `core/core/router.py` | ~115 | The scoring formula and the filter logic |
| `core/core/registry.py` | ~380 | Bootstrap loading, live discovery, Elo update |
| `core/core/capabilities.py` | ~125 | Keyword lists + normalisation |
| `core/core/judge.py` | ~95 | Prompt building, WINNERS parsing, synthesis fallback |
| `core/providers/openai_compatible.py` | ~120 | The contract every remote provider must satisfy |
| `api/main.py` | ~300 | All three API surfaces in one file |
| `tests/test_pipeline.py` | ~230 | Worked example of running the pipeline end-to-end with stubs |
