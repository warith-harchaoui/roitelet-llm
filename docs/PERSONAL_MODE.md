# Personal mode

Drop your own files into Roitelet and have the LLM answer questions
about them. Combines two patterns:

- **Karpathy-style LLM wiki** — a small curated folder of
  Markdown notes, concatenated and injected into the prompt as
  long-context.
- **Local RAG** — when the folder grows large, switch automatically
  to embedding-based retrieval (top-K chunks injected instead of the
  whole corpus).

Both run **fully local** by default. Conversions reuse the existing
multimodal extractors (whisper.cpp transcription + NeMo Sortformer
diarisation, Ollama VLM image captioning, kreuzberg PDF text); the
embedding model is `nomic-embed-text` via Ollama.

---

## Two folders, one feature

```
<ROITELET_DATA_DIR>/personal/
├── inbox/        ← drop raw files here (audio, image, PDF, .md, .txt)
├── wiki/         ← converted text + your hand-written notes
└── index.json    ← manifest tracking what's been ingested
```

- The **inbox** is where you drop new things. Use whatever filename
  you want — Roitelet converts a slugified copy into `wiki/`.
- The **wiki** holds plain `.md` files. Anything you write directly
  here is treated as first-class content (the Karpathy pattern). The
  auto-converted entries get a provenance header (`_Auto-converted
  from <source> on <UTC timestamp>._`) so you can always trace where
  they came from.
- The **manifest** keeps ingestion idempotent: re-running `ingest`
  doesn't reprocess files. Pass `--force` to override.

---

## CLI

```bash
# See what's there
python -m cli personal status

# Convert any new inbox files (idempotent)
python -m cli personal ingest

# Re-convert everything from scratch (after an extractor upgrade, say)
python -m cli personal ingest --force

# List the wiki entries
python -m cli personal list

# Ask a question with the wiki injected
python -m cli personal ask "what did I write about Q3 revenue?"
```

The interactive REPL (`python -m cli chat`) also accepts the
`/personal` slash command per turn — see below.

---

## REST API

| Endpoint | Body | Returns |
|---|---|---|
| `GET /api/personal` | — | Counts and the active context strategy (`wiki`, `rag`, or `empty`). |
| `POST /api/personal/ingest?force=false` | empty JSON | Per-file results + updated status. |
| `POST /api/chat` with `"prompt": "/personal ..."` | standard | Personal context block prepended before fan-out. |

Both endpoints respect `ROITELET_API_TOKEN` when set.

---

## Slash command (chat)

The chat endpoint understands `/personal <question>`:

```
/personal what did I write about RAG?
```

The handler:
1. Strips the `/personal` prefix.
2. Loads the wiki (full inline for small corpora, top-K retrieval for
   large ones).
3. Prepends the formatted block in front of the original question.
4. Runs the standard top-K fan-out + fusion judge against the
   augmented prompt.

Combine with other slash commands as you would expect:

```
/local /personal explain my notes
```

— forces independence mode (local models only) and injects the wiki.

---

## Web control room

Open the **Settings** sheet (gear icon, sidebar bottom). The first
panel is the **Personal knowledge base** card. It shows:

- how many files are in `inbox/` vs `wiki/`,
- the current context mode (`wiki` / `rag` / `empty`),
- an **Ingest inbox** button that walks the inbox and converts new
  files.

Beneath, the standard credential fields apply to paid LLMs that might
be used as candidates when the personal-augmented prompt fans out.

---

## When the wiki gets big — RAG kicks in

The size threshold lives at
`core.personal._WIKI_MAX_INLINE_CHARS` (32 000 chars ≈ 8 k tokens):

- **Below the threshold** — the entire wiki is concatenated and
  injected in front of the question. Karpathy-style, no retrieval,
  trivial latency.
- **Above the threshold** — chunks of size
  `_RAG_CHUNK_CHARS` (default 1 200 chars) with
  `_RAG_CHUNK_OVERLAP` (200 chars) are embedded via Ollama's
  `nomic-embed-text`; the top `_RAG_TOP_K` (default 5) chunks are
  injected.

### Persistent index + ANN (turbovec)

Roitelet's RAG path is **not** a re-embed-on-every-query implementation:

- **Embeddings are computed once per wiki revision.** The first
  retrieval (or `personal ingest`) walks the wiki, embeds every chunk
  via `nomic-embed-text`, and persists three sidecar files next to
  the wiki:
  - `.rag_index.json` — fingerprint + chunk text manifest
  - `.rag_embeddings.npy` — canonical dense `(N, dim)` float32 matrix
  - `.rag_index.tq` — compressed [turbovec](https://github.com/RyanCodrai/turbovec)
    `IdMapIndex` (when the `[personal]` extra is installed)
- **Subsequent queries embed only the question** and search the
  cached index. Cold-start is dominated by the first wiki ingest;
  every later turn is one embedding call + one ANN lookup.
- **Auto-invalidation.** The sidecar JSON carries a SHA-256 fingerprint
  over each wiki file's `(name, mtime, size)` plus the chunking knobs.
  Any change — adding, editing, deleting a wiki file, or changing
  `_RAG_CHUNK_CHARS` — drifts the fingerprint and triggers a rebuild
  on the next call.
- **Two search backends, same on-disk format.** With
  `pip install -e .[personal]` the compressed turbovec index is used
  (~16× embedding compression, sub-millisecond search at 100k+
  chunks). Without it, the pure-numpy brute-force scan over the
  `.npy` matrix runs — still fast on personal-scale corpora.

If the embedding call fails (server down, model not pulled), Roitelet
**skips** the personal-context injection rather than guessing — the
turn runs as a normal chat. The CLI / API report mode=`rag` in either
case; the absence of injected context surfaces in the assistant
response (it says "I don't have that in your notes").

---

## What this is not

- **Not** a server-grade vector database. The persistent on-disk
  index handles personal-scale corpora well (tested up to a few
  thousand chunks); swap in pgvector / Qdrant / LanceDB if you scale
  to a corporate wiki.
- **Not** a multi-user knowledge base. Everything is one folder, one
  user, one machine. Multi-tenant support would need a per-user
  `data_dir`.
- **Not** an editor. Use whatever you use for `.md` files. The CLI
  doesn't shell out to `$EDITOR`; that would belong in a separate
  `personal edit` subcommand if there's demand.

---

## See also

- [`docs/SLASH_COMMANDS.md`](SLASH_COMMANDS.md) — full slash-command
  catalogue.
- [`MECHANISM.md`](../MECHANISM.md) — full architecture, including
  the personal-mode pipeline integration.
