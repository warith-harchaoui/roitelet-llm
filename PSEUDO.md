# Pseudonymization

> **Status:** implemented in `core/pseudo.py`, wired through pipeline,
> available on every surface (GUI / CLI / API / MCP).
> **Origin:** request 2026-05-30.

A per-conversation toggle that lets a Roitelet turn:

1. **Forward pass.** Before the prompt reaches the router, a local
   Ollama model rewrites every piece of **personally-identifying
   information** into a **plausible same-origin substitute**. The
   rewritten prompt is what the remote candidates see; the original
   prompt is what gets persisted in the conversation log.

2. **Reverse pass.** After the synthesis judge fuses the candidate
   answers, every substitute is mapped back to the user's original
   string. Literal substring replace covers most of it; an LLM repair
   pass handles inflected forms when the literal pass leaves orphans
   (e.g. `Camille Lefèvre` came back as `Mme Lefèvre`).

Both passes use the same local model and produce an audit trail —
the GUI shows it as a collapsible diff under the user bubble, the
CLI prints it when `--verbose` is on, and every API response carries
it on `ChatResponse.pseudonymization`.

---

## How to turn it on

| Surface | Per-turn | Persisted default |
|---|---|---|
| Web | Sliders icon next to the send button → "Pseudonymize remote calls". | Settings sheet → "Pseudonymize remote calls (PII swap)". |
| CLI | `--pseudonymize` / `--no-pseudonymize` on `roitelet ask` / `chat`. | `roitelet settings set enable_pseudonymization true`. |
| API | `preferences.pseudonymize: true` on `POST /api/chat`; `Form('pseudonymize', 'true')` on `/api/chat/multimodal`. | `POST /api/settings` with `enable_pseudonymization: true`. |
| MCP | `pseudonymize: true` on the `roitelet.chat` tool call. | (uses the persisted setting if the field is absent.) |

The redactor model defaults to your local synthesis model. Override
with `pseudo_model_id` in the settings if you want a cheaper or
faster local model for the redaction pass.

---

## Threat model — one sentence

This feature reduces casual provider-side PII exposure (provider
audit logs, retained training corpora, careless logging). It does
**not** anonymize the user against a hostile counterparty: a
determined adversary with control of the provider can infer identity
from context. The README, the docs, and the GUI's toast all say so.

---

## PII taxonomy

The forward prompt enumerates a precise list of categories the model
must address. The slugs are the values of `core.schemas.PIIKind`:

| Category | Slug | Example | Substitution rule |
|---|---|---|---|
| Person | `person_name` | `Marie Dupont` | Same locale, same gender when inferable. |
| Username | `username` | `@mlefevre` | Structurally-similar handle. |
| Date of birth | `date_of_birth` | `1989-04-12` | Shift ±2 years; preserve format. |
| Place | `place_name` | `Lyon` | Same country, comparable size. |
| Street address | `street_address` | `42 rue de la Paix, 75002 Paris` | Substitute street + number + postcode; keep country. |
| GPS | `coordinates` | `45.760, 4.842` | Shift ~1°; stay in country. |
| Organization | `organization` | `Acme Corp` | Same sector and country. |
| Job title | `job_title` | `CEO of Acme` | Only when the title narrows the person. |
| Email | `email` | `marie@orange.fr` | Substitute user part; keep domain class. |
| Phone | `phone` | `+33 6 12 34 56 78` | Substitute digits; keep country code + format. |
| Social URL | `url_handle` | `github.com/mlefevre` | Substitute the handle; keep the host. |
| IP | `ip_address` | `192.168.1.42` | Substitute address; same locale. |
| Gov. ID | `national_id` | `987-65-4321` | Digit-for-digit; preserve checksum shape. |
| Financial | `financial_id` | `IBAN FR76 …` | Digit-for-digit; preserve format. |
| Medical | `medical_id` | `MRN-44291` | Substitute structurally. |
| Employee | `employee_id` | `EMP-7714` | Substitute structurally. |
| Account | `account_id` | `TICKET-22-9981` | Substitute structurally. |
| Vehicle | `vehicle_id` | `AB-123-CD` | Preserve country plate prefix. |
| Fallback | `other_identifier` | (anything PII-class not above) | LLM-judged substitution. |

What the model is told **not** to substitute:

- Programming-language names, library names, function names, file
  paths, file extensions.
- Public-documentation URLs (`docs.python.org`, `en.wikipedia.org`).
- Mathematical / scientific terms, units of measurement, well-known
  constants.
- Anything inside Markdown fenced code blocks or inline code (code is
  technical content; substituting an identifier inside it breaks the
  answer).
- Names of historical / public figures when the prompt is asking
  about them as historical figures. When in doubt, the model
  substitutes and the user can disable pseudonymization for that
  one prompt with the per-turn toggle.

---

## Fail-closed contract

If the local model is unreachable, returns malformed JSON, or
produces a mapping that violates any invariant:

- every `original` must appear literally in the input prompt;
- every `substitute` must appear literally in the rewritten prompt;
- no `original` may leak into the rewritten prompt;
- no `original == substitute` no-op;

then `core.pseudo.PseudonymizationError` fires and the pipeline
**aborts the turn**. The unredacted prompt is never sent. The user
sees a clear error and can either retry, fix the prompt, or disable
the toggle. A safety toggle that silently fails open is worse than
no toggle.

---

## Reverse pass

Two stages, in order:

1. **Literal pass** — substitute → original substring replace,
   ordered by descending substitute length so a longer substitute
   that contains a shorter one wins. Deterministic, cheap, covers
   the typical case.
2. **LLM repair pass** *(automatic when needed)* — fires only when
   a multi-token substitute's distinctive token survived the
   literal pass (e.g. `Camille Lefèvre` paraphrased to
   `Mme Lefèvre`). One repair call, no recursion. If the repair
   pass fails or returns empty, we keep the literal-only output
   and flag `repair_used=False` in the audit so the operator can
   investigate.

The `tests/test_pseudo.py::TestRestoreText` family covers both
stages with monkeypatched models so the suite stays offline.

---

## Audit trail

Every turn that ran with `pseudonymize=true` carries a
`PseudonymizationAudit` on the response:

```jsonc
{
  "mappings": [
    {"original": "Marie Dupont", "substitute": "Camille Lefèvre", "kind": "person_name"},
    {"original": "Lyon",         "substitute": "Toulouse",        "kind": "place_name"}
  ],
  "pseudonymized_prompt": "Email Camille Lefèvre about the Toulouse meeting.",
  "model_id": "ollama/qwen3:8b",
  "forward_latency_s": 0.74,
  "reverse_latency_s": 0.00,
  "repair_used": false
}
```

The GUI renders this as a collapsible card under the user bubble.
The CLI prints it when `--verbose` is set. The conversation log
stores the **original** prompt as the user message content and the
audit as `metadata.pseudonymization`, so on reload the chat reads
naturally and the diff is still inspectable.

---

## Code surface

| File | Role |
|---|---|
| `core/pseudo.py` | Forward + reverse + validation; one local Ollama call per pass. |
| `core/schemas.py` | `PIIKind`, `PIIMapping`, `PseudonymizationAudit`; `RouterPreferences.pseudonymize`; `AppSettingsPayload.enable_pseudonymization` + `pseudo_model_id`; `ChatResponse.pseudonymization`. |
| `core/pipeline.py` | Calls `pseudonymize_prompt` before fan-out, `restore_text` after the judge, attaches the audit to both messages. |
| `api/main.py` | `preferences.pseudonymize` on chat; `pseudonymize` Form field on multimodal. |
| `core/mcp.py` | `pseudonymize` on the `roitelet.chat` tool schema. |
| `cli/main.py` | `--pseudonymize` / `--no-pseudonymize` flags; verbose audit printer. |
| `web/app.js` + `web/index.html` | Per-turn sliders popover, settings checkbox, audit affordance. |
| `tests/test_pseudo.py` | Offline unit tests — fail-closed contract, taxonomy, reverse-pass repair. |
| `tests/eval/test_pseudo_quality.py` | Opt-in DeepEval correctness floor on the treatment arm. |

---

## What this trade looks like to the user

| Prompt class | Expected behaviour with pseudo ON |
|---|---|
| Math / coding / general knowledge | No-op or near-no-op; the mapping table is usually empty. |
| Personal data ("email Marie at Lyon") | Substitutes leave the box; originals come back; audit shows what happened. |
| Named-entity QA ("what did Napoleon do in 1812?") | Wrong answer expected — the model is forced to redact the entity the prompt is about. Use the per-turn toggle to switch off for this one prompt. |

The eval ablation in `tests/eval/test_pseudo_quality.py` is how we
measure this delta empirically — opt-in (`pytest -m eval`).
