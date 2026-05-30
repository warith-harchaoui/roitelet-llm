# Pseudonymization — design note + roadmap

> **Status:** proposal. Not implemented. Tracking issue: TBD.
> **Owner:** Warith.
> **Origin:** request 2026-05-30 — "swap PII before remote calls, swap it
> back when the answer returns, transparent to the user."

---

## 1. What the feature is

A per-conversation **Pseudonymization** toggle (sidebar checkbox + persisted
setting + `/pseudo` slash command for the current turn). When ON, every
prompt that is about to leave the box for a remote candidate is first
rewritten by a **local** LLM that replaces named entities — people,
places, organizations, identifiers — with **plausible same-origin**
substitutes:

- *Marie Dupont* (FR) → *Camille Lefèvre* (FR).
- *Berlin* → *Hamburg*.
- *Acme Corp.* → *Globex Industries*.
- `marie.dupont@orange.fr` → `camille.lefevre@orange.fr`.

The same local model holds a **per-turn mapping table** keyed by
substring. After the remote candidates answer and the synthesis judge
fuses them, a second local pass walks the fused text and **reverses**
every substitution, so the user sees the original names back.

The diff — what got pseudonymized and what the substitute looked like —
is rendered as a collapsible affordance below the user bubble (think
`details`-style "show what left the box"). Trust comes from inspectable
state, not from invisible safety theater.

When the toggle is OFF, the pipeline is unchanged: this is opt-in by
design, because the substitution adds latency and can change answer
quality on prompts where the named entities carry real meaning (e.g.,
"What did Napoleon do in 1812?" — pseudonymizing *Napoleon* makes the
answer wrong).

---

## 2. Why this is worth doing

Roitelet's current privacy story (see `docs/PRIVACY.md`) is honest about
the trade: in **local-first** mode, the user prompt is sent verbatim to
every selected remote candidate. The mitigation list today is binary —
either you flip *Local-only* and lose remote candidates entirely, or you
accept that the prompt leaves the box as-is. Pseudonymization adds a
**third option**: keep the remote candidates, but strip the high-signal
PII that a typical user actually wants out of the audit trail of a paid
provider. That fits the local-first thesis: a small local move that
materially changes the privacy/quality tradeoff curve without giving up
the strongest external models.

The threat model this addresses is **the routine provider audit log /
data-retention surface**, not a targeted adversary. A determined attacker
with control of the provider can still infer who the prompt is about
from context. We will say so in the README, the toast, and PRIVACY.md —
the feature reduces casual PII exposure, it does not make the user
anonymous to a hostile counterparty.

---

## 3. UX sketch

### 3.1 Composer

- New compact toggle next to the existing *Allow vision-language* in
  Settings — label **"Pseudonymize remote calls"**, persisted as
  `enable_pseudonymization` in `AppSettingsPayload`.
- Per-turn override: prefix the prompt with `/pseudo` to force it ON
  for the next turn even if the setting is OFF, and `/nopseudo` to
  force OFF when the setting is ON. Same flavour as `/local`,
  `/cheap`, `/k`.
- Status pill in the header turns from "Ready" to "Ready · pseudo" so
  the active state is visible without opening Settings.

### 3.2 Audit trail in the message bubble

Below the user bubble, render a collapsible **"View pseudonymization
diff"** that shows two columns: the original tokens and the substitutes
the local model picked. This is the user-visible proof that the feature
actually fired and what landed on the wire.

### 3.3 Failure mode

If the local pseudonymizer can't reach its model, or returns a
malformed mapping (see §5.3), the turn **does not silently fall through
to an unredacted remote call**. Instead it surfaces a clear inline error
("Pseudonymization unavailable — turn aborted. Disable to send the
original prompt.") and the user must explicitly retry without the
toggle. Silent fallback would convert a safety toggle into safety
theater.

---

## 4. Data flow

```
┌──────────────┐    raw prompt                    ┌──────────────┐
│  composer    │ ─────────────────────────────▶   │  pipeline    │
└──────────────┘                                  │  (FastAPI)   │
                                                  └──────┬───────┘
                                                         │
                            pseudonymization OFF?        │
                             ─── yes ─── continue ───────┤
                                                         │ pseudonymization ON
                                                         ▼
                                              ┌──────────────────────┐
                                              │  pseudo.forward()    │
                                              │  local Ollama call   │
                                              │  returns:            │
                                              │   - rewritten prompt │
                                              │   - mapping {orig:   │
                                              │       substitute}    │
                                              └──────────┬───────────┘
                                                         │
                                       rewritten prompt │
                                                         ▼
                          ┌──────────────────────────────────────┐
                          │  router → fan-out → candidates       │
                          │  (remote API calls see substitutes)  │
                          └──────────────────┬───────────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  judge synthesizes   │
                                  │  fused answer (still │
                                  │  contains            │
                                  │  substitutes)        │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  pseudo.reverse()    │
                                  │  string-replace each │
                                  │  substitute back to  │
                                  │  the original.       │
                                  │  Falls back to a     │
                                  │  second local LLM    │
                                  │  pass if the         │
                                  │  fused text uses an  │
                                  │  inflected form.     │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  ChatResponse        │
                                  │  + pseudonymization  │
                                  │    audit payload     │
                                  └──────────────────────┘
```

Persistence: the **mapping is per-turn** (lives inside the request
lifecycle, never on disk) and the **audit diff** is stored as
`ConversationMessage.metadata.pseudonymization = {"mappings": [...], "model_id": ...}`
so future sidebar loads can re-render the affordance. The original
prompt is what we persist in the conversation log — substitutes live
only in the metadata column, so reloading the chat shows the user the
same text they typed.

---

## 5. Hard parts (write these down so we don't pretend they're easy)

### 5.1 Reversibility under model paraphrase

The synthesis judge is free to **rephrase, decline, or inflect** any
substitute. *"Camille Lefèvre"* may come back as *"Mme Lefèvre"*, or
*"Camille"*, or *"she"*. Naive string replace will miss those forms.
Two-stage reverse pass:

1. **Literal pass** — exact substring replacement for each substitute
   we issued. Cheap, deterministic, covers ≥80% of mentions in
   practice.
2. **LLM repair pass** — feed the literally-reversed text + the mapping
   back through the same local model with a prompt of the form
   *"Restore the original names according to this table. Preserve every
   inflection, contraction, possessive. Do not invent new mappings."*

The repair pass runs only when the literal pass left orphan substitutes
unmatched in the fused text and the user has visible-PII tolerance for
a second local hop. Otherwise we accept the literal-only output and
note the partial reverse in the audit trail.

### 5.2 Origin-preserving substitution

*"Replace a French name with another French name"* is the part that
makes this feel honest. The local Ollama model handles this with a
prompt that explicitly asks for **same-locale**, **same-gender** (when
inferable), **same-formality-register** substitutes. We don't ship our
own name banks because:

- Bundling locale-specific name lists triggers licensing and
  completeness concerns we don't want to own.
- The LLM is already on disk; a one-shot rewrite call is cheap (~1s on
  qwen3:8b for a typical chat prompt).

Acceptable failure mode: if the model produces a substitute that's the
*same* name in different casing, or a clearly wrong-locale name, the
audit diff makes that visible and the user retries. We measure this in
§7.

### 5.3 What counts as PII

Out of scope to define a perfect ontology. The local model is asked to
substitute:

- Person names (full, first, last, nicknames).
- Place names (cities, regions, countries, addresses).
- Organization names.
- Email addresses and phone numbers — substitute the *user part*, keep
  the domain/provider class so the answer can still reason about
  *"Orange's email format"* or *"a French mobile number"*.
- Free-form identifiers that the model can flag (employee IDs, ticket
  numbers).

Explicitly **not** substituted: technical content (file paths, code,
URLs to public docs, function names, library versions). The substitution
prompt enumerates this so the model doesn't over-redact and break the
answer.

### 5.4 Test surface for ablation

Quality regression matters. We add an evaluation mode (see §7) that
runs the full DeepEval suite twice — once with pseudonymization off,
once on — and reports the delta. Two scenarios where we know quality
will move:

- **Named-entity QA** ("What did Napoleon do in 1812?") — pseudonymizing
  *Napoleon* destroys the answer. Expected regression. The user has to
  know that pseudo is the wrong toggle for this prompt class. The
  README will say so.
- **Coding / math** — should be a no-op delta. If it isn't, the
  substitution is leaking into technical content.

---

## 6. API + code surface

| Surface                          | Change                                                     |
|----------------------------------|------------------------------------------------------------|
| `core/pseudo.py` (new)           | `forward(prompt) -> (rewritten, mapping)` and `reverse(text, mapping) -> text`. Holds prompts, talks to `core.registry` for the local model id. |
| `core/schemas.py`                | Add `RouterPreferences.pseudonymize: bool = False`. Add a `PseudonymizationAudit` schema (list of `{original, substitute, kind}`). Add it to `ConversationMessage.metadata` and the API response payload. |
| `core/commands.py`               | `/pseudo` and `/nopseudo` slash overrides — mirror the existing `/local` plumbing. |
| `core/pipeline.py`               | Call `pseudo.forward()` before router fan-out when active; call `pseudo.reverse()` on `synthesis.content` before returning. Audit attached to `assistant_payload['metadata']`. |
| `core/config.py`                 | `pseudo_model_id: str = "qwen3:8b"` (default to the same judge model so we don't bloat the model footprint). |
| `core/storage.py`                | No schema change — audit lives inside the existing `metadata` JSON column. |
| `web/app.js` + `web/index.html`  | Settings toggle, status pill, diff affordance. |
| `docs/PRIVACY.md`                | New section. Make the threat-model boundary explicit. |
| `tests/test_pseudo.py` (new)     | Unit tests for `forward/reverse` round-trip on a fixed mapping; integration test stub that monkeypatches the model call so the suite stays offline. |
| `tests/eval/test_pseudo_quality.py` (new, `-m eval`) | DeepEval pass that runs the held-out prompt set with and without pseudo and reports the delta. |

---

## 7. Roadmap — concrete milestones

The order is from "smallest verifiable slice" to "user-visible polish."
Each step ends with something we can demo and a measurable acceptance
criterion. Don't merge a step until the previous step's criterion holds.

### Step 1 — `core/pseudo.py` skeleton + offline tests *(half a day)*

- Write `forward()` and `reverse()` with a hand-rolled prompt against
  the local Ollama provider client (`core.providers.ollama`).
- Add `tests/test_pseudo.py` with deterministic round-trip tests that
  monkeypatch the model call: `forward(prompt)` returns a known
  mapping, `reverse(answer, mapping)` reproduces the original tokens.
- **Acceptance:** `pytest -m 'not eval'` stays green; new test file
  covers literal-pass reverse for ≥90% of cases.

### Step 2 — wire into `core.pipeline.run_roitelet_chat` *(half a day)*

- Gate behind `RouterPreferences.pseudonymize`.
- Add the `/pseudo` and `/nopseudo` slash commands; persist the
  per-turn override the same way `/local` does.
- Attach the audit to `ConversationMessage.metadata` and to the
  `ChatResponse`.
- **Acceptance:** an integration test that asserts (a) the rewritten
  prompt is what the provider client sees, (b) the original prompt is
  what gets persisted, (c) the response content has the originals
  restored.

### Step 3 — Settings UI + slash + audit affordance *(half a day)*

- Add `enable_pseudonymization` to `AppSettingsPayload` and to the
  `SETTINGS_FIELDS` array in `web/app.js`.
- Render a collapsible diff under the user bubble when
  `metadata.pseudonymization` is present.
- Status pill in the header gains a "pseudo" tag when the toggle is
  on.
- **Acceptance:** manual GUI test — toggle on, send "Marie Dupont à
  Lyon", see substitutes in the diff, see *Marie Dupont* and *Lyon*
  back in the assistant bubble.

### Step 4 — LLM repair pass for inflected reverses *(half a day)*

- Add the second-stage repair prompt and call it only when the literal
  pass left orphan substitutes detectable in the fused text.
- Cap the repair pass at one local model call per turn — never recurse.
- **Acceptance:** add a fixture where the synthesis uses
  *"Mme Lefèvre"* and the repair pass restores *"Mme Dupont"*.

### Step 5 — `docs/PRIVACY.md` rewrite *(short)*

- Update §1's three-mode table with a fourth row: **Pseudonymized
  local-first**.
- Add the threat-model caveat: pseudonymization reduces casual
  provider-side PII exposure; it does not anonymize the user.
- Add the per-prompt-class quality warning (named-entity QA breaks;
  coding/math is a no-op).
- **Acceptance:** one paragraph of plain English the user can read in
  20s and decide whether to flip the toggle.

### Step 6 — eval ablation *(a day, optional first cut)*

- Add `tests/eval/test_pseudo_quality.py`, marked `-m eval`. Re-uses
  the existing held-out prompt set; reports DeepEval correctness with
  and without pseudo.
- Land a §4.5 entry in `docs/EVALUATION.md` with the delta.
- **Acceptance:** numbers in the table; no claims unless the run is
  reproducible from the repo.

### Step 7 — README/LISEZMOI blurb *(short)*

- One paragraph in the Features section pointing at PRIVACY.md.
- Screenshot of the audit diff (the same way commit 4deca3d added the
  interface screenshot).

### Out of scope for the initial cut

- **Custom name banks.** The LLM does the substitution. Adding curated
  banks is a quality bump we revisit after Step 6's eval shows where
  the model is weakest.
- **Streaming mode.** The reverse pass needs the full text. Streaming
  responses are deferred until the rest of the feature is stable.
- **Image and PDF attachments.** Pseudonymizing OCR text adds an
  attachment-modality dimension we want to handle separately so the
  first cut doesn't bog down in multimodal edge cases.

---

## 8. How we know this was worth shipping

Three weak-ish but real signals — none alone is sufficient:

1. **The eval delta in §6 is small for non-named-entity prompts.** If
   pseudonymization tanks coding/math/RAG answers, the prompt design
   is wrong and we re-tune before user-facing release.
2. **The audit diff looks plausible to a human review.** Same-origin,
   same-gender, same-register substitutes. If a French name becomes
   *"John Smith"*, the substitution prompt is too loose.
3. **Local-only and pseudonymized-remote both feel like real options.**
   The privacy story should now read as a 4-row table (today: 3), and
   we should expect roughly half of cautious users to pick row 4 over
   row 1, trading some quality for the remote candidates.

If, after a week of use, no one flips the toggle on, the feature is a
false positive on the roadmap and we revert. We'll instrument the
toggle's persisted state and add a one-line `git log`-style summary to
`/api/personal` so we don't have to guess.
