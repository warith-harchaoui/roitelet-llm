"""Pseudonymization — swap PII before remote calls, restore it on the way back.

The feature exposed here lets a Roitelet turn do two things invisibly to
the user:

1. **Forward pass** — before the prompt reaches the router, a local
   Ollama model rewrites every piece of personally-identifying
   information into a **plausible same-origin substitute**. The
   rewritten prompt is what the remote candidates see; the original
   prompt is what gets persisted in the conversation log.

2. **Reverse pass** — after the synthesis judge fuses the candidate
   answers, every substitute is mapped back to the user's original
   string. Literal substring replace covers most of it; an LLM repair
   pass handles inflected forms when the literal pass leaves orphans.

Both passes use the same local model and produce an audit trail
(``PseudonymizationAudit``) that the GUI / CLI render verbatim so the
user can see what *actually* left the box on a given turn.

Threat model in one sentence
----------------------------
This feature reduces casual provider-side PII exposure (provider audit
logs, retained training corpora, careless logging). It does **not**
anonymize the user against a hostile counterparty: a determined
adversary with control of the provider can infer identity from
context. The docs say so plainly; the audit affordance is what makes
the trade legible to the user.

PII taxonomy
------------
The forward prompt enumerates a precise list of categories the model
must address (see :data:`_FORWARD_SYSTEM_PROMPT`). The categories
match :data:`core.schemas.PIIKind`:

* identity — ``person_name``, ``username``, ``date_of_birth``;
* geography — ``place_name``, ``street_address``, ``coordinates``;
* organisation — ``organization``, ``job_title``;
* contact — ``email``, ``phone``, ``url_handle``;
* network — ``ip_address``;
* identifiers — ``national_id``, ``financial_id``, ``medical_id``,
  ``employee_id``, ``account_id``, ``vehicle_id``;
* fallback — ``other_identifier`` when the model judges a string
  PII-class but none of the above slots fit.

What the model is told **not** to substitute:

* programming-language names, library names, function names, file
  paths, file extensions;
* public-documentation URLs (``docs.python.org``,
  ``en.wikipedia.org``) and their anchors;
* mathematical / scientific terms and well-known constants;
* generic vocabulary (cities used as common nouns like "marathon",
  units of measurement, etc.);
* anything inside a fenced code block — code is a regime where
  substituting an identifier breaks the answer entirely.

Fail-closed contract
--------------------
If the local model is unreachable, returns malformed JSON, or
produces a mapping where *any* original isn't a literal substring of
the input prompt or *any* substitute isn't present in the rewritten
prompt, :func:`pseudonymize_prompt` raises
:class:`PseudonymizationError`. The pipeline layer surfaces that
straight to the user instead of silently sending the unmodified
prompt — a safety toggle that secretly fails open is worse than no
toggle.

Examples
--------
>>> from core.pseudo import literal_restore
>>> mappings = [
...     {'original': 'Marie Dupont', 'substitute': 'Camille Lefèvre', 'kind': 'person_name'},
...     {'original': 'Lyon', 'substitute': 'Toulouse', 'kind': 'place_name'},
... ]
>>> # Build the audit and check the literal restore on a fused answer.
>>> from core.schemas import PIIMapping
>>> literal_restore(
...     "Camille Lefèvre est à Toulouse depuis lundi.",
...     [PIIMapping(**m) for m in mappings],
... )
'Marie Dupont est à Lyon depuis lundi.'
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable

from . import storage as _storage_mod
from .providers.factory import get_provider_client
from .schemas import ChatMessage, PIIMapping, PseudonymizationAudit

logger = logging.getLogger(__name__)


class PseudonymizationError(RuntimeError):
    """Raised when the forward or reverse pass cannot be completed safely.

    The pipeline layer catches this and surfaces it as a clean error to
    the user (HTTP 502 on the API; non-zero exit on the CLI). Critically,
    the pipeline does **not** silently fall through to an unredacted
    remote call when this fires — that would defeat the purpose of the
    toggle.
    """


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# The forward prompt is intentionally long. It enumerates the PII
# taxonomy, gives examples of same-origin substitution per category,
# states the "do not substitute" list, and demands a strict JSON schema
# for the output. Tuning is empirical; if a category misses too often,
# add a one-line clarification — don't rewrite the whole prompt.
_FORWARD_SYSTEM_PROMPT = """\
You are a privacy-preserving rewriter. The user will give you a prompt
that may leave the local machine. Your job is to swap every piece of
**personally identifying information (PII)** for a **plausible
same-origin substitute** so the rewritten prompt can be sent to a
third-party language model without leaking the user's identity, while
still letting the model produce a useful answer.

# PII categories you MUST substitute

1. **person_name** — given names, surnames, full names, nicknames.
   Substitute by another name of the **same locale and same gender**
   (when inferable from context). Examples:
   - "Marie Dupont" (FR) → "Camille Lefèvre" (FR).
   - "John Smith" (EN) → "Liam Carter" (EN).
   - "山田太郎" (JP) → "佐藤健" (JP).

2. **username** — standalone handles or logins not in a URL.
   Substitute by a structurally-similar handle.

3. **date_of_birth** — DOBs or other identity-bound dates.
   Shift by a plausible amount (±2 years; preserve format).

4. **place_name** — cities, regions, countries, neighbourhoods,
   landmarks. Substitute by a place of **the same country and a
   comparable size**. "Lyon" → "Toulouse", not "Tokyo".

5. **street_address** — full postal addresses.
   Substitute the street, number, and postcode; keep the country.

6. **coordinates** — GPS lat/lng pairs. Shift by ~1° while staying
   in the same country.

7. **organization** — companies, schools, government bodies, NGOs,
   hospitals. Substitute by another organisation of the **same
   sector and country**.

8. **job_title** — role descriptions tied to a named person.
   Substitute when the title narrows the person ("CEO of Acme" is
   identifying; "an engineer" is not).

9. **email** — full email addresses. Substitute the user part with
   a plausible name; **keep the domain's class** (gmail.com →
   gmail.com, orange.fr → orange.fr) so the answer can still reason
   about the provider.

10. **phone** — phone numbers. Substitute digits; **keep the country
    calling code** and overall format so locale stays correct.

11. **url_handle** — social-media URLs whose path contains a personal
    handle (twitter.com/<user>, github.com/<user>). Substitute the
    handle; keep the host.

12. **ip_address** — IPv4/IPv6 literals. Substitute by another
    address in a different /24 (IPv4) or /48 (IPv6).

13. **national_id** — SSN, INSEE, NIR, NHS numbers, passport numbers,
    driver's licenses. Substitute digit-for-digit while preserving
    format and any checksum-shape.

14. **financial_id** — credit card, IBAN, BIC, bank account.
    Substitute digit-for-digit; preserve format.

15. **medical_id** — MRN, patient numbers. Substitute structurally.

16. **employee_id** — workplace identifiers. Substitute structurally.

17. **account_id** — customer / ticket numbers that personally
    identify a user.

18. **vehicle_id** — license plates, VINs. Substitute structurally;
    keep the country code prefix.

19. **other_identifier** — any free-form string that is clearly PII
    but doesn't fit the slots above.

# What you MUST NOT substitute

- Programming-language names, library names, function names, file
  paths, file extensions.
- Public documentation URLs (e.g. ``docs.python.org``,
  ``en.wikipedia.org``).
- Mathematical / scientific terms, units of measurement, well-known
  constants.
- Generic vocabulary that *happens to share a spelling* with a
  proper noun (the word "marathon" used as a common noun is not a
  city name).
- Anything inside a Markdown fenced code block (```...```) or inline
  code (`...`). Code is technical content; substituting an
  identifier inside it breaks the answer.
- Names of historical / public figures **only when the prompt is
  asking about them as historical figures** (e.g. "What did
  Napoleon do in 1812?"). When in doubt, substitute — the user can
  disable pseudonymization for that prompt class with
  ``/nopseudo``.

# Output format — STRICT JSON

Return a single JSON object with exactly two keys:

```
{
  "pseudonymized_prompt": "<rewritten prompt with substitutes in place>",
  "mappings": [
    {"original": "<string from the input>", "substitute": "<substitute>", "kind": "<one of the categories above>"},
    ...
  ]
}
```

Hard constraints, in order of importance:

- Every ``original`` MUST appear verbatim in the input prompt.
- Every ``substitute`` MUST appear verbatim in the
  ``pseudonymized_prompt``.
- The ``pseudonymized_prompt`` MUST NOT contain any ``original``.
- ``kind`` MUST be one of the category slugs listed above.
- If you cannot find any PII, return ``"mappings": []`` and a
  ``pseudonymized_prompt`` that equals the input verbatim.
- Do NOT output anything outside the JSON object — no preamble,
  no markdown fence, no trailing commentary.
"""


_REPAIR_SYSTEM_PROMPT = """\
You are a privacy-preserving text editor. You will receive:

1. A text where placeholder names / places / identifiers were used
   to keep PII off a third-party model.
2. A mapping table from those placeholders to the user's original
   strings.

Your job is to walk through the text and **restore the originals**
according to the table. Notes:

- The text may use inflected forms of a placeholder (possessives,
  abbreviations, honorifics). Restore those too, in a form that
  matches the original's inflection — "Mme Lefèvre" → "Mme Dupont"
  when the table says ``Lefèvre → Dupont``.
- Do not invent new mappings. Do not change anything that is not
  listed in the table or a clear inflection of a listed entry.
- Preserve the original markdown / code formatting.

Return the restored text only — no preamble, no JSON wrapping.
"""


# Surface mask used in the pseudonymized prompt to make the swaps
# legible to humans reading the audit. Not used as the substitute
# itself — the model picks plausible names — but referenced in the
# diff renderer so we have a stable visual token.
PSEUDO_AUDIT_ARROW = '→'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_model_id() -> str:
    """Pick the local Ollama model used for the forward + reverse passes.

    Order of precedence:

    1. ``AppSettingsPayload.pseudo_model_id`` if non-empty (the user
       configured a redactor-specific model).
    2. ``AppSettingsPayload.local_synthesis_model`` — re-use the judge
       so the local footprint is one model unless the user opts in to
       a cheaper redactor.
    """
    runtime = _storage_mod.get_storage().load_app_settings()
    chosen = (runtime.pseudo_model_id or runtime.local_synthesis_model or '').strip()
    if not chosen:
        raise PseudonymizationError(
            'No local model is configured for pseudonymization. Set '
            '`pseudo_model_id` or `local_synthesis_model` in the settings.'
        )
    if '/' in chosen:
        return chosen
    # Default the provider prefix to ollama — the only local provider
    # that has a chat endpoint in the registry today.
    return f'ollama/{chosen}'


def _validate_forward(
    original_prompt: str,
    rewritten: str,
    mappings: list[PIIMapping],
) -> None:
    """Enforce the fail-closed contract on the model's forward output.

    Raises
    ------
    PseudonymizationError
        On any of: a missing ``original`` substring in the input
        prompt, a missing ``substitute`` substring in the rewritten
        prompt, a leaked ``original`` still present in the rewritten
        prompt, or a no-op mapping where ``original == substitute``.
    """
    seen_pairs: set[tuple[str, str]] = set()
    for mapping in mappings:
        if mapping.original == mapping.substitute:
            raise PseudonymizationError(
                f'Pseudonymizer returned a no-op mapping for {mapping.original!r}.'
            )
        if mapping.original not in original_prompt:
            raise PseudonymizationError(
                f'Pseudonymizer claimed {mapping.original!r} was in the prompt '
                f'but it is not. Refusing to send the rewritten prompt.'
            )
        if mapping.substitute not in rewritten:
            raise PseudonymizationError(
                f'Pseudonymizer did not actually use {mapping.substitute!r} as '
                f'the replacement for {mapping.original!r}. Refusing to send.'
            )
        if mapping.original in rewritten:
            raise PseudonymizationError(
                f'Pseudonymizer left the original {mapping.original!r} in the '
                f'rewritten prompt. Refusing to send.'
            )
        pair = (mapping.original, mapping.substitute)
        if pair in seen_pairs:
            # Duplicates are fine semantically (the mapping is the same)
            # but they bloat the audit. Drop silently rather than fail.
            continue
        seen_pairs.add(pair)


def _parse_json_object(raw: str) -> dict:
    r"""Best-effort JSON object extraction from a model's reply.

    Some local models wrap structured output in ``\`\`\`json ... \`\`\```
    fences despite the prompt telling them not to. Strip those before
    parsing.
    """
    stripped = raw.strip()
    if stripped.startswith('```'):
        # Drop the opening fence plus optional language tag.
        first_newline = stripped.find('\n')
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        if stripped.endswith('```'):
            stripped = stripped[: -3]
        stripped = stripped.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Last-ditch: locate the first ``{`` and the last ``}`` and
        # try again. Tolerates trailing commentary the model leaked.
        start = stripped.find('{')
        end = stripped.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(stripped[start: end + 1])
            except json.JSONDecodeError:
                raise PseudonymizationError(
                    f'Pseudonymizer did not return valid JSON: {exc}'
                ) from exc
        else:
            raise PseudonymizationError(
                f'Pseudonymizer did not return valid JSON: {exc}'
            ) from exc
    if not isinstance(parsed, dict):
        raise PseudonymizationError(
            f'Pseudonymizer returned {type(parsed).__name__}, not a JSON object.'
        )
    return parsed


async def pseudonymize_prompt(
    prompt: str,
    *,
    model_id: str | None = None,
) -> PseudonymizationAudit:
    """Run the forward (pseudonymize) pass against a local model.

    Parameters
    ----------
    prompt : str
        The user's original prompt — the text that, without this
        feature, would be sent verbatim to remote candidates.
    model_id : str, optional
        Override for the local model id (e.g. ``'ollama/qwen3:8b'``).
        Defaults to :func:`_resolve_model_id` which reads the
        persisted ``pseudo_model_id`` setting.

    Returns
    -------
    PseudonymizationAudit
        Carries the rewritten prompt, the mapping table, and timing.

    Raises
    ------
    PseudonymizationError
        On any validation failure or model error. Callers must NOT
        fall back to the original prompt — the safety toggle requires
        fail-closed behaviour.
    """
    resolved = model_id or _resolve_model_id()
    provider_key = resolved.split('/', 1)[0]
    client = get_provider_client(provider_key)

    messages = [
        ChatMessage(role='system', content=_FORWARD_SYSTEM_PROMPT),
        ChatMessage(role='user', content=prompt),
    ]
    started = time.perf_counter()
    try:
        response = await client.generate(model_id=resolved, messages=messages)
    except Exception as exc:
        raise PseudonymizationError(
            f'Pseudonymizer call failed: {exc}'
        ) from exc
    latency = time.perf_counter() - started
    if response.error:
        raise PseudonymizationError(
            f'Pseudonymizer model returned an error: {response.error}'
        )
    if not response.content.strip():
        raise PseudonymizationError(
            'Pseudonymizer returned an empty response.'
        )

    parsed = _parse_json_object(response.content)
    rewritten = parsed.get('pseudonymized_prompt')
    raw_mappings = parsed.get('mappings', [])
    if not isinstance(rewritten, str) or not isinstance(raw_mappings, list):
        raise PseudonymizationError(
            'Pseudonymizer JSON did not contain `pseudonymized_prompt` '
            'and `mappings` of the expected types.'
        )

    mappings: list[PIIMapping] = []
    for raw in raw_mappings:
        if not isinstance(raw, dict):
            raise PseudonymizationError(
                f'Pseudonymizer emitted a non-object mapping entry: {raw!r}'
            )
        try:
            mappings.append(PIIMapping.model_validate(raw))
        except Exception as exc:
            raise PseudonymizationError(
                f'Pseudonymizer emitted an invalid mapping entry {raw!r}: {exc}'
            ) from exc

    _validate_forward(prompt, rewritten, mappings)

    return PseudonymizationAudit(
        mappings=mappings,
        pseudonymized_prompt=rewritten,
        model_id=resolved,
        forward_latency_s=latency,
    )


def literal_restore(text: str, mappings: Iterable[PIIMapping]) -> str:
    """Reverse every substitute back to its original via plain substring replace.

    Order matters: substitutes that are substrings of other substitutes
    must be replaced last so the longer match wins. We sort by
    descending length to make that property explicit rather than
    relying on Python dict ordering.
    """
    ordered = sorted(mappings, key=lambda m: len(m.substitute), reverse=True)
    out = text
    for mapping in ordered:
        if mapping.substitute and mapping.substitute in out:
            out = out.replace(mapping.substitute, mapping.original)
    return out


def _has_orphan_substitutes(text: str, mappings: Iterable[PIIMapping]) -> bool:
    """Detect literal-pass misses where a substitute still appears in the text.

    The forward pass guarantees substitutes are present in the
    pseudonymized prompt, but the synthesis judge sees the
    pseudonymized prompt and is free to inflect substitutes
    ("Camille Lefèvre" → "Mme Lefèvre" or "Camille"). After the
    literal pass, any *whole* substitute substring left in the text
    is a sign the judge is paraphrasing — repair pass material.
    """
    return any(m.substitute in text for m in mappings)


async def restore_text(
    text: str,
    mappings: list[PIIMapping],
    *,
    model_id: str | None = None,
    allow_llm_repair: bool = True,
) -> tuple[str, bool]:
    """Reverse pseudonymization on a fused answer.

    Two-stage:

    1. **Literal pass** — exact substring substitute → original.
       Deterministic, cheap, covers most real cases.
    2. **LLM repair pass** *(opt-in)* — when the literal pass leaves
       a substitute substring untouched (rare; mostly when the
       synthesis judge inflected a name), feed the literally-reversed
       text + the mapping table back through the local model with a
       short repair prompt. Single call, no recursion.

    Parameters
    ----------
    text : str
        The fused synthesis content (still in pseudonymized form).
    mappings : list of PIIMapping
        Mappings the forward pass produced. Empty list → return the
        text unchanged.
    model_id : str, optional
        Override for the repair-pass model.
    allow_llm_repair : bool, default=True
        Set to ``False`` to suppress the repair pass entirely
        (deterministic-only). Used by the CLI's ``--fast-restore``
        mode and by tests that don't want network calls.

    Returns
    -------
    (restored_text, repair_used) : tuple
        ``repair_used`` reflects whether the repair pass actually
        fired this turn — used by telemetry and the GUI audit
        affordance.
    """
    if not mappings:
        return text, False

    literal = literal_restore(text, mappings)

    if not allow_llm_repair or not _has_orphan_substitutes(literal, mappings):
        return literal, False

    resolved = model_id or _resolve_model_id()
    provider_key = resolved.split('/', 1)[0]
    client = get_provider_client(provider_key)

    table = '\n'.join(
        f'- {m.substitute} → {m.original} (kind: {m.kind})'
        for m in mappings
    )
    user = (
        'Mapping table (substitute → original):\n'
        f'{table}\n\n'
        'Text to restore:\n'
        f'{literal}'
    )
    messages = [
        ChatMessage(role='system', content=_REPAIR_SYSTEM_PROMPT),
        ChatMessage(role='user', content=user),
    ]
    try:
        response = await client.generate(model_id=resolved, messages=messages)
    except Exception as exc:
        logger.warning('Pseudo repair pass failed (%s); returning literal-only.', exc)
        return literal, False
    if response.error or not response.content.strip():
        logger.warning(
            'Pseudo repair pass returned empty/error (%s); returning literal-only.',
            response.error,
        )
        return literal, False
    return response.content.strip(), True


__all__ = [
    'PseudonymizationError',
    'pseudonymize_prompt',
    'restore_text',
    'literal_restore',
]
