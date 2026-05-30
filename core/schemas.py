"""Pydantic schemas shared by the API, web UI, and orchestration code.

Examples
--------
>>> from core.schemas import ChatMessage
>>> ChatMessage(role="user", content="Bonjour")
ChatMessage(role='user', content='Bonjour')
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal['system', 'user', 'assistant']


class ChatMessage(BaseModel):
    """A single chat message.

    Parameters
    ----------
    role:
        Message role in chat history.
    content:
        Free-form text content.
    """

    role: Role
    content: str


class ModelCapabilityScore(BaseModel):
    """Capability-specific score for a model on a prompt."""

    capability: str
    score: float
    rationale: str


class ModelCandidate(BaseModel):
    """A candidate model and the router metadata attached to it.

    Attributes
    ----------
    model_id : str
        The full string identifier for the model.
    provider : str
        Local provider (e.g. 'ollama') or remote (e.g. 'openai-compatible').
    selected : bool, default=False
        Whether the routing algorithm picked this candidate for query dispatch.
    score : float
        The computed score evaluating its suitability — arbitrary
        units (a weighted blend of quality + ecofrugality bonuses).
    quality_probability : float
        Normalised quality estimator in [0.0, 1.0]. Computed by the
        router as ``(quality_score - min) / (max - min)`` across the
        eligible candidate pool on this turn — so the top-quality
        candidate is 1.0, the worst is 0.0. Comparable to
        RouteLLM's calibrated ``P(strong wins)`` in *shape* (one
        scalar, monotonic, threshold-able) though derived from
        rolling Elo + capability priors, not preference labels.
        Drives the :attr:`RouterPreferences.quality_threshold`
        filter that traces the Pareto frontier.
    estimated_cost_usd : float
        Pessimistic cost computation assuming context bounds.
    estimated_latency_s : float
        Latency estimated through historical benchmarks.
    capability_scores : List[ModelCapabilityScore]
        Specific scores broken down by distinct capabilities.
    """

    model_id: str
    provider: str
    selected: bool = False
    score: float
    quality_probability: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_latency_s: float = 0.0
    capability_scores: list[ModelCapabilityScore] = Field(default_factory=list)


class RouterPreferences(BaseModel):
    """User-configurable preferences that influence the router.

    Parameters
    ----------
    raw_power:
        Weight given to pure quality and benchmark strength.
    ecofrugality:
        Weight given to the combination of low cost (USD) and low energy
        consumption (kWh). Replaces the legacy ``frugality`` knob, which
        only spoke of "cost + energy" abstractly; the new name keeps the
        meaning honest — money **and** energy, jointly.
    independence:
        If true, remote models are filtered out and only local models are used.
    allow_vlms:
        If true, visual-language models can be considered.
    max_cost_usd:
        Per-turn cost ceiling (sum of input + output 1k-token prices, USD).
        Candidates whose estimated cost exceeds the budget are filtered
        out before top-K selection. ``None`` disables the budget. Local
        models are always under the budget (their pricing is zero), so
        setting a tight budget is functionally similar to flipping
        ``independence`` on — except it lets one cheap paid model
        through if its price still fits.
    """

    raw_power: float = 0.7
    ecofrugality: float = 0.3
    independence: bool = False
    allow_vlms: bool = False
    max_cost_usd: float | None = None
    # Calibrated quality floor in [0.0, 1.0]. Candidates whose
    # normalised ``quality_probability`` (computed across the eligible
    # pool on each turn) falls below this value are filtered *before*
    # top-K selection. This is the single-knob operating point on the
    # Roitelet cost/quality Pareto frontier: 0.0 = no filter, 1.0 =
    # only the very best candidate. Mirrors the role of RouteLLM's
    # threshold on its calibrated ``P(strong wins)``.
    quality_threshold: float = 0.0
    # When True, a local LLM rewrites the prompt before fan-out, swapping
    # personally-identifying information (names, addresses, contact
    # details, financial / national / medical IDs, IPs, …) for
    # plausible same-locale substitutes; the inverse swap is applied
    # to the fused answer before it returns to the user. See
    # :mod:`core.pseudo` for the taxonomy, the structured-output prompt,
    # and the fail-closed validation. Opt-in because (a) it adds one
    # local model hop, and (b) it can degrade answer quality on
    # named-entity-bound prompts ("What did Napoleon do in 1812?").
    pseudonymize: bool = False


class RouterDecision(BaseModel):
    """Structured result of prompt routing.
    
    Attributes
    ----------
    prompt : str
        The raw text provided by the human context.
    categories : Dict[str, float]
        Detected capabilities mapped to their probability weights.
    candidates : List[ModelCandidate]
        Evaluated model candidates and respective meta attributes.
    selected_model_ids : List[str]
        IDs of models ultimately picked for query.
    reasoning : List[str]
        Verbalized traces of the routing decisions for audit logs.
    """

    prompt: str
    categories: dict[str, float]
    candidates: list[ModelCandidate]
    selected_model_ids: list[str]
    reasoning: list[str]


class ModelResponse(BaseModel):
    """An answer emitted by an upstream model.
    
    Attributes
    ----------
    model_id : str
        String identifier matched with the candidate.
    provider : str
        The platform queried (OpenRouter, local Ollama, etc.).
    content : str
        The raw LLM response.
    latency_s : float
        Round-trip inference time locally measured.
    usage : Dict[str, float]
        Inbound/outbound token accounting given by the provider.
    energy_kwh : float
        Watt-hours consumed during the specific query transaction.
    carbon_g : float
        Grams of CO2 dynamically approximated based on grid intensity.
    cost_usd : float
        Real invoiced cost based strictly on token counts.
    error : Optional[str]
        Description if the request abruptly halted.
    """

    model_id: str
    provider: str
    content: str
    latency_s: float
    usage: dict[str, float] = Field(default_factory=dict)
    energy_kwh: float = 0.0
    carbon_g: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class SynthesisResult(BaseModel):
    """Final fused answer produced by the local synthesis model.

    Attributes
    ----------
    model_id : str
        Judge model identifier (e.g. ``ollama/qwen3:8b``).
    provider : str
        Judge provider key.
    content : str
        The fused, user-facing answer.
    judge_summary : str
        Raw judge transcript including the winners block (or its
        absence). Useful for debugging fusion failures.
    winning_model_ids : list of str
        Candidate model ids the judge credited in the fused answer.
        Empty when the judge failed to emit a valid winners block — in
        that case the Elo loop receives no reward signal.
    latency_s : float
        Wall-clock seconds spent inside the judge call (network + judge
        generation). Reported alongside candidate latencies so the
        total user-perceived latency
        ``max(candidate_latencies) + latency_s`` is reconstructable.
    """

    model_id: str
    provider: str
    content: str
    judge_summary: str
    winning_model_ids: list[str]
    latency_s: float = 0.0


class TelemetryRecord(BaseModel):
    """Per-turn telemetry written to disk for monitoring and shadow evaluation."""

    record_id: str
    created_at: datetime
    conversation_id: str
    prompt: str
    router_decision: RouterDecision
    model_responses: list[ModelResponse]
    synthesis: SynthesisResult
    reward_model_ids: list[str]
    shadow_reference_model_ids: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


PIIKind = Literal[
    # Personal identity
    'person_name',          # any name (first, last, full, nickname); honorific kept as-is
    'username',             # standalone handle / login when not in a URL
    'date_of_birth',        # DOB or other identity-bound date
    # Geography
    'place_name',           # city, region, country, neighborhood, landmark
    'street_address',       # full postal address (street + number, optional postcode)
    'coordinates',          # GPS latitude/longitude pair
    # Organisation
    'organization',         # company, school, government body, NGO, hospital
    'job_title',            # role tied to a specific named person ("CEO of Acme")
    # Contact
    'email',                # full RFC-shaped email address
    'phone',                # phone number with optional country code
    'url_handle',           # social-media URL whose path contains a personal handle
    # Network
    'ip_address',           # IPv4 or IPv6 literal
    # Government / financial / medical / vehicular identifiers
    'national_id',          # SSN, INSEE, NIR, NHS, passport, driver's license
    'financial_id',         # credit card, IBAN, BIC, bank account number
    'medical_id',           # MRN, patient number
    'employee_id',          # workplace identifier
    'account_id',           # customer / ticket number that personally identifies a user
    'vehicle_id',           # license plate, VIN
    # Catch-all
    'other_identifier',     # any free-form identifier the model judges PII-class
]


class PIIMapping(BaseModel):
    """One ``original → substitute`` PII swap performed during pseudonymization.

    Attributes
    ----------
    original : str
        The PII string as it appeared in the user's prompt. The forward
        pass guarantees this substring exists in the input prompt.
    substitute : str
        The plausible same-origin replacement the local LLM produced.
        The forward pass guarantees this substring exists in the
        rewritten prompt sent to remote candidates.
    kind : PIIKind
        Coarse taxonomy slot — used by the UI's audit diff to colour
        and group entries, and by the eval suite to measure
        category-specific recall.
    """

    original: str
    substitute: str
    kind: PIIKind


class PseudonymizationAudit(BaseModel):
    """Audit record of one pseudonymization turn.

    Attached to ``ConversationMessage.metadata['pseudonymization']`` on
    the user turn so the GUI / CLI can render the diff, and to the
    assistant turn's metadata so reverse-pass diagnostics survive a
    page reload.

    Attributes
    ----------
    mappings : list of PIIMapping
        Every PII swap the forward pass produced. May be empty when the
        prompt contained no PII the model could identify — that's a
        legitimate outcome, not a failure.
    pseudonymized_prompt : str
        The exact text sent to the router and remote candidates. Stored
        so a privacy-conscious user can verify, after the fact,
        what *actually* left the box.
    model_id : str
        Which local model performed both the forward and reverse
        passes. Recorded so version-to-version quality drift is
        auditable.
    forward_latency_s : float
        Wall-clock cost of the forward (pseudonymize) pass.
    reverse_latency_s : float
        Wall-clock cost of the reverse (restore) pass.
    repair_used : bool
        ``True`` when the literal-pass reverse left orphan substitutes
        in the synthesis and the LLM repair pass was invoked. Useful
        for telemetry: a high repair rate signals the synthesis judge
        is paraphrasing substitutes more than expected.
    """

    mappings: list[PIIMapping] = Field(default_factory=list)
    pseudonymized_prompt: str
    model_id: str
    forward_latency_s: float = 0.0
    reverse_latency_s: float = 0.0
    repair_used: bool = False


class ConversationMessage(BaseModel):
    """A stored conversation message for the simple prompt interface."""

    role: Literal['user', 'assistant']
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Persistent conversation state."""

    conversation_id: str
    title: str
    created_at: datetime
    messages: list[ConversationMessage] = Field(default_factory=list)


# Placeholder shown to the web UI in place of stored API keys. When the UI
# POSTs settings back with this sentinel, the server preserves the on-disk
# key rather than overwriting it. The exact string must be stable across
# releases — changing it would silently blank existing keys.
SECRET_MASK = '••••••••'

# AppSettingsPayload fields that hold credentials and must be masked on read.
SECRET_FIELDS = (
    'openrouter_api_key',
    'openai_api_key',
    'anthropic_api_key',
    'gemini_api_key',
    'perplexity_api_key',
    'openai_compatible_api_key',
)


class CustomEngine(BaseModel):
    """One user-added OpenAI-compatible engine.

    Attributes
    ----------
    label : str
        Short identifier used as a namespace in model ids
        (``openai-compatible/<label>/<model>``). Must be a non-empty
        slug; the GUI enforces this client-side.
    base_url : str
        OpenAI-compatible endpoint root (no trailing
        ``/chat/completions``). Example: ``https://api.mistral.ai/v1``.
    api_key : str
        Bearer token sent to the endpoint. Empty strings disable the
        engine — the registry auto-prunes models from unauthorized
        engines.
    models : list of str
        Model names served by this endpoint. Each registers as
        ``openai-compatible/<label>/<name>``.
    """

    label: str
    base_url: str
    api_key: str = ''
    models: list[str] = Field(default_factory=list)


class AppSettingsPayload(BaseModel):
    """Settings payload edited from the web control room."""

    openrouter_api_key: str = ''
    openai_api_key: str = ''
    anthropic_api_key: str = ''
    gemini_api_key: str = ''
    perplexity_api_key: str = ''
    openai_compatible_api_key: str = ''
    openai_compatible_base_url: str = ''
    openai_compatible_model: str = ''
    ollama_base_url: str = 'http://localhost:11434'
    local_synthesis_model: str = 'qwen3:8b'
    local_vlm_model: str = 'qwen2.5vl:7b'
    enable_vlms: bool = False
    raw_power_weight: float = 0.7
    ecofrugality_weight: float = 0.3
    independence_local_only: bool = False
    # Default state of the per-turn ``preferences.pseudonymize`` flag.
    # The GUI's checkbox mirrors this value on open; the CLI's
    # ``--pseudonymize`` / ``--no-pseudonymize`` overrides win for one
    # turn but don't write back. See PSEUDO.md for the user-facing
    # docs and core/pseudo.py for the implementation.
    enable_pseudonymization: bool = False
    # Local Ollama model used for both the forward (pseudonymize) and
    # reverse (restore) passes. Empty string falls back to
    # ``local_synthesis_model`` — the same judge model — so the local
    # footprint stays at one model unless the user explicitly wants a
    # cheaper / smaller redactor.
    pseudo_model_id: str = ''
    selected_ollama_models: list[str] = Field(default_factory=list)
    paid_openrouter_models: list[str] = Field(default_factory=list)
    # Universal extension: any number of OpenAI-compatible engines,
    # each with its own label / URL / key / model list. Edited
    # dynamically from the Settings sheet via the "+ Add engine"
    # button. Each engine's models register as
    # ``openai-compatible/<label>/<model_name>`` in the routing pool.
    custom_engines: list[CustomEngine] = Field(default_factory=list)

    def masked(self) -> AppSettingsPayload:
        """Return a copy with non-empty secret fields replaced by ``SECRET_MASK``.

        Covers both the top-level ``*_api_key`` fields and the
        per-engine ``api_key`` inside every :class:`CustomEngine`.
        Empty keys stay empty (so the UI can tell which engines have
        no credentials configured yet).
        """
        replacements = {
            field: SECRET_MASK
            for field in SECRET_FIELDS
            if getattr(self, field)
        }
        masked_engines = [
            engine.model_copy(update={'api_key': SECRET_MASK} if engine.api_key else {})
            for engine in self.custom_engines
        ]
        replacements['custom_engines'] = masked_engines
        return self.model_copy(update=replacements)

    def merge_unmasked(self, incoming: AppSettingsPayload) -> AppSettingsPayload:
        """Merge an incoming payload over self, preserving masked secrets.

        Secrets whose incoming value still equals ``SECRET_MASK`` keep
        the stored value; every other field is overwritten. Also
        round-trips per-engine API keys: an incoming custom engine
        whose ``api_key`` is ``SECRET_MASK`` inherits the stored key
        from the engine with the same ``label`` on the server. New
        engines (label not on file) keep whatever they were sent with;
        deleted engines disappear because they're simply absent from
        ``incoming.custom_engines``.

        This lets the web UI round-trip settings without ever seeing the real
        API keys: it reads masked values, sends them back unchanged, and the
        server keeps the stored credentials.
        """
        replacements: dict = {}
        for field in SECRET_FIELDS:
            if getattr(incoming, field) == SECRET_MASK:
                replacements[field] = getattr(self, field)
        # Per-engine key round-trip, keyed on engine label.
        stored_by_label = {e.label: e for e in self.custom_engines}
        merged_engines: list[CustomEngine] = []
        for engine in incoming.custom_engines:
            if engine.api_key == SECRET_MASK and engine.label in stored_by_label:
                merged_engines.append(
                    engine.model_copy(update={'api_key': stored_by_label[engine.label].api_key})
                )
            else:
                merged_engines.append(engine)
        replacements['custom_engines'] = merged_engines
        return incoming.model_copy(update=replacements)


class ChatRequest(BaseModel):
    """Roitelet-native chat request payload."""

    prompt: str
    conversation_id: str | None = None
    preferences: RouterPreferences = Field(default_factory=RouterPreferences)
    # K=2 — see docs/EVALUATION.md §4.3 for the ablation that picked it.
    top_k: int = 2
    shadow_full_pool: bool = True
    # When true, ``/api/chat`` returns a ``text/event-stream`` NDJSON-style
    # progressive response so clients can render the synthesis as it lands
    # rather than blocking on the full fusion call.
    stream: bool = False


class ChatResponse(BaseModel):
    """Roitelet-native chat response payload.

    Attributes
    ----------
    conversation_id : str
        UUID of the conversation the turn belongs to.
    router : RouterDecision
        Full routing decision (candidates, scores, regime, reasoning).
    responses : list of ModelResponse
        One entry per candidate the router selected, including failed
        ones (the ``error`` field is populated on failure).
    synthesis : SynthesisResult
        Fused answer + judge metadata + judge latency.
    telemetry_id : str
        UUID of the persisted telemetry record on disk.
    total_latency_s : float
        Wall-clock seconds of the full pipeline turn: router decision +
        candidate fan-out (bounded by the slowest candidate) + judge
        synthesis + telemetry persist. This is the user-perceived
        latency and the canonical number for ablations.
    """

    conversation_id: str
    router: RouterDecision
    responses: list[ModelResponse]
    synthesis: SynthesisResult
    telemetry_id: str
    total_latency_s: float = 0.0
    # Populated only when ``preferences.pseudonymize`` was True on the
    # request. Carries the rewritten prompt that actually went to the
    # remote candidates and the substitution table, so any UI surface
    # can render a verifiable audit of what left the box.
    pseudonymization: PseudonymizationAudit | None = None


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI Chat Completions API accepted by Roitelet."""

    model: str = 'roitelet'
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageGenRequest(BaseModel):
    """Roitelet-native image-generation request.

    Mirrors the shape of :class:`ChatRequest` so the same router can be
    consulted (with ``image_gen`` as the dominant capability). Fan-out
    is K=1 by design — there's no useful fusion operation over images,
    so the strongest single candidate's output *is* the answer.
    """

    prompt: str
    conversation_id: str | None = None
    preferences: RouterPreferences = Field(default_factory=RouterPreferences)
    size: Literal['256x256', '512x512', '1024x1024', '1792x1024'] = '1024x1024'
    n: int = 1


class GeneratedImage(BaseModel):
    """One image produced by an image-generation provider."""

    path: str  # filesystem path under data/images/, served via /data/images/<uuid>.png
    model_id: str
    provider: str
    revised_prompt: str | None = None  # some providers (e.g. DALL-E) rewrite the user prompt
    error: str | None = None


class ImageGenResponse(BaseModel):
    """Roitelet-native image-generation response."""

    conversation_id: str
    model_id: str
    provider: str
    images: list[GeneratedImage]
    latency_s: float
    cost_usd: float = 0.0


class OpenAIImagesRequest(BaseModel):
    """OpenAI-compatible ``/v1/images/generations`` payload subset."""

    prompt: str
    model: str = 'roitelet-image'
    n: int = 1
    size: Literal['256x256', '512x512', '1024x1024', '1792x1024'] = '1024x1024'


class MCPRequest(BaseModel):
    """Minimal JSON-RPC payload used by the embedded MCP endpoint."""

    jsonrpc: str = '2.0'
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
