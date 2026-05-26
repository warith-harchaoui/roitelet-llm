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
        The computed score evaluating its suitability.
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
    estimated_cost_usd: float = 0.0
    estimated_latency_s: float = 0.0
    capability_scores: list[ModelCapabilityScore] = Field(default_factory=list)


class RouterPreferences(BaseModel):
    """User-configurable preferences that influence the router.

    Parameters
    ----------
    raw_power:
        Weight given to pure quality and benchmark strength.
    frugality:
        Weight given to low cost and low energy usage.
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
    frugality: float = 0.3
    independence: bool = False
    allow_vlms: bool = False
    max_cost_usd: float | None = None


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
    frugality_weight: float = 0.3
    independence_local_only: bool = False
    selected_ollama_models: list[str] = Field(default_factory=list)
    paid_openrouter_models: list[str] = Field(default_factory=list)
    # Any paid LLM with an OpenAI-compatible chat-completions endpoint can
    # be added without a bootstrap edit. Pair these with
    # ``openai_compatible_base_url`` + ``openai_compatible_api_key``; each
    # entry becomes a registry model under the ``openai-compatible``
    # provider. Useful for Mistral, Together, Groq, llama.cpp's
    # ``llama-server``, and any future provider that ships an
    # ``/v1/chat/completions`` interface.
    paid_openai_compatible_models: list[str] = Field(default_factory=list)

    def masked(self) -> AppSettingsPayload:
        """Return a copy with non-empty secret fields replaced by ``SECRET_MASK``."""
        replacements = {
            field: SECRET_MASK
            for field in SECRET_FIELDS
            if getattr(self, field)
        }
        return self.model_copy(update=replacements)

    def merge_unmasked(self, incoming: AppSettingsPayload) -> AppSettingsPayload:
        """Merge an incoming payload over self, preserving masked secrets.

        Secrets whose incoming value still equals ``SECRET_MASK`` keep
        the stored value; every other field is overwritten.

        This lets the web UI round-trip settings without ever seeing the real
        API keys: it reads masked values, sends them back unchanged, and the
        server keeps the stored credentials.
        """
        replacements = {}
        for field in SECRET_FIELDS:
            if getattr(incoming, field) == SECRET_MASK:
                replacements[field] = getattr(self, field)
        return incoming.model_copy(update=replacements)


class ChatRequest(BaseModel):
    """Roitelet-native chat request payload."""

    prompt: str
    conversation_id: str | None = None
    preferences: RouterPreferences = Field(default_factory=RouterPreferences)
    top_k: int = 3
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


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str


class OpenAIChatCompletionRequest(BaseModel):
    """Subset of the OpenAI Chat Completions API accepted by Roitelet."""

    model: str = 'roitelet-llm'
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
