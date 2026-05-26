"""Storage backend Protocol.

A typing seam that documents the public surface every storage backend must
implement. Today there's exactly one implementation
(:class:`core.storage.StorageManager`, JSON-per-file on the local
filesystem). When usage outgrows that (telemetry > ~10 k rows, multi-user
deployments, etc.) a SQLite backend can be added without touching call
sites — the Protocol pins the contract.

This module deliberately contains only typing-level state: no instances,
no factories, no behaviour. ``StorageManager`` is registered as a
:class:`Storage` via ``__init_subclass__`` for free, since Protocol
classes accept structural matches without explicit ``isinstance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .schemas import AppSettingsPayload, Conversation, ConversationMessage, TelemetryRecord


@runtime_checkable
class Storage(Protocol):
    """Public surface every Roitelet storage backend must implement.

    Methods are grouped by domain:

    * Conversations — UUID-keyed JSON blobs, one per chat thread.
    * Telemetry — append-only audit trail, one record per turn.
    * App settings — single JSON document, mutated by the web UI.
    * Provider cache — opaque request/response replay layer used by
      paid providers to avoid re-billing identical payloads.

    Implementations must be safe to share across concurrent async tasks
    (the pipeline fans out provider calls with :func:`asyncio.gather`).
    The current JSON backend achieves that through atomic
    write-then-rename; a SQLite backend would lean on its transaction
    isolation level.
    """

    # Conversations ------------------------------------------------------

    def create_conversation(self, title: str = ...) -> Conversation:
        """Create a new conversation and return it persisted."""
        ...

    def conversation_path(self, conversation_id: str) -> Path:
        """Return the on-disk path for a conversation.

        Must reject non-UUID identifiers so untrusted HTTP path params
        cannot escape the data directory.
        """
        ...

    def save_conversation(self, conversation: Conversation) -> None:
        """Persist the full conversation document atomically."""
        ...

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return the conversation by id, or ``None`` if it does not exist."""
        ...

    def list_conversations(self) -> list[Conversation]:
        """Return every stored conversation, newest first."""
        ...

    def append_message(
        self, conversation_id: str, message: ConversationMessage
    ) -> Conversation:
        """Append ``message`` to the conversation and return the updated record."""
        ...

    # Telemetry ----------------------------------------------------------

    def save_telemetry(self, record: TelemetryRecord) -> Path:
        """Persist one per-turn telemetry record and return its path."""
        ...

    def list_telemetry(self) -> list[TelemetryRecord]:
        """Return every stored telemetry record, newest first."""
        ...

    # App settings -------------------------------------------------------

    def load_app_settings(self) -> AppSettingsPayload:
        """Return the persisted control-room settings (with secrets present)."""
        ...

    def save_app_settings(self, payload: AppSettingsPayload) -> None:
        """Persist control-room settings, replacing the previous payload."""
        ...

    # Provider cache -----------------------------------------------------

    def get_cache(self, provider_name: str, payload_str: str) -> dict | None:
        """Return a cached provider response or ``None`` on miss/expiry.

        Implementations must honour the TTL configured via
        :class:`core.config.Settings.provider_cache_ttl_seconds`.
        """
        ...

    def set_cache(
        self, provider_name: str, payload_str: str, response_data: dict
    ) -> None:
        """Store one provider response in the cache."""
        ...
