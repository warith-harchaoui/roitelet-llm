"""MongoDB-backed implementation of the :class:`~core.storage_protocol.Storage` Protocol.

When the JSON-per-file backend (the default) starts to feel heavy —
typically once telemetry crosses a few thousand records or the deploy
needs multi-process safety beyond what filesystem rename + lru_cache
can offer — flip ``ROITELET_STORAGE_BACKEND=mongodb`` and point
``ROITELET_MONGO_URI`` at a server. The conversation, telemetry, and
settings shapes are unchanged because they're Pydantic models, which
serialise straight to BSON-friendly dicts.

Why MongoDB rather than SQLite or Postgres
------------------------------------------
The Roitelet storage layer is already document-shaped:
conversations and telemetry are large nested JSON blobs (router
decision, candidate responses, judge transcript, …) that we never
query into. MongoDB's document store is the natural fit — no schema
migration, no JSON-extract dance.

Trade-offs we accept:

* an external service (the JSON backend was zero-deps);
* eventual-write semantics under heavy fan-out — Roitelet is not
  multi-process-critical, so this is fine for our case;
* ``pymongo`` is a sync driver. The Roitelet pipeline is async but
  storage calls happen in short bursts (one ``save_telemetry``,
  one ``append_message``) that the GIL handles without deadlock.
  A future motor-based async backend can replace this module
  without touching call sites.

What the backend stores
-----------------------
One database, three collections:

* ``conversations`` — full :class:`~core.schemas.Conversation`
  documents, ``_id`` = ``conversation_id``.
* ``telemetry`` — :class:`~core.schemas.TelemetryRecord` documents
  with a TTL-free index on ``created_at`` so ``list_telemetry`` is
  O(log n) instead of O(n) like the JSON glob.
* ``app_settings`` — single document with ``_id="singleton"``
  carrying the persisted control-room payload.

The provider cache is still backed by the JSON backend's
``data/cache/*.jsonl`` files when the MongoDB backend is active —
high-churn rows with a TTL fit a file better than a collection, and
the cache is opt-in (default off) so most users never touch it.

Selection
---------
:func:`core.storage.get_storage` consults
``ROITELET_STORAGE_BACKEND`` (env var). Supported values:

* unset / ``json`` — :class:`core.storage.StorageManager` (default).
* ``mongodb`` — this class. Requires ``pip install -e .[scale]``
  to pull in ``pymongo``.

If the env var is ``mongodb`` but the optional dep is missing the
factory raises a clear ``ImportError`` rather than silently falling
back to JSON.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings
from .schemas import AppSettingsPayload, Conversation, ConversationMessage, TelemetryRecord

logger = logging.getLogger(__name__)


def _to_doc(model) -> dict:
    """Round-trip a Pydantic model to a MongoDB-friendly dict.

    Roitelet's models contain ``datetime`` objects (tz-aware UTC) plus
    enum-flavoured Literal strings. Both are already BSON-friendly via
    ``model_dump``; the explicit dict materialisation here just keeps
    ``model_dump`` calls out of every read site.
    """
    return model.model_dump()


class MongoStorageManager:
    """Document-store implementation of the Roitelet storage Protocol.

    Public surface matches :class:`core.storage.StorageManager` so the
    pipeline and the API layer are unchanged. The only behavioural
    differences:

    * ``conversation_path`` still validates the UUID for parity with
      the JSON backend's traversal-rejection contract, but returns a
      synthetic path (the backend doesn't use it for I/O).
    * ``list_conversations`` / ``list_telemetry`` sort server-side on
      ``created_at`` via an index, so they stay fast at scale.
    """

    def __init__(self, *, uri: str | None = None, db_name: str | None = None) -> None:
        """Open the Mongo connection and ensure indexes exist."""
        try:
            from pymongo import ASCENDING, DESCENDING, MongoClient  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                'pymongo is not installed. `pip install -e .[scale]` to '
                'enable the MongoDB storage backend, or unset '
                'ROITELET_STORAGE_BACKEND to use the JSON default.'
            ) from exc

        settings = get_settings()
        uri = uri or os.environ.get('ROITELET_MONGO_URI', 'mongodb://localhost:27017')
        db_name = db_name or os.environ.get('ROITELET_MONGO_DB', 'roitelet')

        # ``serverSelectionTimeoutMS`` keeps the boot path responsive
        # when the operator has flipped the env var but the server
        # isn't actually running — we'd rather fail fast than hang.
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3_000)
        self._db = self._client[db_name]
        self.conversations = self._db['conversations']
        self.telemetry = self._db['telemetry']
        self.app_settings = self._db['app_settings']

        # Indexes — newest-first listing is the dominant read pattern.
        # ``background=True`` so the call doesn't block boot on a
        # large existing collection.
        self.conversations.create_index([('created_at', DESCENDING)], background=True)
        self.telemetry.create_index([('created_at', DESCENDING)], background=True)
        # Stable lookup index on the human-facing id (separate from _id).
        self.conversations.create_index('conversation_id', unique=True, background=True)
        # Telemetry has its own UUID. The default _id is fine, but a
        # secondary unique index lets the rest of the codebase keep
        # talking about ``record_id``.
        self.telemetry.create_index('record_id', unique=True, background=True)

        # Provider cache stays JSON-backed (high-churn JSONL with TTL).
        # The path is the same as the JSON manager so existing cache
        # files keep working.
        self.cache_dir = settings.data_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # The ASCENDING import is unused at this point but kept in
        # scope for forward-compat: future indexes (e.g. on
        # router_decision.selected_model_ids for ablation queries)
        # will need it.
        _ = ASCENDING

    # ─── Conversations ──────────────────────────────────────────────

    def create_conversation(self, title: str = 'New flight') -> Conversation:
        """Create + persist a new conversation."""
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.now(UTC),
            messages=[],
        )
        self.save_conversation(conversation)
        return conversation

    def conversation_path(self, conversation_id: str) -> Path:
        """Validate the id and return a synthetic path.

        The Mongo backend doesn't use a filesystem path, but the
        Protocol returns one. We still parse the id as a UUID so the
        traversal-rejection contract is preserved across backends.
        """
        uuid.UUID(str(conversation_id))
        # Synthetic — never read or written. Mirrors the JSON layout
        # for parity in logs and audit messages.
        return Path('mongodb://conversations') / f'{conversation_id}'

    def save_conversation(self, conversation: Conversation) -> None:
        """Upsert the full conversation document."""
        self.conversations.replace_one(
            {'conversation_id': conversation.conversation_id},
            _to_doc(conversation),
            upsert=True,
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Fetch a conversation by id, or ``None`` on miss."""
        try:
            uuid.UUID(str(conversation_id))
        except ValueError:
            return None
        doc = self.conversations.find_one({'conversation_id': conversation_id}, {'_id': 0})
        return Conversation.model_validate(doc) if doc else None

    def list_conversations(self) -> list[Conversation]:
        """Newest-first list. Server-side sort via the index."""
        cursor = self.conversations.find({}, {'_id': 0}).sort('created_at', -1)
        return [Conversation.model_validate(doc) for doc in cursor]

    def append_message(
        self, conversation_id: str, message: ConversationMessage,
    ) -> Conversation:
        """Append a message via ``$push`` for atomicity under fan-out."""
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f'Conversation not found: {conversation_id}')
        # Use an atomic push so two concurrent appenders never lose a
        # message. The Python copy below stays consistent because we
        # re-read in the same call.
        self.conversations.update_one(
            {'conversation_id': conversation_id},
            {'$push': {'messages': _to_doc(message)}},
        )
        conversation.messages.append(message)
        return conversation

    # ─── Telemetry ──────────────────────────────────────────────────

    def save_telemetry(self, record: TelemetryRecord) -> Path:
        """Persist one telemetry record."""
        self.telemetry.insert_one(_to_doc(record))
        return Path('mongodb://telemetry') / f'{record.record_id}'

    def list_telemetry(self) -> list[TelemetryRecord]:
        """Newest-first telemetry list, indexed sort."""
        cursor = self.telemetry.find({}, {'_id': 0}).sort('created_at', -1)
        return [TelemetryRecord.model_validate(doc) for doc in cursor]

    # ─── App settings ───────────────────────────────────────────────

    def load_app_settings(self) -> AppSettingsPayload:
        """Read the single settings document, or fall back to env defaults."""
        doc = self.app_settings.find_one({'_id': 'singleton'}, {'_id': 0})
        if doc is None:
            settings = get_settings()
            return AppSettingsPayload(
                openrouter_api_key=settings.openrouter_api_key,
                openai_compatible_api_key=settings.openai_compatible_api_key,
                openai_compatible_base_url=settings.openai_compatible_base_url,
                openai_compatible_model=settings.openai_compatible_model,
                ollama_base_url=settings.local_llm_base_url,
                local_synthesis_model=settings.local_llm_model,
                local_vlm_model=settings.local_vlm_model,
            )
        return AppSettingsPayload.model_validate(doc)

    def save_app_settings(self, payload: AppSettingsPayload) -> None:
        """Upsert the settings singleton."""
        self.app_settings.replace_one(
            {'_id': 'singleton'},
            {**_to_doc(payload), '_id': 'singleton'},
            upsert=True,
        )

    # ─── Provider cache ─────────────────────────────────────────────
    #
    # Same JSONL-on-disk implementation the JSON backend uses. The
    # cache is opt-in (TTL=0 disables it) and its access pattern
    # (append + tail-scan with TTL) is a poor fit for a document
    # store. Re-using the JSON behaviour keeps both backends behaving
    # identically on the cache axis.

    def get_cache(self, provider_name: str, payload_str: str) -> dict | None:
        """Return a cached provider response or ``None`` on miss/expiry."""
        ttl = get_settings().provider_cache_ttl_seconds
        if ttl == 0:
            return None
        path = self.cache_dir / f'{provider_name}.jsonl'
        if not path.exists():
            return None
        match: dict | None = None
        match_cached_at: datetime | None = None
        try:
            with path.open('r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get('payload') != payload_str:
                        continue
                    match = record.get('response')
                    raw_ts = record.get('cached_at')
                    match_cached_at = (
                        datetime.fromisoformat(raw_ts) if raw_ts else None
                    )
        except Exception as exc:
            logger.warning('Provider cache read failed for %s: %s', provider_name, exc)
            return None
        if match is None:
            return None
        if ttl < 0:
            return match
        if match_cached_at is None:
            return None
        age = (datetime.now(UTC) - match_cached_at).total_seconds()
        return match if age <= ttl else None

    def set_cache(self, provider_name: str, payload_str: str, response_data: dict) -> None:
        """Append a provider response to the JSONL cache."""
        if get_settings().provider_cache_ttl_seconds == 0:
            return
        path = self.cache_dir / f'{provider_name}.jsonl'
        record = {
            'payload': payload_str,
            'response': response_data,
            'cached_at': datetime.now(UTC).isoformat(),
        }
        try:
            # Atomic-append via tempfile+rename is overkill for a
            # high-churn cache; a plain ``a`` open is fine here.
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as exc:
            logger.warning('Provider cache write failed for %s: %s', provider_name, exc)


# Avoid an unused-import warning on the helper at module load —
# ``tempfile`` and ``contextlib`` are listed because the cache write
# path may evolve to an atomic variant; remove if/when that lands.
_ = tempfile
_ = contextlib
