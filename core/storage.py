"""Persistence helpers for conversations, telemetry, and user settings.

Examples
--------
>>> from core.storage import StorageManager
>>> storage = StorageManager()
>>> convo = storage.create_conversation(title="Demo")
>>> convo.title
'Demo'

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_settings
from .schemas import AppSettingsPayload, Conversation, ConversationMessage, TelemetryRecord

logger = logging.getLogger(__name__)


class StorageManager:
    """Small JSON-backed persistence layer.

    This class intentionally favors readability and hackability over databases.
    It works well for local development, demos, and early self-hosted setups.
    """

    def __init__(self) -> None:
        """Initialize all required directories."""
        settings = get_settings()
        self.root = settings.data_dir
        self.conversations_dir = self.root / 'conversations'
        self.telemetry_dir = self.root / 'telemetry'
        self.runtime_dir = self.root / 'runtime'
        self.cache_dir = self.root / 'cache'
        for directory in (self.conversations_dir, self.telemetry_dir, self.runtime_dir, self.cache_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default: Any) -> Any:
        """Read JSON content from disk.

        Parameters
        ----------
        path:
            File to read.
        default:
            Value returned when the file does not exist.

        Returns
        -------
        Any
            Parsed JSON content or the provided default.
        """
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding='utf-8'))

    def _write_json(self, path: Path, payload: Any) -> None:
        """Write JSON content to disk atomically with pretty formatting.

        An atomic write (write-then-rename) prevents file corruption when
        two concurrent requests flush the same file simultaneously.

        Parameters
        ----------
        path:
            Output file path.
        payload:
            JSON-serializable object.
        """
        content = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        dir_ = path.parent
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def create_conversation(self, title: str = 'New flight') -> Conversation:
        """Create a conversation.

        Parameters
        ----------
        title:
            Display title used in the left history panel.

        Returns
        -------
        Conversation
            Persisted conversation object.
        """
        conversation = Conversation(
            conversation_id=str(uuid.uuid4()),
            title=title,
            created_at=datetime.now(timezone.utc),
            messages=[],
        )
        self.save_conversation(conversation)
        return conversation

    def conversation_path(self, conversation_id: str) -> Path:
        """Return the JSON path for a conversation.

        Validates the identifier is a UUID so untrusted callers (HTTP path
        params) cannot escape ``conversations_dir`` via traversal sequences.
        """
        # UUID() rejects traversal payloads ("..", slashes, NULs) by construction.
        uuid.UUID(str(conversation_id))
        return self.conversations_dir / f'{conversation_id}.json'

    def save_conversation(self, conversation: Conversation) -> None:
        """Persist a conversation to disk."""
        path = self.conversation_path(conversation.conversation_id)
        self._write_json(path, conversation.model_dump())

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load a conversation if it exists."""
        try:
            path = self.conversation_path(conversation_id)
        except ValueError:
            return None
        payload = self._read_json(path, None)
        return Conversation.model_validate(payload) if payload else None

    def list_conversations(self) -> List[Conversation]:
        """List all persisted conversations sorted by newest first."""
        conversations = [
            Conversation.model_validate(self._read_json(path, {}))
            for path in sorted(self.conversations_dir.glob('*.json'), reverse=True)
        ]
        return sorted(conversations, key=lambda item: item.created_at, reverse=True)

    def append_message(self, conversation_id: str, message: ConversationMessage) -> Conversation:
        """Append a message to a conversation and return the updated object."""
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f'Conversation not found: {conversation_id}')
        conversation.messages.append(message)
        self.save_conversation(conversation)
        return conversation

    def save_telemetry(self, record: TelemetryRecord) -> Path:
        """Persist one telemetry record and return its path."""
        path = self.telemetry_dir / f'{record.record_id}.json'
        self._write_json(path, record.model_dump())
        return path

    def list_telemetry(self) -> List[TelemetryRecord]:
        """Load all telemetry records from disk."""
        records = [
            TelemetryRecord.model_validate(self._read_json(path, {}))
            for path in sorted(self.telemetry_dir.glob('*.json'), reverse=True)
        ]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def get_cache(self, provider_name: str, payload_str: str) -> Optional[dict]:
        """Retrieve a cached provider response, honouring the configured TTL.

        Returns ``None`` (cache miss) when:
        * the cache file doesn't exist,
        * no record matches ``payload_str``,
        * the most recent matching record is older than the configured TTL,
        * caching is disabled (``ROITELET_PROVIDER_CACHE_TTL`` = 0).

        A negative TTL means "cache forever" — the pre-TTL behaviour, kept
        as an explicit opt-in so an unintended default can't silently serve
        stale answers in perpetuity.
        """
        ttl = get_settings().provider_cache_ttl_seconds
        if ttl == 0:
            return None  # cache disabled
        path = self.cache_dir / f'{provider_name}.jsonl'
        if not path.exists():
            return None
        match = None
        match_cached_at: Optional[datetime] = None
        try:
            with path.open('r', encoding='utf-8') as f:
                # JSONL accumulates duplicates on cache overwrite; the *last*
                # matching record wins so updates supersede old ones.
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
        # ttl < 0 = cache forever; skip the freshness check entirely.
        if ttl < 0:
            return match
        if match_cached_at is None:
            return None  # legacy record without a timestamp — treat as stale
        age = (datetime.now(timezone.utc) - match_cached_at).total_seconds()
        return match if age <= ttl else None

    def set_cache(self, provider_name: str, payload_str: str, response_data: dict) -> None:
        """Append a provider response to the JSONL cache.

        No-op when caching is disabled (``ROITELET_PROVIDER_CACHE_TTL`` = 0)
        so the cache file doesn't grow when reads never hit.
        """
        if get_settings().provider_cache_ttl_seconds == 0:
            return
        path = self.cache_dir / f'{provider_name}.jsonl'
        record = {
            'payload': payload_str,
            'response': response_data,
            'cached_at': datetime.now(timezone.utc).isoformat(),
        }
        try:
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as exc:
            logger.warning('Provider cache write failed for %s: %s', provider_name, exc)

    def settings_path(self) -> Path:
        """Return the path used for persisted UI settings."""
        return self.runtime_dir / 'settings.json'

    def load_app_settings(self) -> AppSettingsPayload:
        """Load UI-edited settings, falling back to environment defaults.

        Returns
        -------
        AppSettingsPayload
            The active global configuration settings for the system.
        """
        settings = get_settings()
        payload = self._read_json(self.settings_path(), None)
        if payload is None:
            return AppSettingsPayload(
                openrouter_api_key=settings.openrouter_api_key,
                openai_compatible_api_key=settings.openai_compatible_api_key,
                openai_compatible_base_url=settings.openai_compatible_base_url,
                openai_compatible_model=settings.openai_compatible_model,
                ollama_base_url=settings.local_llm_base_url,
                local_synthesis_model=settings.local_llm_model,
                local_vlm_model=settings.local_vlm_model,
            )
        return AppSettingsPayload.model_validate(payload)

    def save_app_settings(self, payload: AppSettingsPayload) -> None:
        """Persist the control-room settings edited from the web UI."""
        self._write_json(self.settings_path(), payload.model_dump())


@lru_cache(maxsize=1)
def get_storage() -> StorageManager:
    """Return the process-wide :class:`StorageManager` instance.

    Wrapped in :func:`functools.lru_cache` so the manager is built on
    first call (no import-time filesystem side effects) and shared
    everywhere afterwards. Tests reset the cache with
    ``get_storage.cache_clear()`` to force a fresh instance against
    ``tmp_path``-rooted settings.
    """
    return StorageManager()


def __getattr__(name: str):
    """Backwards-compatible lazy access for ``from core.storage import storage``.

    Existing call sites that did ``from core.storage import storage`` and
    then used ``storage.<method>`` keep working — accessing ``storage`` on
    this module simply returns the cached singleton. New code should call
    :func:`get_storage` instead so test overrides (and future DI) compose.
    """
    if name == 'storage':
        return get_storage()
    raise AttributeError(f"module 'core.storage' has no attribute {name!r}")
