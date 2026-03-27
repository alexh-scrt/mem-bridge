"""Claude reader adapter for mem_bridge.

This module implements parsing of Claude conversation export JSON files into
canonical :class:`MemoryProfile` objects.

Claude exports conversations as a JSON array where each element represents a
conversation thread.  The structure used by Anthropic's data-export feature
looks roughly like::

    [
      {
        "uuid": "<id>",
        "name": "<title>",
        "created_at": "<ISO timestamp>",
        "updated_at": "<ISO timestamp>",
        "account": {"uuid": "<user_id>"},
        "chat_messages": [
          {
            "uuid": "<msg_id>",
            "sender": "human" | "assistant",
            "text": "<message text>",
            "created_at": "<ISO timestamp>",
            "updated_at": "<ISO timestamp>",
            "attachments": [...],
            "files": [...]
          },
          ...
        ]
      },
      ...
    ]

Claude does not currently export a separate "memory" file; memories are
therefore extracted as an empty list unless the input file contains a
``memories`` key at the top level.

Example usage::

    from mem_bridge.adapters.claude import ClaudeAdapter

    adapter = ClaudeAdapter()
    profile = adapter.read("claude_export.json")
    print(profile.conversation_count)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mem_bridge.models import (
    Conversation,
    ConversationEntry,
    MemoryEntry,
    MemoryProfile,
    Role,
    SourcePlatform,
)


class ClaudeParseError(ValueError):
    """Raised when a Claude export file cannot be parsed."""


class ClaudeAdapter:
    """Reader adapter for Claude conversation exports.

    Accepts a JSON file exported from Anthropic's data-export feature
    (Settings → Privacy → Export data).  The file may contain either:

    * A JSON **array** of conversation objects (the most common format).
    * A JSON **object** with a ``conversations`` key (alternative wrapper).
    * A JSON object with both ``conversations`` and ``memories`` keys.

    The adapter produces a :class:`~mem_bridge.models.MemoryProfile` with
    ``source_platform`` set to :attr:`~mem_bridge.models.SourcePlatform.CLAUDE`.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self, source: str | Path) -> MemoryProfile:
        """Parse a Claude export JSON file into a canonical :class:`MemoryProfile`.

        Parameters
        ----------
        source:
            Path to the Claude export JSON file.

        Returns
        -------
        MemoryProfile
            Populated canonical profile.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist on the filesystem.
        ClaudeParseError
            If the content cannot be parsed as a valid Claude export.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Claude export not found: {path}")

        raw = self._load_json_file(path)
        conversations_data, memory_data, top_metadata = self._unwrap(raw, path)

        memories = self._parse_memories(memory_data)
        conversations = self._parse_conversations(conversations_data)
        display_name = self._infer_display_name(conversations_data, top_metadata)

        return MemoryProfile(
            source_platform=SourcePlatform.CLAUDE,
            display_name=display_name,
            memories=memories,
            conversations=conversations,
            metadata={
                "source_file": str(path),
                "raw_conversation_count": len(conversations_data),
                **{k: v for k, v in top_metadata.items()
                   if k not in ("conversations", "memories", "chat_messages")},
            },
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_file(path: Path) -> Any:
        """Load and parse a JSON file from disk."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ClaudeParseError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Structure detection
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(
        raw: Any,
        path: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Detect the top-level structure and return (conversations, memories, metadata).

        Parameters
        ----------
        raw:
            Parsed JSON value from the export file.
        path:
            Original file path (used for error messages).

        Returns
        -------
        tuple
            A 3-tuple of (conversations_list, memories_list, extra_metadata).

        Raises
        ------
        ClaudeParseError
            If the structure is not recognised.
        """
        if isinstance(raw, list):
            # Plain array of conversation objects
            return raw, [], {}

        if isinstance(raw, dict):
            conversations: list[dict[str, Any]] = []
            memories: list[dict[str, Any]] = []

            # Extract conversations
            conv_value = raw.get("conversations")
            if isinstance(conv_value, list):
                conversations = conv_value
            elif "chat_messages" in raw:
                # Single conversation wrapped in an object
                conversations = [raw]
            elif "uuid" in raw or "id" in raw:
                # Single bare conversation object
                conversations = [raw]

            # Extract memories if present
            mem_value = raw.get("memories")
            if isinstance(mem_value, list):
                memories = mem_value

            extra: dict[str, Any] = {
                k: v for k, v in raw.items()
                if k not in ("conversations", "memories")
            }
            return conversations, memories, extra

        raise ClaudeParseError(
            f"Unexpected top-level JSON type in {path}: expected array or object, "
            f"got {type(raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_memories(
        self, memory_data: list[dict[str, Any]]
    ) -> list[MemoryEntry]:
        """Convert raw memory records into :class:`MemoryEntry` objects."""
        entries: list[MemoryEntry] = []
        for item in memory_data:
            try:
                entry = self._parse_memory_item(item)
                if entry is not None:
                    entries.append(entry)
            except (TypeError, KeyError, ValueError):
                continue
        return entries

    @staticmethod
    def _parse_memory_item(item: Any) -> MemoryEntry | None:
        """Parse a single memory record."""
        if isinstance(item, str):
            text = item.strip()
            return MemoryEntry(content=text) if text else None

        if not isinstance(item, dict):
            return None

        text: str | None = None
        for field in ("text", "content", "memory", "value", "description"):
            candidate = item.get(field)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break

        if not text:
            return None

        created_at = _parse_iso_timestamp(item.get("created_at"))
        updated_at = _parse_iso_timestamp(item.get("updated_at"))

        item_id = item.get("uuid") or item.get("id")

        metadata = {
            k: v for k, v in item.items()
            if k not in ("text", "content", "memory", "value", "description",
                         "created_at", "updated_at", "uuid", "id")
        }

        return MemoryEntry(
            id=str(item_id) if item_id is not None else None,
            content=text,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

    def _parse_conversations(
        self, conversations_data: list[dict[str, Any]]
    ) -> list[Conversation]:
        """Convert raw conversation records into :class:`Conversation` objects."""
        result: list[Conversation] = []
        for raw_conv in conversations_data:
            try:
                conv = self._parse_conversation(raw_conv)
                result.append(conv)
            except (TypeError, KeyError, ValueError):
                continue
        return result

    def _parse_conversation(self, raw: dict[str, Any]) -> Conversation:
        """Parse a single Claude conversation record.

        Expected structure::

            {
              "uuid": "<id>",
              "name": "<title>",
              "created_at": "2024-01-15T10:00:00.000000Z",
              "updated_at": "2024-01-15T10:05:00.000000Z",
              "account": {"uuid": "<user_id>"},
              "chat_messages": [ ... ]
            }
        """
        conv_id = (
            str(raw["uuid"]) if "uuid" in raw
            else str(raw["id"]) if "id" in raw
            else None
        )
        title = (
            raw.get("name")
            or raw.get("title")
            or "Untitled Conversation"
        )
        created_at = _parse_iso_timestamp(raw.get("created_at"))
        updated_at = _parse_iso_timestamp(raw.get("updated_at"))

        messages_raw = raw.get("chat_messages") or raw.get("messages") or []
        entries = self._parse_messages(messages_raw)

        # Model: Claude doesn't always embed the model in the export;
        # try to extract it from message metadata if present.
        model: str | None = None
        for entry in entries:
            if entry.model:
                model = entry.model
                break

        exclude_keys = {
            "uuid", "id", "name", "title",
            "created_at", "updated_at",
            "chat_messages", "messages",
        }
        metadata = {k: v for k, v in raw.items() if k not in exclude_keys}

        return Conversation(
            id=conv_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            entries=entries,
            model=model,
            metadata=metadata,
        )

    def _parse_messages(
        self, messages_raw: list[Any]
    ) -> list[ConversationEntry]:
        """Parse a list of raw message dicts into :class:`ConversationEntry` objects."""
        entries: list[ConversationEntry] = []
        for msg in messages_raw:
            try:
                entry = self._parse_message(msg)
                if entry is not None:
                    entries.append(entry)
            except (TypeError, KeyError, ValueError):
                continue
        return entries

    @staticmethod
    def _parse_message(msg: Any) -> ConversationEntry | None:
        """Parse a single Claude chat message.

        Claude messages use ``sender`` (``'human'`` | ``'assistant'``) rather
        than ``role``.  We normalise ``'human'`` → ``Role.USER``.
        """
        if not isinstance(msg, dict):
            return None

        # Sender / role
        sender = msg.get("sender") or msg.get("role") or "unknown"
        if isinstance(sender, str):
            sender = sender.lower().strip()
        # Normalise Claude-specific role names
        if sender == "human":
            sender = "user"

        # Content – Claude uses "text" for the message body
        content_text = ""
        for field in ("text", "content"):
            candidate = msg.get(field)
            if isinstance(candidate, str):
                content_text = candidate
                break
            elif isinstance(candidate, list):
                # Some versions wrap content in a list of blocks
                parts = []
                for block in candidate:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        text_val = block.get("text") or block.get("content") or ""
                        if isinstance(text_val, str):
                            parts.append(text_val)
                content_text = "\n".join(parts)
                break

        msg_id = msg.get("uuid") or msg.get("id")
        ts = _parse_iso_timestamp(msg.get("created_at") or msg.get("timestamp"))

        # Model information (present in some export variants)
        model: str | None = msg.get("model") or None

        exclude_keys = {
            "uuid", "id", "sender", "role", "text", "content",
            "created_at", "updated_at", "timestamp", "model",
        }
        metadata = {k: v for k, v in msg.items() if k not in exclude_keys}

        return ConversationEntry(
            id=str(msg_id) if msg_id is not None else None,
            role=sender,
            content=content_text,
            timestamp=ts,
            model=model,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_display_name(
        conversations_data: list[dict[str, Any]],
        top_metadata: dict[str, Any],
    ) -> str:
        """Attempt to infer a display name from the export metadata."""
        # Try top-level metadata keys
        for key in ("name", "display_name", "user_name", "username", "email"):
            value = top_metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Try account info inside conversations
        for conv in conversations_data:
            account = conv.get("account")
            if isinstance(account, dict):
                for key in ("name", "display_name", "email"):
                    value = account.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        return "Claude User"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO 8601 timestamp string (or Unix epoch number) to a UTC datetime.

    Parameters
    ----------
    value:
        A string in ISO 8601 format, a Unix epoch number, or ``None``.

    Returns
    -------
    datetime | None
        UTC-aware datetime, or ``None`` if *value* is absent or unparseable.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

        # Try common ISO 8601 variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        # Python 3.11 fromisoformat handles most remaining variants
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None

    return None


__all__ = ["ClaudeAdapter", "ClaudeParseError"]
