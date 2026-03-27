"""ChatGPT reader adapter for mem_bridge.

This module implements parsing of ChatGPT data-export archives (ZIP files
or extracted directories/JSON files) into canonical :class:`MemoryProfile`
objects.

ChatGPT exports typically contain:
- ``conversations.json`` – full conversation history as a JSON array
- ``memory.json``        – saved memory facts (if Memory is enabled)

Both flat ZIP archives and pre-extracted directories are supported.  A
plain ``conversations.json`` or ``memory.json`` file may also be passed
directly.

Example usage::

    from mem_bridge.adapters.chatgpt import ChatGPTAdapter

    adapter = ChatGPTAdapter()
    profile = adapter.read("chatgpt_export.zip")
    print(profile.memory_count, profile.conversation_count)
"""

from __future__ import annotations

import json
import zipfile
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


class ChatGPTParseError(ValueError):
    """Raised when a ChatGPT export file cannot be parsed."""


class ChatGPTAdapter:
    """Reader adapter for ChatGPT data exports.

    Supports the following source formats:

    * A ``.zip`` archive (as downloaded from ChatGPT's data-export feature).
    * A directory containing ``conversations.json`` and/or ``memory.json``.
    * A single ``conversations.json`` file.
    * A single ``memory.json`` file.

    The adapter produces a :class:`~mem_bridge.models.MemoryProfile` with
    ``source_platform`` set to :attr:`~mem_bridge.models.SourcePlatform.CHATGPT`.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self, source: str | Path) -> MemoryProfile:
        """Parse a ChatGPT export into a canonical :class:`MemoryProfile`.

        Parameters
        ----------
        source:
            Path to one of: a ``.zip`` archive, a directory, a
            ``conversations.json`` file, or a ``memory.json`` file.

        Returns
        -------
        MemoryProfile
            Populated canonical profile.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist on the filesystem.
        ChatGPTParseError
            If the content cannot be parsed as a valid ChatGPT export.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"ChatGPT export not found: {path}")

        conversations_data: list[dict[str, Any]] = []
        memory_data: list[dict[str, Any]] = []

        if path.is_file() and path.suffix.lower() == ".zip":
            conversations_data, memory_data = self._read_zip(path)
        elif path.is_dir():
            conversations_data, memory_data = self._read_directory(path)
        elif path.is_file():
            conversations_data, memory_data = self._read_single_file(path)
        else:
            raise ChatGPTParseError(f"Cannot determine how to read source: {path}")

        memories = self._parse_memories(memory_data)
        conversations = self._parse_conversations(conversations_data)

        return MemoryProfile(
            source_platform=SourcePlatform.CHATGPT,
            display_name=self._infer_display_name(conversations_data),
            memories=memories,
            conversations=conversations,
            metadata={
                "source_file": str(path),
                "raw_conversation_count": len(conversations_data),
                "raw_memory_count": len(memory_data),
            },
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_zip(self, path: Path) -> tuple[list[dict], list[dict]]:
        """Extract and parse a ChatGPT export ZIP archive."""
        conversations: list[dict] = []
        memories: list[dict] = []

        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                # Find conversations.json (may be nested in a sub-directory)
                conv_names = [
                    n for n in names if n.endswith("conversations.json")
                ]
                mem_names = [
                    n for n in names if n.endswith("memory.json")
                ]

                if conv_names:
                    raw = zf.read(conv_names[0])
                    conversations = self._load_json_bytes(raw, conv_names[0])
                    if not isinstance(conversations, list):
                        conversations = []

                if mem_names:
                    raw = zf.read(mem_names[0])
                    memories = self._load_json_bytes(raw, mem_names[0])
                    if not isinstance(memories, list):
                        memories = []

        except zipfile.BadZipFile as exc:
            raise ChatGPTParseError(f"Invalid ZIP file: {path}") from exc

        return conversations, memories

    def _read_directory(self, path: Path) -> tuple[list[dict], list[dict]]:
        """Read ``conversations.json`` and ``memory.json`` from a directory."""
        conversations: list[dict] = []
        memories: list[dict] = []

        conv_path = path / "conversations.json"
        if conv_path.exists():
            conversations = self._load_json_file(conv_path)
            if not isinstance(conversations, list):
                conversations = []

        mem_path = path / "memory.json"
        if mem_path.exists():
            memories = self._load_json_file(mem_path)
            if not isinstance(memories, list):
                memories = []

        return conversations, memories

    def _read_single_file(
        self, path: Path
    ) -> tuple[list[dict], list[dict]]:
        """Read a single JSON file as either conversations or memory data."""
        data = self._load_json_file(path)
        if path.name.lower() == "memory.json":
            if isinstance(data, list):
                return [], data
            return [], []
        # Default: treat as conversations.json
        if isinstance(data, list):
            return data, []
        return [], []

    # ------------------------------------------------------------------
    # JSON loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_file(path: Path) -> Any:
        """Load and parse a JSON file from disk."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ChatGPTParseError(
                f"Invalid JSON in {path}: {exc}"
            ) from exc

    @staticmethod
    def _load_json_bytes(data: bytes, name: str) -> Any:
        """Parse JSON from a bytes object (e.g. read from a ZIP entry)."""
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ChatGPTParseError(
                f"Invalid JSON in ZIP entry {name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_memories(
        self, memory_data: list[dict[str, Any]]
    ) -> list[MemoryEntry]:
        """Convert raw memory.json records into :class:`MemoryEntry` objects.

        ChatGPT memory records may take several shapes depending on the
        export version.  We handle the most common ones:

        * ``{"text": "..."}``
        * ``{"memory": "..."}``
        * ``{"content": "..."}``
        * Plain strings
        """
        entries: list[MemoryEntry] = []
        for item in memory_data:
            try:
                entry = self._parse_memory_item(item)
                if entry is not None:
                    entries.append(entry)
            except (TypeError, KeyError, ValueError):
                # Skip malformed items rather than aborting the whole import
                continue
        return entries

    @staticmethod
    def _parse_memory_item(
        item: Any,
    ) -> MemoryEntry | None:
        """Parse a single memory record into a :class:`MemoryEntry`."""
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return MemoryEntry(content=text)

        if not isinstance(item, dict):
            return None

        # Try common field names in priority order
        text: str | None = None
        for field in ("text", "memory", "content", "value", "description"):
            candidate = item.get(field)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break

        if not text:
            return None

        created_at: datetime | None = None
        updated_at: datetime | None = None

        for ts_field in ("created_at", "create_time", "timestamp"):
            raw_ts = item.get(ts_field)
            if raw_ts is not None:
                created_at = _parse_timestamp(raw_ts)
                break

        for ts_field in ("updated_at", "update_time"):
            raw_ts = item.get(ts_field)
            if raw_ts is not None:
                updated_at = _parse_timestamp(raw_ts)
                break

        metadata = {
            k: v
            for k, v in item.items()
            if k not in ("text", "memory", "content", "value", "description",
                         "created_at", "create_time", "updated_at", "update_time",
                         "timestamp", "id")
        }

        return MemoryEntry(
            id=str(item["id"]) if "id" in item else None,
            content=text,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata,
        )

    def _parse_conversations(
        self, conversations_data: list[dict[str, Any]]
    ) -> list[Conversation]:
        """Convert raw conversations.json records into :class:`Conversation` objects."""
        result: list[Conversation] = []
        for raw_conv in conversations_data:
            try:
                conv = self._parse_conversation(raw_conv)
                result.append(conv)
            except (TypeError, KeyError, ValueError):
                continue
        return result

    def _parse_conversation(self, raw: dict[str, Any]) -> Conversation:
        """Parse a single conversation record.

        ChatGPT conversations.json structure (simplified)::

            {
              "id": "...",
              "title": "...",
              "create_time": 1700000000.0,
              "update_time": 1700001000.0,
              "mapping": {
                "<node_id>": {
                  "message": {
                    "id": "...",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                    "create_time": 1700000000.0,
                    "metadata": {"model_slug": "gpt-4o"}
                  },
                  "parent": null | "<node_id>",
                  "children": ["<node_id>", ...]
                },
                ...
              }
            }
        """
        conv_id = str(raw.get("id", "")) or None
        title = raw.get("title", "Untitled Conversation") or "Untitled Conversation"
        created_at = _parse_timestamp(raw.get("create_time"))
        updated_at = _parse_timestamp(raw.get("update_time"))

        # Extract model from conversation-level metadata if available
        top_model: str | None = None

        entries: list[ConversationEntry] = []

        mapping: dict[str, Any] = raw.get("mapping") or {}
        if mapping:
            # Reconstruct the linear message order from the tree
            ordered_nodes = _linearise_mapping(mapping)
            for node in ordered_nodes:
                message = node.get("message")
                if not message:
                    continue
                entry = self._parse_message(message)
                if entry is not None:
                    entries.append(entry)
                    if entry.model and top_model is None:
                        top_model = entry.model
        else:
            # Fallback: try a flat "messages" list
            for msg in raw.get("messages") or []:
                entry = self._parse_message(msg)
                if entry is not None:
                    entries.append(entry)

        return Conversation(
            id=conv_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            entries=entries,
            model=top_model,
            metadata={
                k: v
                for k, v in raw.items()
                if k not in ("id", "title", "create_time", "update_time",
                             "mapping", "messages")
            },
        )

    @staticmethod
    def _parse_message(msg: dict[str, Any]) -> ConversationEntry | None:
        """Parse a single message node into a :class:`ConversationEntry`."""
        if not isinstance(msg, dict):
            return None

        # Author role
        author = msg.get("author") or {}
        raw_role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"

        # Content extraction – supports text parts and plain string
        content_obj = msg.get("content") or {}
        content_text = ""
        if isinstance(content_obj, str):
            content_text = content_obj
        elif isinstance(content_obj, dict):
            parts = content_obj.get("parts") or []
            text_parts = [
                p for p in parts if isinstance(p, str)
            ]
            content_text = "\n".join(text_parts)
        elif isinstance(content_obj, list):
            content_text = "\n".join(str(p) for p in content_obj if p)

        # Skip empty system bootstrap messages
        if not content_text.strip() and raw_role in ("system", "tool"):
            return None

        # Timestamp
        ts = _parse_timestamp(msg.get("create_time"))

        # Model slug from metadata
        meta: dict[str, Any] = msg.get("metadata") or {}
        model_slug: str | None = None
        if isinstance(meta, dict):
            model_slug = meta.get("model_slug") or meta.get("model") or None

        return ConversationEntry(
            id=str(msg["id"]) if "id" in msg else None,
            role=raw_role,
            content=content_text,
            timestamp=ts,
            model=model_slug,
            metadata={
                k: v
                for k, v in msg.items()
                if k not in ("id", "author", "content", "create_time", "metadata")
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_display_name(
        conversations_data: list[dict[str, Any]],
    ) -> str:
        """Attempt to infer a display name from conversation metadata."""
        # ChatGPT exports don't reliably embed the user's name, so we
        # return a sensible default.
        return "ChatGPT User"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_timestamp(value: Any) -> datetime | None:
    """Convert a raw timestamp value to a timezone-aware :class:`datetime`.

    Handles:
    - Unix epoch floats / ints (ChatGPT's most common format)
    - ISO 8601 strings
    - ``None`` / missing values

    Parameters
    ----------
    value:
        The raw timestamp from the export JSON.

    Returns
    -------
    datetime | None
        UTC datetime, or ``None`` if *value* is absent or unparseable.
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
        # Try ISO 8601 first
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        # Fall back to fromisoformat (Python 3.11+ is more permissive)
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _linearise_mapping(
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct a linear message sequence from a ChatGPT mapping tree.

    ChatGPT stores conversations as a tree where each node has a ``parent``
    pointer and a ``children`` list.  This function performs a depth-first
    traversal starting from the root node (where ``parent`` is ``None``) to
    produce an ordered list of nodes.

    Parameters
    ----------
    mapping:
        The ``mapping`` dict from a ChatGPT conversation record.

    Returns
    -------
    list[dict[str, Any]]
        Ordered list of node dicts in conversation order.
    """
    if not mapping:
        return []

    # Identify the root node (parent is None or parent key not in mapping)
    root_id: str | None = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        # Fallback: just return all nodes in dict order
        return list(mapping.values())

    ordered: list[dict[str, Any]] = []
    visited: set[str] = set()
    stack: list[str] = [root_id]

    while stack:
        current_id = stack.pop(0)  # BFS-style to preserve chat order
        if current_id in visited:
            continue
        visited.add(current_id)
        node = mapping.get(current_id)
        if node is None:
            continue
        ordered.append(node)
        for child_id in node.get("children") or []:
            if child_id not in visited:
                stack.append(child_id)

    return ordered


__all__ = ["ChatGPTAdapter", "ChatGPTParseError"]
