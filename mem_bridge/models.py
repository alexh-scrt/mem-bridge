"""Canonical Pydantic data models for mem_bridge.

This module defines the internal interchange format used by all adapters,
formatters, and the diff engine. All reader adapters produce instances of
these models; all writer adapters and formatters consume them.

Core models
-----------
- :class:`MemoryEntry`       – A single atomic memory fact or preference.
- :class:`ConversationEntry` – A single turn (message) within a conversation.
- :class:`Conversation`      – A full conversation thread with metadata.
- :class:`MemoryProfile`     – Top-level container aggregating memories and
                               conversations for a single user / export.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Role(str, Enum):  # noqa: D101
    """Speaker role within a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"


class SourcePlatform(str, Enum):
    """Originating AI platform of an export."""

    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    UNKNOWN = "unknown"


class OutputFormat(str, Enum):
    """Supported output/export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    GEMINI = "gemini"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single atomic memory fact or user preference stored by an AI platform.

    Attributes
    ----------
    id:
        Optional opaque identifier assigned by the source platform.
    content:
        The human-readable text of the memory fact.
    created_at:
        When this memory was first recorded. ``None`` if unknown.
    updated_at:
        When this memory was last updated. ``None`` if unknown.
    tags:
        Optional list of topic tags associated with this memory.
    metadata:
        Arbitrary extra fields from the source format that do not map to
        a named attribute.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, description="Opaque platform-assigned identifier.")
    content: str = Field(..., min_length=1, description="Text of the memory fact.")
    created_at: datetime | None = Field(default=None, description="Creation timestamp.")
    updated_at: datetime | None = Field(default=None, description="Last-updated timestamp.")
    tags: list[str] = Field(default_factory=list, description="Topic tags.")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra source-specific fields."
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        """Ensure content is not whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("MemoryEntry.content must not be blank or whitespace-only.")
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def normalise_tags(cls, v: Any) -> list[str]:
        """Ensure tags is always a list of non-empty stripped strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        return [str(tag).strip() for tag in v if str(tag).strip()]


class ConversationEntry(BaseModel):
    """A single turn (message) within an AI conversation.

    Attributes
    ----------
    id:
        Optional opaque message identifier from the source platform.
    role:
        The speaker role for this turn.
    content:
        The text content of the message.
    timestamp:
        When this message was created. ``None`` if unknown.
    model:
        The AI model that produced this response (for assistant turns).
    metadata:
        Arbitrary extra fields from the source format.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, description="Opaque platform-assigned message ID.")
    role: Role = Field(default=Role.UNKNOWN, description="Speaker role.")
    content: str = Field(default="", description="Text content of the message.")
    timestamp: datetime | None = Field(default=None, description="Message creation timestamp.")
    model: str | None = Field(
        default=None, description="AI model identifier for assistant turns."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra source-specific fields."
    )

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: Any) -> Role:
        """Accept plain strings and coerce to ``Role``; fall back to UNKNOWN."""
        if isinstance(v, Role):
            return v
        try:
            return Role(str(v).lower().strip())
        except ValueError:
            return Role.UNKNOWN


class Conversation(BaseModel):
    """A full conversation thread composed of ordered message turns.

    Attributes
    ----------
    id:
        Optional opaque conversation identifier from the source platform.
    title:
        Display title of the conversation.
    created_at:
        When the conversation was started.
    updated_at:
        When the conversation was last updated.
    entries:
        Ordered list of :class:`ConversationEntry` objects.
    model:
        Primary AI model used in this conversation.
    metadata:
        Arbitrary extra fields from the source format.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, description="Opaque conversation identifier.")
    title: str = Field(default="Untitled Conversation", description="Display title.")
    created_at: datetime | None = Field(default=None, description="Conversation start time.")
    updated_at: datetime | None = Field(default=None, description="Last message time.")
    entries: list[ConversationEntry] = Field(
        default_factory=list, description="Ordered list of conversation turns."
    )
    model: str | None = Field(
        default=None, description="Primary AI model used in this conversation."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra source-specific fields."
    )

    @property
    def message_count(self) -> int:
        """Return the total number of turns in this conversation."""
        return len(self.entries)

    @property
    def user_messages(self) -> list[ConversationEntry]:
        """Return only the user-role entries."""
        return [e for e in self.entries if e.role == Role.USER]

    @property
    def assistant_messages(self) -> list[ConversationEntry]:
        """Return only the assistant-role entries."""
        return [e for e in self.entries if e.role == Role.ASSISTANT]


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


class MemoryProfile(BaseModel):
    """Top-level canonical container for a user's AI memory export.

    This is the internal interchange object that flows between all adapters,
    formatters, and the diff engine.

    Attributes
    ----------
    id:
        Optional opaque profile identifier.
    source_platform:
        The platform this profile was exported from.
    display_name:
        Human-readable label for this profile (e.g. user's name or export
        file stem).
    exported_at:
        When the export was generated. ``None`` if unknown.
    memories:
        List of discrete memory facts extracted from the export.
    conversations:
        List of full conversation threads.
    schema_version:
        Semantic version string for the mem_bridge internal schema. Defaults
        to ``"1.0"``.
    metadata:
        Arbitrary extra fields from the source format not captured elsewhere.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, description="Opaque profile identifier.")
    source_platform: SourcePlatform = Field(
        default=SourcePlatform.UNKNOWN,
        description="Originating AI platform.",
    )
    display_name: str = Field(
        default="Unknown User",
        description="Human-readable profile label.",
    )
    exported_at: datetime | None = Field(
        default=None, description="Timestamp of the source export."
    )
    memories: list[MemoryEntry] = Field(
        default_factory=list, description="List of atomic memory facts."
    )
    conversations: list[Conversation] = Field(
        default_factory=list, description="List of full conversation threads."
    )
    schema_version: str = Field(
        default="1.0",
        description="mem_bridge internal schema version.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra source-specific fields."
    )

    @field_validator("source_platform", mode="before")
    @classmethod
    def coerce_source_platform(cls, v: Any) -> SourcePlatform:
        """Accept plain strings and coerce to :class:`SourcePlatform`."""
        if isinstance(v, SourcePlatform):
            return v
        try:
            return SourcePlatform(str(v).lower().strip())
        except ValueError:
            return SourcePlatform.UNKNOWN

    @model_validator(mode="after")
    def ensure_display_name_not_blank(self) -> "MemoryProfile":
        """Fall back to 'Unknown User' if display_name is blank."""
        if not self.display_name or not self.display_name.strip():
            self.display_name = "Unknown User"
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def memory_count(self) -> int:
        """Return the number of memory entries in this profile."""
        return len(self.memories)

    @property
    def conversation_count(self) -> int:
        """Return the number of conversations in this profile."""
        return len(self.conversations)

    @property
    def total_message_count(self) -> int:
        """Return the total number of messages across all conversations."""
        return sum(c.message_count for c in self.conversations)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        """Serialise this profile to a JSON string.

        Parameters
        ----------
        indent:
            Number of spaces to use for JSON indentation.

        Returns
        -------
        str
            Pretty-printed JSON representation.
        """
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain Python dictionary representation of this profile.

        Enum values are serialised as their string values so the result is
        JSON/YAML-serialisable without further conversion.

        Returns
        -------
        dict[str, Any]
            Dictionary representation with enum values as strings.
        """
        return json.loads(self.model_dump_json())

    def to_yaml(self) -> str:
        """Serialise this profile to a YAML string.

        Returns
        -------
        str
            YAML representation of the profile.
        """
        return yaml.dump(
            self.to_dict(),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "MemoryProfile":
        """Deserialise a :class:`MemoryProfile` from a JSON string or bytes.

        Parameters
        ----------
        data:
            JSON string or bytes produced by :meth:`to_json`.

        Returns
        -------
        MemoryProfile
            The deserialised profile instance.

        Raises
        ------
        ValueError
            If *data* is not valid JSON or does not match the model schema.
        """
        return cls.model_validate_json(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryProfile":
        """Construct a :class:`MemoryProfile` from a plain dictionary.

        Parameters
        ----------
        data:
            Dictionary that maps to the model's fields, typically obtained
            by deserialising a JSON or YAML file.

        Returns
        -------
        MemoryProfile
            The constructed profile instance.

        Raises
        ------
        ValueError
            If *data* does not conform to the model schema.
        """
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "MemoryProfile":
        """Load a :class:`MemoryProfile` from a JSON or YAML file on disk.

        The file format is inferred from the file extension:
        - ``.json`` – parsed as JSON
        - ``.yaml`` / ``.yml`` – parsed as YAML

        Parameters
        ----------
        path:
            Path to the source file.

        Returns
        -------
        MemoryProfile
            The loaded profile instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported or the content is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile file not found: {path}")

        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8")

        if suffix == ".json":
            return cls.from_json(text)
        elif suffix in (".yaml", ".yml"):
            raw = yaml.safe_load(text)
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a YAML mapping at the top level, got {type(raw)}.")
            return cls.from_dict(raw)
        else:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. Expected .json, .yaml, or .yml."
            )

    def save(self, path: str | Path, fmt: str = "json") -> None:
        """Persist this profile to a file.

        Parameters
        ----------
        path:
            Destination file path.
        fmt:
            Format to write: ``'json'`` or ``'yaml'``. Defaults to ``'json'``.

        Raises
        ------
        ValueError
            If *fmt* is not ``'json'`` or ``'yaml'``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fmt = fmt.lower().strip()
        if fmt == "json":
            path.write_text(self.to_json(), encoding="utf-8")
        elif fmt in ("yaml", "yml"):
            path.write_text(self.to_yaml(), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format {fmt!r}. Use 'json' or 'yaml'.")


__all__ = [
    "ConversationEntry",
    "Conversation",
    "MemoryEntry",
    "MemoryProfile",
    "OutputFormat",
    "Role",
    "SourcePlatform",
]
