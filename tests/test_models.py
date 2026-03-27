"""Unit tests for mem_bridge.models.

Tests cover:
- MemoryEntry validation and field coercion
- ConversationEntry role coercion
- Conversation properties
- MemoryProfile construction, properties, and serialisation round-trips
- Edge cases: blank fields, unknown enum values, empty collections
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from mem_bridge.models import (
    Conversation,
    ConversationEntry,
    MemoryEntry,
    MemoryProfile,
    OutputFormat,
    Role,
    SourcePlatform,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_memory_entry(**kwargs) -> MemoryEntry:
    """Return a valid MemoryEntry with sensible defaults."""
    defaults: dict = {
        "content": "User prefers dark mode in all applications.",
        "tags": ["ui", "preferences"],
    }
    defaults.update(kwargs)
    return MemoryEntry(**defaults)


def make_conversation_entry(**kwargs) -> ConversationEntry:
    """Return a valid ConversationEntry with sensible defaults."""
    defaults: dict = {
        "role": Role.USER,
        "content": "Hello, how are you?",
    }
    defaults.update(kwargs)
    return ConversationEntry(**defaults)


def make_conversation(**kwargs) -> Conversation:
    """Return a valid Conversation with two entries."""
    entries = [
        make_conversation_entry(role=Role.USER, content="What is Python?"),
        make_conversation_entry(role=Role.ASSISTANT, content="A programming language."),
    ]
    defaults: dict = {
        "title": "Test Conversation",
        "entries": entries,
    }
    defaults.update(kwargs)
    return Conversation(**defaults)


def make_profile(**kwargs) -> MemoryProfile:
    """Return a valid MemoryProfile with one memory and one conversation."""
    defaults: dict = {
        "source_platform": SourcePlatform.CHATGPT,
        "display_name": "Alice",
        "memories": [make_memory_entry()],
        "conversations": [make_conversation()],
    }
    defaults.update(kwargs)
    return MemoryProfile(**defaults)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestRole:
    """Tests for the Role enumeration."""

    def test_all_expected_values_exist(self):
        expected = {"user", "assistant", "system", "tool", "unknown"}
        assert {r.value for r in Role} == expected

    def test_str_representation(self):
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"


class TestSourcePlatform:
    """Tests for the SourcePlatform enumeration."""

    def test_all_expected_values_exist(self):
        expected = {"chatgpt", "claude", "gemini", "unknown"}
        assert {p.value for p in SourcePlatform} == expected


class TestOutputFormat:
    """Tests for the OutputFormat enumeration."""

    def test_all_expected_values_exist(self):
        expected = {"markdown", "json", "yaml", "text", "gemini"}
        assert {f.value for f in OutputFormat} == expected


# ---------------------------------------------------------------------------
# MemoryEntry tests
# ---------------------------------------------------------------------------


class TestMemoryEntry:
    """Tests for the MemoryEntry model."""

    def test_minimal_valid_entry(self):
        entry = MemoryEntry(content="Remember Python 3.12")
        assert entry.content == "Remember Python 3.12"
        assert entry.tags == []
        assert entry.metadata == {}
        assert entry.id is None
        assert entry.created_at is None
        assert entry.updated_at is None

    def test_content_is_stripped(self):
        entry = MemoryEntry(content="  some fact  ")
        assert entry.content == "some fact"

    def test_blank_content_raises(self):
        with pytest.raises(ValueError, match="must not be blank"):
            MemoryEntry(content="   ")

    def test_empty_content_raises(self):
        with pytest.raises(ValueError):
            MemoryEntry(content="")

    def test_tags_normalisation_from_list(self):
        entry = MemoryEntry(content="fact", tags=["  AI  ", "memory", ""])
        assert entry.tags == ["AI", "memory"]

    def test_tags_normalisation_from_none(self):
        entry = MemoryEntry(content="fact", tags=None)
        assert entry.tags == []

    def test_tags_normalisation_from_string(self):
        entry = MemoryEntry(content="fact", tags="single")
        assert entry.tags == ["single"]

    def test_full_entry_with_timestamps(self):
        now = datetime.now(tz=timezone.utc)
        entry = MemoryEntry(
            id="abc123",
            content="User speaks French.",
            created_at=now,
            updated_at=now,
            tags=["language"],
            metadata={"source": "memory.json"},
        )
        assert entry.id == "abc123"
        assert entry.created_at == now
        assert entry.tags == ["language"]
        assert entry.metadata["source"] == "memory.json"

    def test_serialisation_round_trip(self):
        entry = make_memory_entry(id="m1")
        data = entry.model_dump()
        restored = MemoryEntry(**data)
        assert restored.content == entry.content
        assert restored.tags == entry.tags


# ---------------------------------------------------------------------------
# ConversationEntry tests
# ---------------------------------------------------------------------------


class TestConversationEntry:
    """Tests for the ConversationEntry model."""

    def test_minimal_entry(self):
        entry = ConversationEntry()
        assert entry.role == Role.UNKNOWN
        assert entry.content == ""
        assert entry.id is None

    def test_role_coercion_from_string(self):
        entry = ConversationEntry(role="user")
        assert entry.role == Role.USER

    def test_role_coercion_case_insensitive(self):
        entry = ConversationEntry(role="ASSISTANT")
        assert entry.role == Role.ASSISTANT

    def test_unknown_role_falls_back(self):
        entry = ConversationEntry(role="nonexistent_role")
        assert entry.role == Role.UNKNOWN

    def test_role_enum_passthrough(self):
        entry = ConversationEntry(role=Role.SYSTEM)
        assert entry.role == Role.SYSTEM

    def test_full_entry(self):
        ts = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        entry = ConversationEntry(
            id="msg_001",
            role=Role.ASSISTANT,
            content="I can help with that.",
            timestamp=ts,
            model="gpt-4o",
            metadata={"tokens": 42},
        )
        assert entry.id == "msg_001"
        assert entry.model == "gpt-4o"
        assert entry.metadata["tokens"] == 42

    def test_serialisation_round_trip(self):
        entry = make_conversation_entry(content="Hello")
        data = json.loads(entry.model_dump_json())
        restored = ConversationEntry(**data)
        assert restored.content == "Hello"
        assert restored.role == Role.USER


# ---------------------------------------------------------------------------
# Conversation tests
# ---------------------------------------------------------------------------


class TestConversation:
    """Tests for the Conversation model."""

    def test_empty_conversation(self):
        conv = Conversation()
        assert conv.title == "Untitled Conversation"
        assert conv.entries == []
        assert conv.message_count == 0

    def test_message_count_property(self):
        conv = make_conversation()
        assert conv.message_count == 2

    def test_user_messages_property(self):
        conv = make_conversation()
        user_msgs = conv.user_messages
        assert len(user_msgs) == 1
        assert all(m.role == Role.USER for m in user_msgs)

    def test_assistant_messages_property(self):
        conv = make_conversation()
        assistant_msgs = conv.assistant_messages
        assert len(assistant_msgs) == 1
        assert all(m.role == Role.ASSISTANT for m in assistant_msgs)

    def test_conversation_with_metadata(self):
        conv = Conversation(
            id="conv_001",
            title="Python Discussion",
            model="claude-3-opus",
            metadata={"archived": False},
        )
        assert conv.id == "conv_001"
        assert conv.model == "claude-3-opus"
        assert conv.metadata["archived"] is False

    def test_timestamps(self):
        created = datetime(2024, 1, 1, tzinfo=timezone.utc)
        updated = datetime(2024, 6, 1, tzinfo=timezone.utc)
        conv = Conversation(created_at=created, updated_at=updated)
        assert conv.created_at == created
        assert conv.updated_at == updated


# ---------------------------------------------------------------------------
# MemoryProfile tests
# ---------------------------------------------------------------------------


class TestMemoryProfile:
    """Tests for the MemoryProfile model."""

    def test_default_profile(self):
        profile = MemoryProfile()
        assert profile.source_platform == SourcePlatform.UNKNOWN
        assert profile.display_name == "Unknown User"
        assert profile.memories == []
        assert profile.conversations == []
        assert profile.schema_version == "1.0"

    def test_blank_display_name_falls_back(self):
        profile = MemoryProfile(display_name="")
        assert profile.display_name == "Unknown User"

    def test_whitespace_display_name_falls_back(self):
        profile = MemoryProfile(display_name="   ")
        assert profile.display_name == "Unknown User"

    def test_source_platform_coercion_from_string(self):
        profile = MemoryProfile(source_platform="chatgpt")
        assert profile.source_platform == SourcePlatform.CHATGPT

    def test_source_platform_unknown_string(self):
        profile = MemoryProfile(source_platform="fictional_platform")
        assert profile.source_platform == SourcePlatform.UNKNOWN

    def test_memory_count_property(self):
        profile = make_profile()
        assert profile.memory_count == 1

    def test_conversation_count_property(self):
        profile = make_profile()
        assert profile.conversation_count == 1

    def test_total_message_count_property(self):
        profile = make_profile()
        # make_conversation() creates 2 entries
        assert profile.total_message_count == 2

    def test_total_message_count_multiple_conversations(self):
        convs = [make_conversation() for _ in range(3)]
        profile = MemoryProfile(conversations=convs)
        assert profile.total_message_count == 6

    def test_memory_count_empty(self):
        profile = MemoryProfile()
        assert profile.memory_count == 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def test_to_json_produces_valid_json(self):
        profile = make_profile()
        raw = profile.to_json()
        data = json.loads(raw)
        assert data["display_name"] == "Alice"
        assert data["source_platform"] == "chatgpt"

    def test_to_dict_contains_expected_keys(self):
        profile = make_profile()
        d = profile.to_dict()
        expected_keys = {
            "id", "source_platform", "display_name", "exported_at",
            "memories", "conversations", "schema_version", "metadata",
        }
        assert expected_keys <= set(d.keys())

    def test_to_dict_enums_are_strings(self):
        profile = make_profile()
        d = profile.to_dict()
        assert isinstance(d["source_platform"], str)
        assert d["source_platform"] == "chatgpt"
        for entry in d["conversations"][0]["entries"]:
            assert isinstance(entry["role"], str)

    def test_to_yaml_produces_valid_yaml(self):
        profile = make_profile()
        raw = profile.to_yaml()
        data = yaml.safe_load(raw)
        assert data["display_name"] == "Alice"
        assert isinstance(data["memories"], list)

    def test_from_json_round_trip(self):
        profile = make_profile()
        restored = MemoryProfile.from_json(profile.to_json())
        assert restored.display_name == profile.display_name
        assert restored.source_platform == profile.source_platform
        assert restored.memory_count == profile.memory_count
        assert restored.conversation_count == profile.conversation_count

    def test_from_dict_round_trip(self):
        profile = make_profile()
        restored = MemoryProfile.from_dict(profile.to_dict())
        assert restored.memories[0].content == profile.memories[0].content
        assert restored.conversations[0].title == profile.conversations[0].title

    def test_from_json_bytes(self):
        profile = make_profile()
        json_bytes = profile.to_json().encode("utf-8")
        restored = MemoryProfile.from_json(json_bytes)
        assert restored.display_name == profile.display_name

    def test_from_json_invalid_raises(self):
        with pytest.raises((ValueError, Exception)):
            MemoryProfile.from_json("{not valid json")

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def test_save_and_load_json(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "profile.json"
        profile.save(dest, fmt="json")
        assert dest.exists()
        loaded = MemoryProfile.from_file(dest)
        assert loaded.display_name == profile.display_name
        assert loaded.memory_count == profile.memory_count

    def test_save_and_load_yaml(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "profile.yaml"
        profile.save(dest, fmt="yaml")
        assert dest.exists()
        loaded = MemoryProfile.from_file(dest)
        assert loaded.display_name == profile.display_name

    def test_save_and_load_yml_extension(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "profile.yml"
        profile.save(dest, fmt="yaml")
        loaded = MemoryProfile.from_file(dest)
        assert loaded.source_platform == SourcePlatform.CHATGPT

    def test_save_unsupported_format_raises(self, tmp_path: Path):
        profile = make_profile()
        with pytest.raises(ValueError, match="Unsupported format"):
            profile.save(tmp_path / "profile.xml", fmt="xml")

    def test_from_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            MemoryProfile.from_file("/nonexistent/path/profile.json")

    def test_from_file_unsupported_extension_raises(self, tmp_path: Path):
        p = tmp_path / "profile.toml"
        p.write_text("display_name = 'Alice'")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            MemoryProfile.from_file(p)

    def test_save_creates_parent_directories(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "nested" / "dir" / "profile.json"
        profile.save(dest, fmt="json")
        assert dest.exists()

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_metadata_preserved_through_round_trip(self):
        profile = MemoryProfile(
            display_name="Bob",
            metadata={"export_version": "2.0", "user_id": "u_123"},
        )
        restored = MemoryProfile.from_dict(profile.to_dict())
        assert restored.metadata["export_version"] == "2.0"
        assert restored.metadata["user_id"] == "u_123"

    def test_multiple_memories_preserved(self):
        memories = [
            MemoryEntry(content=f"Memory fact number {i}") for i in range(5)
        ]
        profile = MemoryProfile(memories=memories)
        assert profile.memory_count == 5
        restored = MemoryProfile.from_dict(profile.to_dict())
        assert restored.memory_count == 5
        assert restored.memories[3].content == "Memory fact number 3"

    def test_exported_at_timestamp_preserved(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        profile = MemoryProfile(exported_at=ts)
        d = profile.to_dict()
        # datetime is serialised as ISO string in JSON
        assert "2024-06-15" in d["exported_at"]
        restored = MemoryProfile.from_dict(d)
        assert restored.exported_at is not None
        # Compare date portion to avoid tz representation differences
        assert restored.exported_at.year == 2024
        assert restored.exported_at.month == 6
        assert restored.exported_at.day == 15

    def test_schema_version_default(self):
        profile = MemoryProfile()
        assert profile.schema_version == "1.0"

    def test_schema_version_custom(self):
        profile = MemoryProfile(schema_version="2.3")
        assert profile.schema_version == "2.3"
