"""Unit tests for the ChatGPT and Claude reader adapters.

Tests verify:
- Correct parsing of fixture JSON files into canonical MemoryProfile objects
- Conversation count, message count, and role normalisation
- Timestamp parsing
- Metadata extraction
- Edge cases: empty arrays, malformed records, missing fields
- Error handling for non-existent files and bad ZIP archives
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mem_bridge.adapters.chatgpt import ChatGPTAdapter, ChatGPTParseError
from mem_bridge.adapters.claude import ClaudeAdapter, ClaudeParseError
from mem_bridge.models import MemoryProfile, Role, SourcePlatform

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CHATGPT_FIXTURE = FIXTURES_DIR / "chatgpt_export.json"
CLAUDE_FIXTURE = FIXTURES_DIR / "claude_export.json"


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def make_chatgpt_zip(tmp_path: Path, conversations: list, memories: list | None = None) -> Path:
    """Create a temporary ChatGPT-style ZIP archive for testing."""
    zip_path = tmp_path / "chatgpt_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
        if memories is not None:
            zf.writestr("memory.json", json.dumps(memories))
    return zip_path


def make_minimal_chatgpt_conversation(
    conv_id: str = "conv_1",
    title: str = "Test",
    user_text: str = "Hello",
    asst_text: str = "Hi there",
    model: str = "gpt-4o",
) -> dict:
    """Build a minimal ChatGPT conversation dict with a mapping tree."""
    return {
        "id": conv_id,
        "title": title,
        "create_time": 1700000000.0,
        "update_time": 1700001000.0,
        "mapping": {
            "root": {
                "id": "root",
                "message": None,
                "parent": None,
                "children": ["u1"],
            },
            "u1": {
                "id": "u1",
                "message": {
                    "id": "msg_u1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [user_text]},
                    "create_time": 1700000010.0,
                    "metadata": {},
                },
                "parent": "root",
                "children": ["a1"],
            },
            "a1": {
                "id": "a1",
                "message": {
                    "id": "msg_a1",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": [asst_text]},
                    "create_time": 1700000020.0,
                    "metadata": {"model_slug": model},
                },
                "parent": "u1",
                "children": [],
            },
        },
    }


def make_minimal_claude_conversation(
    conv_id: str = "claude_conv_001",
    title: str = "Test Conv",
    user_text: str = "Hello",
    asst_text: str = "Hi!",
) -> dict:
    """Build a minimal Claude conversation dict."""
    return {
        "uuid": conv_id,
        "name": title,
        "created_at": "2024-01-01T10:00:00.000000Z",
        "updated_at": "2024-01-01T10:05:00.000000Z",
        "account": {"uuid": "user_001", "name": "Test User"},
        "chat_messages": [
            {
                "uuid": "msg_001",
                "sender": "human",
                "text": user_text,
                "created_at": "2024-01-01T10:00:30.000000Z",
                "updated_at": "2024-01-01T10:00:30.000000Z",
                "attachments": [],
                "files": [],
            },
            {
                "uuid": "msg_002",
                "sender": "assistant",
                "text": asst_text,
                "created_at": "2024-01-01T10:01:00.000000Z",
                "updated_at": "2024-01-01T10:01:00.000000Z",
                "attachments": [],
                "files": [],
            },
        ],
    }


# ===========================================================================
# ChatGPT Adapter Tests
# ===========================================================================


class TestChatGPTAdapterFixture:
    """Tests against the bundled chatgpt_export.json fixture."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ChatGPTAdapter()

    def test_fixture_file_exists(self):
        assert CHATGPT_FIXTURE.exists(), f"Fixture not found: {CHATGPT_FIXTURE}"

    def test_read_returns_memory_profile(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert isinstance(profile, MemoryProfile)

    def test_source_platform_is_chatgpt(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert profile.source_platform == SourcePlatform.CHATGPT

    def test_conversation_count(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert profile.conversation_count == 2

    def test_first_conversation_title(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        titles = [c.title for c in profile.conversations]
        assert "Python Best Practices" in titles

    def test_second_conversation_title(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        titles = [c.title for c in profile.conversations]
        assert "Docker Containers Overview" in titles

    def test_first_conversation_has_messages(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        python_conv = next(c for c in profile.conversations if "Python" in c.title)
        # 4 user/assistant messages (system empty message is skipped)
        assert python_conv.message_count >= 2

    def test_user_and_assistant_roles_present(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        python_conv = next(c for c in profile.conversations if "Python" in c.title)
        roles = {e.role for e in python_conv.entries}
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_conversation_ids_extracted(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        ids = [c.id for c in profile.conversations]
        assert "conv_abc123" in ids
        assert "conv_def456" in ids

    def test_model_slug_extracted(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        python_conv = next(c for c in profile.conversations if "Python" in c.title)
        # The model should be extracted from assistant message metadata
        assert python_conv.model == "gpt-4o"

    def test_timestamps_are_datetime_objects(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        for conv in profile.conversations:
            assert conv.created_at is not None
            assert isinstance(conv.created_at, datetime)
            assert conv.created_at.tzinfo is not None

    def test_no_memories_in_fixture(self):
        # The fixture JSON has no memory.json; memories should be empty
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert profile.memory_count == 0

    def test_metadata_contains_source_file(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert "source_file" in profile.metadata

    def test_display_name_is_nonempty(self):
        profile = self.adapter.read(CHATGPT_FIXTURE)
        assert profile.display_name
        assert profile.display_name.strip()


class TestChatGPTAdapterZip:
    """Tests for reading ChatGPT ZIP archives."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ChatGPTAdapter()

    def test_read_zip_with_conversations_only(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        zip_path = make_chatgpt_zip(tmp_path, conversations=[conv])
        profile = self.adapter.read(zip_path)
        assert profile.conversation_count == 1
        assert profile.memory_count == 0

    def test_read_zip_with_memories(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        memories = [
            {"id": "m1", "text": "User prefers dark mode."},
            {"id": "m2", "text": "User works with Python daily."},
        ]
        zip_path = make_chatgpt_zip(tmp_path, conversations=[conv], memories=memories)
        profile = self.adapter.read(zip_path)
        assert profile.conversation_count == 1
        assert profile.memory_count == 2

    def test_read_zip_memory_content_extracted(self, tmp_path: Path):
        memories = [{"id": "m1", "text": "User prefers dark mode."}]
        zip_path = make_chatgpt_zip(tmp_path, conversations=[], memories=memories)
        profile = self.adapter.read(zip_path)
        assert profile.memories[0].content == "User prefers dark mode."

    def test_read_zip_multiple_conversations(self, tmp_path: Path):
        convs = [
            make_minimal_chatgpt_conversation(conv_id=f"conv_{i}", title=f"Conv {i}")
            for i in range(5)
        ]
        zip_path = make_chatgpt_zip(tmp_path, conversations=convs)
        profile = self.adapter.read(zip_path)
        assert profile.conversation_count == 5

    def test_read_zip_source_platform(self, tmp_path: Path):
        zip_path = make_chatgpt_zip(tmp_path, conversations=[])
        profile = self.adapter.read(zip_path)
        assert profile.source_platform == SourcePlatform.CHATGPT

    def test_read_zip_message_roles(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation(user_text="Hi", asst_text="Hello")
        zip_path = make_chatgpt_zip(tmp_path, conversations=[conv])
        profile = self.adapter.read(zip_path)
        roles = {e.role for e in profile.conversations[0].entries}
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_invalid_zip_raises_parse_error(self, tmp_path: Path):
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"this is not a zip file")
        with pytest.raises(ChatGPTParseError, match="Invalid ZIP"):
            self.adapter.read(bad_zip)

    def test_zip_empty_conversations(self, tmp_path: Path):
        zip_path = make_chatgpt_zip(tmp_path, conversations=[])
        profile = self.adapter.read(zip_path)
        assert profile.conversation_count == 0


class TestChatGPTAdapterDirectory:
    """Tests for reading from a pre-extracted ChatGPT export directory."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ChatGPTAdapter()

    def test_read_directory_with_conversations_json(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        (tmp_path / "conversations.json").write_text(
            json.dumps([conv]), encoding="utf-8"
        )
        profile = self.adapter.read(tmp_path)
        assert profile.conversation_count == 1

    def test_read_directory_with_memory_json(self, tmp_path: Path):
        (tmp_path / "conversations.json").write_text("[]", encoding="utf-8")
        memories = [{"text": "User likes Python."}, {"text": "User uses Linux."}]
        (tmp_path / "memory.json").write_text(
            json.dumps(memories), encoding="utf-8"
        )
        profile = self.adapter.read(tmp_path)
        assert profile.memory_count == 2

    def test_read_directory_no_files(self, tmp_path: Path):
        # Empty directory should produce an empty profile
        profile = self.adapter.read(tmp_path)
        assert profile.conversation_count == 0
        assert profile.memory_count == 0

    def test_read_directory_invalid_json_raises(self, tmp_path: Path):
        (tmp_path / "conversations.json").write_text(
            "{not valid json", encoding="utf-8"
        )
        with pytest.raises(ChatGPTParseError):
            self.adapter.read(tmp_path)


class TestChatGPTAdapterSingleFile:
    """Tests for reading a single conversations.json or memory.json file."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ChatGPTAdapter()

    def test_read_conversations_json_directly(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1

    def test_read_memory_json_directly(self, tmp_path: Path):
        memories = [{"text": "Remember to exercise."}, {"text": "Prefer vim."}]
        p = tmp_path / "memory.json"
        p.write_text(json.dumps(memories), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memory_count == 2
        assert profile.conversation_count == 0

    def test_memory_json_content_parsed(self, tmp_path: Path):
        memories = [{"text": "User speaks Spanish."}]
        p = tmp_path / "memory.json"
        p.write_text(json.dumps(memories), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memories[0].content == "User speaks Spanish."


class TestChatGPTAdapterEdgeCases:
    """Edge-case tests for the ChatGPT adapter."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ChatGPTAdapter()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.adapter.read("/nonexistent/path/export.zip")

    def test_memory_with_various_field_names(self, tmp_path: Path):
        memories = [
            {"text": "Entry via text field."},
            {"memory": "Entry via memory field."},
            {"content": "Entry via content field."},
        ]
        p = tmp_path / "memory.json"
        p.write_text(json.dumps(memories), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memory_count == 3
        contents = {m.content for m in profile.memories}
        assert "Entry via text field." in contents
        assert "Entry via memory field." in contents
        assert "Entry via content field." in contents

    def test_memory_plain_string_items(self, tmp_path: Path):
        memories = ["First memory.", "Second memory."]
        p = tmp_path / "memory.json"
        p.write_text(json.dumps(memories), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memory_count == 2

    def test_malformed_memory_items_skipped(self, tmp_path: Path):
        # Null entry, dict without text, and a valid entry
        memories = [None, {"no_text_field": True}, {"text": "Valid memory."}]
        p = tmp_path / "memory.json"
        p.write_text(json.dumps(memories), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memory_count == 1
        assert profile.memories[0].content == "Valid memory."

    def test_conversation_without_mapping_uses_messages(self, tmp_path: Path):
        # Flat messages list (fallback path)
        conv = {
            "id": "conv_flat",
            "title": "Flat Conv",
            "create_time": 1700000000.0,
            "update_time": 1700001000.0,
            "messages": [
                {
                    "id": "m1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hi"]},
                    "create_time": 1700000001.0,
                    "metadata": {},
                }
            ],
        }
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1
        assert profile.conversations[0].message_count >= 1

    def test_system_empty_message_skipped(self, tmp_path: Path):
        """Empty system messages should not appear in entries."""
        conv = make_minimal_chatgpt_conversation()
        # Add an empty system message node to the mapping
        conv["mapping"]["sys_node"] = {
            "id": "sys_node",
            "message": {
                "id": "sys_msg",
                "author": {"role": "system"},
                "content": {"content_type": "text", "parts": [""]},
                "create_time": 1699999999.0,
                "metadata": {},
            },
            "parent": None,
            "children": ["root"],
        }
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        # Verify system empty message was not included
        for entry in profile.conversations[0].entries:
            if entry.role == Role.SYSTEM:
                assert entry.content.strip() != ""

    def test_timestamp_from_unix_epoch(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversations[0].created_at is not None
        assert profile.conversations[0].created_at.tzinfo == timezone.utc

    def test_message_ids_extracted(self, tmp_path: Path):
        conv = make_minimal_chatgpt_conversation()
        p = tmp_path / "conversations.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        for entry in profile.conversations[0].entries:
            assert entry.id is not None


# ===========================================================================
# Claude Adapter Tests
# ===========================================================================


class TestClaudeAdapterFixture:
    """Tests against the bundled claude_export.json fixture."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ClaudeAdapter()

    def test_fixture_file_exists(self):
        assert CLAUDE_FIXTURE.exists(), f"Fixture not found: {CLAUDE_FIXTURE}"

    def test_read_returns_memory_profile(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert isinstance(profile, MemoryProfile)

    def test_source_platform_is_claude(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert profile.source_platform == SourcePlatform.CLAUDE

    def test_conversation_count(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert profile.conversation_count == 3

    def test_conversation_titles(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        titles = {c.title for c in profile.conversations}
        assert "Async Python Discussion" in titles
        assert "Machine Learning Basics" in titles
        assert "Short Greeting" in titles

    def test_first_conversation_message_count(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        async_conv = next(
            c for c in profile.conversations if "Async" in c.title
        )
        assert async_conv.message_count == 4

    def test_human_sender_mapped_to_user_role(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        for conv in profile.conversations:
            for entry in conv.entries:
                if entry.role == Role.USER:
                    # Verify 'human' was normalised to 'user'
                    pass  # role is already USER, which is what we want
            # At least one user entry per conversation
            assert any(e.role == Role.USER for e in conv.entries)

    def test_assistant_sender_mapped_correctly(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        for conv in profile.conversations:
            assert any(e.role == Role.ASSISTANT for e in conv.entries)

    def test_conversation_ids_extracted(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        ids = {c.id for c in profile.conversations}
        assert "claude_conv_001" in ids
        assert "claude_conv_002" in ids

    def test_timestamps_are_datetime_objects(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        for conv in profile.conversations:
            assert isinstance(conv.created_at, datetime)
            assert conv.created_at.tzinfo is not None

    def test_message_timestamps_parsed(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        for conv in profile.conversations:
            for entry in conv.entries:
                assert entry.timestamp is not None
                assert isinstance(entry.timestamp, datetime)

    def test_no_memories_in_fixture(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert profile.memory_count == 0

    def test_display_name_inferred_from_account(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        # Account name "Alice Coder" should be picked up
        assert profile.display_name == "Alice Coder"

    def test_metadata_contains_source_file(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert "source_file" in profile.metadata

    def test_message_content_extracted(self):
        profile = self.adapter.read(CLAUDE_FIXTURE)
        async_conv = next(
            c for c in profile.conversations if "Async" in c.title
        )
        first_user_msg = next(e for e in async_conv.entries if e.role == Role.USER)
        assert "asyncio" in first_user_msg.content.lower()


class TestClaudeAdapterFormats:
    """Tests for various Claude export JSON structures."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ClaudeAdapter()

    def test_plain_array_format(self, tmp_path: Path):
        """Top-level array of conversations (standard export format)."""
        convs = [make_minimal_claude_conversation()]
        p = tmp_path / "claude_export.json"
        p.write_text(json.dumps(convs), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1

    def test_wrapped_object_format(self, tmp_path: Path):
        """Top-level object with 'conversations' key."""
        convs = [make_minimal_claude_conversation()]
        data = {"conversations": convs}
        p = tmp_path / "claude_export.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1

    def test_wrapped_object_with_memories(self, tmp_path: Path):
        convs = [make_minimal_claude_conversation()]
        mems = [{"text": "User prefers brevity."}, {"text": "User uses macOS."}]
        data = {"conversations": convs, "memories": mems}
        p = tmp_path / "claude_export.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1
        assert profile.memory_count == 2

    def test_single_conversation_object(self, tmp_path: Path):
        """Single conversation as a plain JSON object (no array wrapper)."""
        conv = make_minimal_claude_conversation()
        p = tmp_path / "claude_export.json"
        p.write_text(json.dumps(conv), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 1

    def test_role_field_instead_of_sender(self, tmp_path: Path):
        """Some export variants use 'role' instead of 'sender'."""
        conv = {
            "uuid": "c1",
            "name": "Role Test",
            "created_at": "2024-01-01T10:00:00.000000Z",
            "updated_at": "2024-01-01T10:01:00.000000Z",
            "chat_messages": [
                {
                    "uuid": "m1",
                    "role": "user",
                    "text": "Using role instead of sender.",
                    "created_at": "2024-01-01T10:00:30.000000Z",
                },
                {
                    "uuid": "m2",
                    "role": "assistant",
                    "text": "Understood.",
                    "created_at": "2024-01-01T10:01:00.000000Z",
                },
            ],
        }
        p = tmp_path / "export.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        roles = {e.role for e in profile.conversations[0].entries}
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_messages_field_instead_of_chat_messages(self, tmp_path: Path):
        """Some exports use 'messages' instead of 'chat_messages'."""
        conv = {
            "uuid": "c1",
            "name": "Messages Test",
            "created_at": "2024-01-01T10:00:00.000000Z",
            "updated_at": "2024-01-01T10:01:00.000000Z",
            "messages": [
                {
                    "uuid": "m1",
                    "sender": "human",
                    "text": "Hello via messages field.",
                    "created_at": "2024-01-01T10:00:30.000000Z",
                }
            ],
        }
        p = tmp_path / "export.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversations[0].message_count == 1


class TestClaudeAdapterEdgeCases:
    """Edge-case and error-handling tests for the Claude adapter."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = ClaudeAdapter()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.adapter.read("/nonexistent/file.json")

    def test_invalid_json_raises_parse_error(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ClaudeParseError):
            self.adapter.read(p)

    def test_unexpected_top_level_type_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text('"just a string"', encoding="utf-8")
        with pytest.raises(ClaudeParseError):
            self.adapter.read(p)

    def test_empty_conversations_array(self, tmp_path: Path):
        p = tmp_path / "export.json"
        p.write_text("[]", encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversation_count == 0

    def test_malformed_conversations_skipped(self, tmp_path: Path):
        data = [
            None,  # invalid
            make_minimal_claude_conversation(conv_id="good_conv"),
        ]
        p = tmp_path / "export.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profile = self.adapter.read(p)
        # Only the valid conversation should be parsed
        assert profile.conversation_count == 1

    def test_missing_title_uses_default(self, tmp_path: Path):
        conv = {
            "uuid": "c1",
            "created_at": "2024-01-01T10:00:00.000000Z",
            "updated_at": "2024-01-01T10:01:00.000000Z",
            "chat_messages": [],
        }
        p = tmp_path / "export.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversations[0].title == "Untitled Conversation"

    def test_memory_with_content_field(self, tmp_path: Path):
        data = {
            "conversations": [],
            "memories": [{"content": "User is a developer."}],
        }
        p = tmp_path / "export.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memory_count == 1
        assert profile.memories[0].content == "User is a developer."

    def test_memory_ids_extracted(self, tmp_path: Path):
        data = {
            "conversations": [],
            "memories": [
                {"uuid": "mem_001", "text": "User prefers Python."},
            ],
        }
        p = tmp_path / "export.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.memories[0].id == "mem_001"

    def test_display_name_falls_back_to_claude_user(self, tmp_path: Path):
        """When no name info is present, display_name should be 'Claude User'."""
        convs = [
            {
                "uuid": "c1",
                "name": "Test",
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:01:00Z",
                "account": {"uuid": "user_no_name"},
                "chat_messages": [],
            }
        ]
        p = tmp_path / "export.json"
        p.write_text(json.dumps(convs), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.display_name == "Claude User"

    def test_iso_timestamp_with_microseconds(self, tmp_path: Path):
        conv = make_minimal_claude_conversation()
        p = tmp_path / "export.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        assert profile.conversations[0].created_at is not None
        assert profile.conversations[0].created_at.year == 2024

    def test_total_message_count_across_fixture(self):
        """Fixture has 3 conversations with 4+4+2=10 messages total."""
        profile = self.adapter.read(CLAUDE_FIXTURE)
        assert profile.total_message_count == 10

    def test_message_uuids_extracted(self, tmp_path: Path):
        conv = make_minimal_claude_conversation()
        p = tmp_path / "export.json"
        p.write_text(json.dumps([conv]), encoding="utf-8")
        profile = self.adapter.read(p)
        for entry in profile.conversations[0].entries:
            assert entry.id is not None

    def test_profile_is_serialisable(self, tmp_path: Path):
        """Parsed profile should round-trip through JSON without errors."""
        profile = self.adapter.read(CLAUDE_FIXTURE)
        json_str = profile.to_json()
        restored = MemoryProfile.from_json(json_str)
        assert restored.conversation_count == profile.conversation_count
        assert restored.source_platform == SourcePlatform.CLAUDE


# ===========================================================================
# Cross-adapter tests
# ===========================================================================


class TestAdapterRegistry:
    """Verify the adapter registry correctly routes to ChatGPT and Claude adapters."""

    def test_get_chatgpt_reader(self):
        from mem_bridge.adapters import get_adapter

        adapter = get_adapter("chatgpt", mode="read")
        assert isinstance(adapter, ChatGPTAdapter)

    def test_get_claude_reader(self):
        from mem_bridge.adapters import get_adapter

        adapter = get_adapter("claude", mode="read")
        assert isinstance(adapter, ClaudeAdapter)

    def test_get_unknown_platform_raises(self):
        from mem_bridge.adapters import AdapterNotFoundError, get_adapter

        with pytest.raises(AdapterNotFoundError):
            get_adapter("nonexistent_platform", mode="read")

    def test_list_adapters_contains_readers(self):
        from mem_bridge.adapters import list_adapters

        info = list_adapters()
        assert "chatgpt" in info["readers"]
        assert "claude" in info["readers"]

    def test_list_adapters_contains_writers(self):
        from mem_bridge.adapters import list_adapters

        info = list_adapters()
        assert "gemini" in info["writers"]
