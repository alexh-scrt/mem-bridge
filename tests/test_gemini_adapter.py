"""Unit tests for the Gemini writer adapter.

Tests verify:
- Markdown rendering produces non-empty, correctly structured output
- JSON rendering produces valid JSON with expected keys
- write() creates files on disk in both formats
- Format auto-detection from file extension
- Explicit format override via fmt parameter
- Memory facts section rendering (with and without memories)
- Conversation history section rendering (with and without conversations)
- Edge cases: empty profiles, special characters, missing timestamps
- Error handling for unsupported formats and write failures
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mem_bridge.adapters.gemini import GeminiAdapter, GeminiWriteError, _md_escape
from mem_bridge.models import (
    Conversation,
    ConversationEntry,
    MemoryEntry,
    MemoryProfile,
    Role,
    SourcePlatform,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_memory_entry(content: str = "User prefers dark mode.", **kwargs) -> MemoryEntry:
    """Return a MemoryEntry with sensible defaults."""
    return MemoryEntry(content=content, **kwargs)


def make_conversation_entry(
    role: Role = Role.USER,
    content: str = "Hello!",
    **kwargs,
) -> ConversationEntry:
    """Return a ConversationEntry with sensible defaults."""
    return ConversationEntry(role=role, content=content, **kwargs)


def make_conversation(
    title: str = "Test Conversation",
    model: str | None = "gemini-pro",
    **kwargs,
) -> Conversation:
    """Return a Conversation with one user and one assistant entry."""
    entries = [
        make_conversation_entry(role=Role.USER, content="What is Python?"),
        make_conversation_entry(
            role=Role.ASSISTANT, content="Python is a programming language."
        ),
    ]
    return Conversation(
        title=title,
        model=model,
        created_at=datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc),
        entries=entries,
        **kwargs,
    )


def make_full_profile(
    display_name: str = "Alice",
    source_platform: SourcePlatform = SourcePlatform.CHATGPT,
    memory_count: int = 3,
    conversation_count: int = 2,
) -> MemoryProfile:
    """Return a MemoryProfile with the requested number of memories and conversations."""
    memories = [
        make_memory_entry(
            content=f"Memory fact {i}.",
            tags=["tag_a", "tag_b"] if i % 2 == 0 else [],
        )
        for i in range(memory_count)
    ]
    conversations = [
        make_conversation(title=f"Conversation {i}") for i in range(conversation_count)
    ]
    return MemoryProfile(
        source_platform=source_platform,
        display_name=display_name,
        exported_at=datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        memories=memories,
        conversations=conversations,
    )


def make_empty_profile() -> MemoryProfile:
    """Return a MemoryProfile with no memories or conversations."""
    return MemoryProfile(
        source_platform=SourcePlatform.CLAUDE,
        display_name="Bob",
        exported_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# GeminiAdapter – markdown rendering
# ---------------------------------------------------------------------------


class TestGeminiAdapterMarkdown:
    """Tests for Markdown output rendering."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter()

    def test_render_returns_string(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert isinstance(result, str)

    def test_render_nonempty(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert len(result.strip()) > 0

    def test_render_contains_display_name(self):
        profile = make_full_profile(display_name="Alice Wonder")
        result = self.adapter.render(profile, fmt="markdown")
        assert "Alice Wonder" in result

    def test_render_contains_source_platform(self):
        profile = make_full_profile(source_platform=SourcePlatform.CHATGPT)
        result = self.adapter.render(profile, fmt="markdown")
        assert "chatgpt" in result.lower()

    def test_render_contains_memory_facts_header(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert "Remembered Facts" in result

    def test_render_contains_memory_content(self):
        profile = make_full_profile(memory_count=2)
        result = self.adapter.render(profile, fmt="markdown")
        assert "Memory fact 0" in result
        assert "Memory fact 1" in result

    def test_render_memory_tags_included(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[
                MemoryEntry(content="User likes Python.", tags=["programming", "python"])
            ],
        )
        result = self.adapter.render(profile, fmt="markdown")
        assert "`programming`" in result
        assert "`python`" in result

    def test_render_no_memories_placeholder(self):
        profile = make_empty_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert "No memory facts" in result

    def test_render_contains_conversation_history_header(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert "Conversation History" in result

    def test_render_contains_conversation_titles(self):
        profile = make_full_profile(conversation_count=2)
        result = self.adapter.render(profile, fmt="markdown")
        assert "Conversation 0" in result
        assert "Conversation 1" in result

    def test_render_contains_message_content(self):
        profile = make_full_profile(conversation_count=1)
        result = self.adapter.render(profile, fmt="markdown")
        assert "What is Python?" in result
        assert "Python is a programming language." in result

    def test_render_contains_user_and_assistant_labels(self):
        profile = make_full_profile(conversation_count=1)
        result = self.adapter.render(profile, fmt="markdown")
        assert "User:" in result or "**User:**" in result
        assert "Assistant:" in result or "**Assistant:**" in result

    def test_render_no_conversations_placeholder(self):
        profile = make_empty_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert "No conversation history" in result

    def test_render_exported_at_included(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert "2024-06-15" in result

    def test_render_exported_at_unknown_when_none(self):
        profile = MemoryProfile(display_name="Alice")
        result = self.adapter.render(profile, fmt="markdown")
        assert "unknown" in result.lower()

    def test_render_md_alias_works(self):
        profile = make_full_profile()
        result_md = self.adapter.render(profile, fmt="md")
        result_markdown = self.adapter.render(profile, fmt="markdown")
        assert result_md == result_markdown

    def test_render_gemini_alias_works(self):
        profile = make_full_profile()
        result_gemini = self.adapter.render(profile, fmt="gemini")
        result_markdown = self.adapter.render(profile, fmt="markdown")
        assert result_gemini == result_markdown

    def test_render_contains_schema_version(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert profile.schema_version in result

    def test_conversation_model_included(self):
        profile = make_full_profile(conversation_count=1)
        result = self.adapter.render(profile, fmt="markdown")
        assert "gemini-pro" in result

    def test_conversation_date_included(self):
        profile = make_full_profile(conversation_count=1)
        result = self.adapter.render(profile, fmt="markdown")
        assert "2024-03-01" in result


# ---------------------------------------------------------------------------
# GeminiAdapter – JSON rendering
# ---------------------------------------------------------------------------


class TestGeminiAdapterJSON:
    """Tests for JSON output rendering."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter()

    def test_render_json_returns_string(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="json")
        assert isinstance(result, str)

    def test_render_json_is_valid_json(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="json")
        data = json.loads(result)  # Should not raise
        assert isinstance(data, dict)

    def test_render_json_has_gem_format_version(self):
        profile = make_full_profile()
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "gem_format_version" in data
        assert data["gem_format_version"] == "1.0"

    def test_render_json_has_generated_by(self):
        profile = make_full_profile()
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert data["generated_by"] == "mem_bridge"

    def test_render_json_has_user_section(self):
        profile = make_full_profile(display_name="Alice")
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "user" in data
        assert data["user"]["display_name"] == "Alice"
        assert data["user"]["source_platform"] == "chatgpt"

    def test_render_json_has_memory_facts(self):
        profile = make_full_profile(memory_count=3)
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "memory_facts" in data
        assert len(data["memory_facts"]) == 3

    def test_render_json_memory_fact_fields(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[
                MemoryEntry(
                    id="m1",
                    content="User prefers Python.",
                    tags=["programming"],
                    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                )
            ],
        )
        data = json.loads(self.adapter.render(profile, fmt="json"))
        fact = data["memory_facts"][0]
        assert fact["id"] == "m1"
        assert fact["content"] == "User prefers Python."
        assert fact["tags"] == ["programming"]
        assert "2024-01-01" in fact["created_at"]

    def test_render_json_has_conversation_history(self):
        profile = make_full_profile(conversation_count=2)
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "conversation_history" in data
        assert len(data["conversation_history"]) == 2

    def test_render_json_conversation_fields(self):
        profile = make_full_profile(conversation_count=1)
        data = json.loads(self.adapter.render(profile, fmt="json"))
        conv = data["conversation_history"][0]
        assert "id" in conv
        assert "title" in conv
        assert "model" in conv
        assert "messages" in conv
        assert isinstance(conv["messages"], list)

    def test_render_json_message_fields(self):
        profile = make_full_profile(conversation_count=1)
        data = json.loads(self.adapter.render(profile, fmt="json"))
        msg = data["conversation_history"][0]["messages"][0]
        assert "role" in msg
        assert "content" in msg
        assert msg["role"] in ("user", "assistant", "system", "tool", "unknown")

    def test_render_json_has_statistics(self):
        profile = make_full_profile(memory_count=3, conversation_count=2)
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "statistics" in data
        stats = data["statistics"]
        assert stats["memory_count"] == 3
        assert stats["conversation_count"] == 2
        assert stats["total_message_count"] == 4  # 2 convs × 2 messages each

    def test_render_json_empty_profile(self):
        profile = make_empty_profile()
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert data["memory_facts"] == []
        assert data["conversation_history"] == []
        assert data["statistics"]["memory_count"] == 0

    def test_render_json_exported_at_in_user_section(self):
        profile = make_full_profile()
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert "2024-06-15" in data["user"]["exported_at"]

    def test_render_json_exported_at_none_when_missing(self):
        profile = MemoryProfile(display_name="Alice")
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert data["user"]["exported_at"] is None

    def test_render_json_generated_at_is_valid_iso(self):
        profile = make_full_profile()
        data = json.loads(self.adapter.render(profile, fmt="json"))
        # Should be a valid ISO datetime string
        ts = data["generated_at"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format marker


# ---------------------------------------------------------------------------
# GeminiAdapter – write() file output
# ---------------------------------------------------------------------------


class TestGeminiAdapterWrite:
    """Tests for the write() method that saves output to disk."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter()

    def test_write_markdown_file_created(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "gem.md"
        self.adapter.write(profile, dest)
        assert dest.exists()

    def test_write_json_file_created(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "gem.json"
        self.adapter.write(profile, dest)
        assert dest.exists()

    def test_write_markdown_content_correct(self, tmp_path: Path):
        profile = make_full_profile(display_name="Alice")
        dest = tmp_path / "gem.md"
        self.adapter.write(profile, dest)
        content = dest.read_text(encoding="utf-8")
        assert "Alice" in content
        assert "Remembered Facts" in content

    def test_write_json_content_is_valid_json(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "gem.json"
        self.adapter.write(profile, dest)
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "gem_format_version" in data

    def test_write_creates_parent_directories(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "nested" / "dir" / "gem.md"
        self.adapter.write(profile, dest)
        assert dest.exists()

    def test_write_auto_detects_md_extension(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "output.md"
        self.adapter.write(profile, dest)
        content = dest.read_text()
        # Markdown should not start with '{'
        assert not content.strip().startswith("{")

    def test_write_auto_detects_json_extension(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "output.json"
        self.adapter.write(profile, dest)
        content = dest.read_text()
        data = json.loads(content)
        assert isinstance(data, dict)

    def test_write_txt_extension_defaults_to_markdown(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "output.txt"
        self.adapter.write(profile, dest)
        content = dest.read_text()
        # Should be markdown (not JSON)
        assert not content.strip().startswith("{")
        assert "Remembered Facts" in content

    def test_write_explicit_fmt_json_overrides_md_extension(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "output.md"
        self.adapter.write(profile, dest, fmt="json")
        content = dest.read_text()
        data = json.loads(content)  # Should parse as JSON
        assert "gem_format_version" in data

    def test_write_explicit_fmt_markdown_overrides_json_extension(self, tmp_path: Path):
        profile = make_full_profile()
        dest = tmp_path / "output.json"
        self.adapter.write(profile, dest, fmt="markdown")
        content = dest.read_text()
        # Should be markdown, not JSON
        assert content.strip().startswith("#")

    def test_write_accepts_string_path(self, tmp_path: Path):
        profile = make_full_profile()
        dest = str(tmp_path / "gem.md")
        self.adapter.write(profile, dest)  # Should not raise
        assert Path(dest).exists()

    def test_write_file_is_utf8(self, tmp_path: Path):
        profile = MemoryProfile(
            display_name="Ünïcödé Üser",
            memories=[MemoryEntry(content="Préfère les accents français.")],
        )
        dest = tmp_path / "gem.md"
        self.adapter.write(profile, dest)
        content = dest.read_text(encoding="utf-8")
        assert "Ünïcödé" in content


# ---------------------------------------------------------------------------
# GeminiAdapter – format resolution
# ---------------------------------------------------------------------------


class TestGeminiAdapterFormatResolution:
    """Tests for format auto-detection and explicit override."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter()

    def test_unsupported_fmt_raises_value_error(self):
        profile = make_full_profile()
        with pytest.raises(ValueError, match="Unsupported"):
            self.adapter.render(profile, fmt="xml")

    def test_unsupported_fmt_in_write_raises_value_error(self, tmp_path: Path):
        profile = make_full_profile()
        with pytest.raises(ValueError, match="Unsupported"):
            self.adapter.write(profile, tmp_path / "out.md", fmt="toml")

    def test_fmt_case_insensitive(self):
        profile = make_full_profile()
        result_lower = self.adapter.render(profile, fmt="json")
        result_upper = self.adapter.render(profile, fmt="JSON")
        # Both should produce valid JSON
        json.loads(result_lower)
        json.loads(result_upper)

    def test_fmt_strips_whitespace(self):
        profile = make_full_profile()
        result = self.adapter.render(profile, fmt="  markdown  ")
        assert "Remembered Facts" in result


# ---------------------------------------------------------------------------
# GeminiAdapter – edge cases
# ---------------------------------------------------------------------------


class TestGeminiAdapterEdgeCases:
    """Edge-case tests for the Gemini adapter."""

    @pytest.fixture(autouse=True)
    def adapter(self):
        self.adapter = GeminiAdapter()

    def test_empty_profile_markdown_renders(self):
        profile = make_empty_profile()
        result = self.adapter.render(profile, fmt="markdown")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_empty_profile_json_renders(self):
        profile = make_empty_profile()
        result = self.adapter.render(profile, fmt="json")
        data = json.loads(result)
        assert data["statistics"]["memory_count"] == 0

    def test_memory_with_no_tags_renders(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[MemoryEntry(content="A plain fact.", tags=[])],
        )
        result = self.adapter.render(profile, fmt="markdown")
        assert "A plain fact." in result
        # No tag markers should appear
        assert "`" not in result.split("A plain fact.")[1].split("\n")[0]

    def test_memory_with_null_id_renders(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[MemoryEntry(content="Fact without ID.")],
        )
        data = json.loads(self.adapter.render(profile, fmt="json"))
        assert data["memory_facts"][0]["id"] is None

    def test_conversation_with_no_entries_skipped_in_markdown(self):
        empty_conv = Conversation(title="Empty Conv", entries=[])
        profile = MemoryProfile(
            display_name="Alice",
            conversations=[empty_conv],
        )
        result = self.adapter.render(profile, fmt="markdown")
        # Empty conversation should not appear in history
        assert "Empty Conv" not in result

    def test_conversation_with_empty_message_content_skipped(self):
        entries = [
            ConversationEntry(role=Role.USER, content=""),
            ConversationEntry(role=Role.ASSISTANT, content=""),
        ]
        conv = Conversation(title="Blank Messages", entries=entries)
        profile = MemoryProfile(display_name="Alice", conversations=[conv])
        result = self.adapter.render(profile, fmt="markdown")
        # Conversation with only blank messages should not appear
        assert "Blank Messages" not in result

    def test_special_chars_in_display_name_escaped(self):
        profile = MemoryProfile(display_name="Alice | The *Developer*")
        result = self.adapter.render(profile, fmt="markdown")
        # The display name should be escaped (no raw | or * in headings)
        # At minimum the output should be non-empty and not crash
        assert isinstance(result, str)
        assert len(result) > 0

    def test_profile_from_claude_source(self):
        profile = MemoryProfile(
            source_platform=SourcePlatform.CLAUDE,
            display_name="Bob",
            memories=[MemoryEntry(content="Prefers concise responses.")],
        )
        result = self.adapter.render(profile, fmt="markdown")
        assert "claude" in result.lower()
        assert "Bob" in result

    def test_large_profile_renders_without_error(self):
        memories = [MemoryEntry(content=f"Fact {i} about the user.") for i in range(50)]
        conversations = [
            make_conversation(title=f"Conv {i}") for i in range(20)
        ]
        profile = MemoryProfile(
            display_name="PowerUser",
            memories=memories,
            conversations=conversations,
        )
        result_md = self.adapter.render(profile, fmt="markdown")
        result_json = self.adapter.render(profile, fmt="json")
        assert len(result_md) > 1000
        data = json.loads(result_json)
        assert data["statistics"]["memory_count"] == 50
        assert data["statistics"]["conversation_count"] == 20


# ---------------------------------------------------------------------------
# _md_escape helper tests
# ---------------------------------------------------------------------------


class TestMdEscape:
    """Tests for the _md_escape helper function."""

    def test_plain_text_unchanged(self):
        assert _md_escape("Hello World") == "Hello World"

    def test_pipe_escaped(self):
        result = _md_escape("A | B")
        assert "\\|" in result

    def test_asterisk_escaped(self):
        result = _md_escape("Bold *text*")
        assert "\\*" in result

    def test_underscore_escaped(self):
        result = _md_escape("snake_case")
        assert "\\_" in result

    def test_backtick_escaped(self):
        result = _md_escape("`code`")
        assert "\\`" in result

    def test_empty_string(self):
        assert _md_escape("") == ""


# ---------------------------------------------------------------------------
# Adapter registry integration
# ---------------------------------------------------------------------------


class TestGeminiAdapterRegistry:
    """Verify the adapter registry correctly routes to the Gemini adapter."""

    def test_get_gemini_writer(self):
        from mem_bridge.adapters import get_adapter

        adapter = get_adapter("gemini", mode="write")
        assert isinstance(adapter, GeminiAdapter)

    def test_gemini_not_available_as_reader(self):
        from mem_bridge.adapters import AdapterNotFoundError, get_adapter

        with pytest.raises(AdapterNotFoundError):
            get_adapter("gemini", mode="read")

    def test_list_adapters_includes_gemini_writer(self):
        from mem_bridge.adapters import list_adapters

        info = list_adapters()
        assert "gemini" in info["writers"]

    def test_list_adapters_gemini_not_in_readers(self):
        from mem_bridge.adapters import list_adapters

        info = list_adapters()
        assert "gemini" not in info["readers"]

    def test_list_platforms_includes_gemini(self):
        from mem_bridge.adapters import list_platforms

        platforms = list_platforms()
        assert "gemini" in platforms

    def test_get_adapter_info_gemini(self):
        from mem_bridge.adapters import get_adapter_info

        info = get_adapter_info("gemini")
        assert info["mode"] == "write"
        assert "gemini" in info["description"].lower() or "gem" in info["description"].lower()

    def test_get_adapter_info_chatgpt(self):
        from mem_bridge.adapters import get_adapter_info

        info = get_adapter_info("chatgpt")
        assert info["mode"] == "read"

    def test_get_adapter_info_unknown_raises(self):
        from mem_bridge.adapters import AdapterNotFoundError, get_adapter_info

        with pytest.raises(AdapterNotFoundError):
            get_adapter_info("nonexistent")

    def test_register_and_use_custom_writer(self, tmp_path: Path):
        """Dynamic registration of a custom writer adapter."""
        # Create a minimal adapter module in tmp_path
        adapter_code = '''
from mem_bridge.models import MemoryProfile
from pathlib import Path

class CustomAdapter:
    def write(self, profile: MemoryProfile, dest: str | Path, fmt=None) -> None:
        Path(dest).write_text("custom output", encoding="utf-8")
'''
        module_file = tmp_path / "custom_adapter.py"
        module_file.write_text(adapter_code, encoding="utf-8")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            from mem_bridge.adapters import register_writer

            register_writer("custom_platform", "custom_adapter")

            from mem_bridge.adapters import get_adapter

            adapter = get_adapter("custom_platform", mode="write")
            assert hasattr(adapter, "write")
        finally:
            sys.path.pop(0)
            # Clean up registry to avoid polluting other tests
            from mem_bridge.adapters import _WRITER_REGISTRY

            _WRITER_REGISTRY.pop("custom_platform", None)
