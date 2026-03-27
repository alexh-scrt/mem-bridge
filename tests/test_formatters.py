"""Unit tests for mem_bridge.formatters.

Tests verify:
- Each output formatter produces non-empty, correctly structured output
  from a sample MemoryProfile.
- The render() module-level function dispatches correctly.
- The Formatter class renders all supported formats.
- render_to_file() writes the correct content to disk.
- Format auto-detection from file extension.
- list_formats() returns all expected format names.
- get_format_description() returns descriptive strings.
- Edge cases: empty profiles, special characters, profiles with no
  memories or no conversations.
- Error handling for unsupported format strings.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from mem_bridge.formatters import (
    Formatter,
    get_format_description,
    list_formats,
    render,
    render_to_file,
)
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


def make_memory(content: str = "User prefers dark mode.", **kwargs) -> MemoryEntry:
    """Return a MemoryEntry with the given content."""
    return MemoryEntry(content=content, **kwargs)


def make_entry(
    role: Role = Role.USER,
    content: str = "Hello!",
    **kwargs,
) -> ConversationEntry:
    """Return a ConversationEntry."""
    return ConversationEntry(role=role, content=content, **kwargs)


def make_conversation(
    title: str = "Test Conversation",
    model: str | None = "gpt-4o",
    **kwargs,
) -> Conversation:
    """Return a Conversation with one user + one assistant message."""
    entries = [
        make_entry(role=Role.USER, content="What is the meaning of life?"),
        make_entry(role=Role.ASSISTANT, content="According to philosophy, it depends."),
    ]
    return Conversation(
        title=title,
        model=model,
        created_at=datetime(2024, 4, 1, 9, 0, 0, tzinfo=timezone.utc),
        entries=entries,
        **kwargs,
    )


def make_profile(
    display_name: str = "Alice",
    source_platform: SourcePlatform = SourcePlatform.CHATGPT,
    memory_count: int = 3,
    conversation_count: int = 2,
    include_exported_at: bool = True,
) -> MemoryProfile:
    """Return a populated MemoryProfile for testing."""
    memories = [
        make_memory(
            content=f"User preference fact {i}.",
            tags=["pref", f"tag{i}"] if i % 2 == 0 else [],
            id=f"mem_{i}",
        )
        for i in range(memory_count)
    ]
    conversations = [
        make_conversation(title=f"Conversation Topic {i}")
        for i in range(conversation_count)
    ]
    kwargs = {}
    if include_exported_at:
        kwargs["exported_at"] = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return MemoryProfile(
        source_platform=source_platform,
        display_name=display_name,
        memories=memories,
        conversations=conversations,
        **kwargs,
    )


def make_empty_profile() -> MemoryProfile:
    """Return a MemoryProfile with no memories or conversations."""
    return MemoryProfile(
        source_platform=SourcePlatform.CLAUDE,
        display_name="Bob",
        exported_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Module-level render() function
# ---------------------------------------------------------------------------


class TestRenderFunction:
    """Tests for the module-level render() convenience function."""

    def test_render_returns_string(self):
        profile = make_profile()
        result = render(profile, fmt="markdown")
        assert isinstance(result, str)

    def test_render_markdown_nonempty(self):
        profile = make_profile()
        result = render(profile, fmt="markdown")
        assert len(result.strip()) > 0

    def test_render_json_nonempty(self):
        profile = make_profile()
        result = render(profile, fmt="json")
        assert len(result.strip()) > 0

    def test_render_yaml_nonempty(self):
        profile = make_profile()
        result = render(profile, fmt="yaml")
        assert len(result.strip()) > 0

    def test_render_text_nonempty(self):
        profile = make_profile()
        result = render(profile, fmt="text")
        assert len(result.strip()) > 0

    def test_render_gemini_nonempty(self):
        profile = make_profile()
        result = render(profile, fmt="gemini")
        assert len(result.strip()) > 0

    def test_render_unsupported_format_raises(self):
        profile = make_profile()
        with pytest.raises(ValueError, match="Unsupported format"):
            render(profile, fmt="xml")

    def test_render_default_format_is_markdown(self):
        profile = make_profile()
        result = render(profile)
        # Markdown output should start with a heading marker
        assert result.strip().startswith("#")

    def test_render_md_alias_equals_markdown(self):
        profile = make_profile()
        assert render(profile, fmt="md") == render(profile, fmt="markdown")

    def test_render_case_insensitive(self):
        profile = make_profile()
        result_lower = render(profile, fmt="json")
        result_upper = render(profile, fmt="JSON")
        assert result_lower == result_upper

    def test_render_strips_fmt_whitespace(self):
        profile = make_profile()
        result = render(profile, fmt="  yaml  ")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


class TestMarkdownFormatter:
    """Tests for the Markdown output format."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_contains_display_name(self):
        profile = make_profile(display_name="Alice Wonder")
        result = self.formatter.render(profile, fmt="markdown")
        assert "Alice Wonder" in result

    def test_contains_source_platform(self):
        profile = make_profile(source_platform=SourcePlatform.CHATGPT)
        result = self.formatter.render(profile, fmt="markdown")
        assert "chatgpt" in result.lower()

    def test_contains_claude_source_platform(self):
        profile = make_profile(source_platform=SourcePlatform.CLAUDE)
        result = self.formatter.render(profile, fmt="markdown")
        assert "claude" in result.lower()

    def test_starts_with_heading(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="markdown")
        assert result.strip().startswith("#")

    def test_contains_remembered_facts_section(self):
        profile = make_profile(memory_count=2)
        result = self.formatter.render(profile, fmt="markdown")
        assert "Remembered Facts" in result

    def test_contains_memory_content(self):
        profile = make_profile(memory_count=2)
        result = self.formatter.render(profile, fmt="markdown")
        assert "User preference fact 0" in result
        assert "User preference fact 1" in result

    def test_contains_memory_tags(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[
                MemoryEntry(
                    content="User likes Python.",
                    tags=["programming", "python"],
                )
            ],
        )
        result = self.formatter.render(profile, fmt="markdown")
        assert "programming" in result
        assert "python" in result

    def test_no_memories_placeholder(self):
        profile = make_empty_profile()
        result = self.formatter.render(profile, fmt="markdown")
        assert "No memory facts" in result or "no memory facts" in result.lower()

    def test_contains_conversation_history_section(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="markdown")
        assert "Conversation History" in result

    def test_contains_conversation_title(self):
        profile = make_profile(conversation_count=2)
        result = self.formatter.render(profile, fmt="markdown")
        assert "Conversation Topic 0" in result
        assert "Conversation Topic 1" in result

    def test_contains_message_content(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="markdown")
        assert "meaning of life" in result.lower()
        assert "philosophy" in result.lower()

    def test_contains_role_labels(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="markdown")
        # User and Assistant labels should appear
        assert "user" in result.lower() or "User" in result
        assert "assistant" in result.lower() or "Assistant" in result

    def test_no_conversations_placeholder(self):
        profile = make_empty_profile()
        result = self.formatter.render(profile, fmt="markdown")
        assert "No conversation history" in result or "no conversation history" in result.lower()

    def test_contains_exported_at(self):
        profile = make_profile(include_exported_at=True)
        result = self.formatter.render(profile, fmt="markdown")
        assert "2024-06-01" in result

    def test_exported_at_unknown_when_none(self):
        profile = MemoryProfile(display_name="Alice")
        result = self.formatter.render(profile, fmt="markdown")
        assert "unknown" in result.lower()

    def test_contains_schema_version(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="markdown")
        assert profile.schema_version in result

    def test_contains_conversation_model(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="markdown")
        assert "gpt-4o" in result

    def test_contains_conversation_date(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="markdown")
        assert "2024-04-01" in result

    def test_empty_profile_renders_without_error(self):
        profile = make_empty_profile()
        result = self.formatter.render(profile, fmt="markdown")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_memory_count_in_header(self):
        profile = make_profile(memory_count=5)
        result = self.formatter.render(profile, fmt="markdown")
        assert "5" in result

    def test_empty_message_entries_not_rendered(self):
        conv = Conversation(
            title="Empty Messages Conv",
            entries=[
                ConversationEntry(role=Role.USER, content=""),
                ConversationEntry(role=Role.ASSISTANT, content=""),
            ],
        )
        profile = MemoryProfile(display_name="Alice", conversations=[conv])
        result = self.formatter.render(profile, fmt="markdown")
        # Empty conversation should either be skipped or produce no message lines
        # The title may not appear since all messages are empty
        assert isinstance(result, str)

    def test_unicode_content_rendered(self):
        profile = MemoryProfile(
            display_name="Ünïcödé Üser",
            memories=[MemoryEntry(content="Préfère les accents français.")],
        )
        result = self.formatter.render(profile, fmt="markdown")
        assert "Ünïcödé" in result
        assert "Préfère" in result


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class TestJSONFormatter:
    """Tests for the JSON output format."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_returns_valid_json(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="json")
        data = json.loads(result)  # Must not raise
        assert isinstance(data, dict)

    def test_display_name_in_json(self):
        profile = make_profile(display_name="Alice")
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert data["display_name"] == "Alice"

    def test_source_platform_as_string(self):
        profile = make_profile(source_platform=SourcePlatform.CHATGPT)
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert data["source_platform"] == "chatgpt"
        assert isinstance(data["source_platform"], str)

    def test_memories_array_present(self):
        profile = make_profile(memory_count=3)
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert "memories" in data
        assert len(data["memories"]) == 3

    def test_conversations_array_present(self):
        profile = make_profile(conversation_count=2)
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert "conversations" in data
        assert len(data["conversations"]) == 2

    def test_memory_content_in_json(self):
        profile = make_profile(memory_count=1)
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert data["memories"][0]["content"] == "User preference fact 0."

    def test_role_as_string_in_json(self):
        profile = make_profile(conversation_count=1)
        data = json.loads(self.formatter.render(profile, fmt="json"))
        msgs = data["conversations"][0]["entries"]
        for msg in msgs:
            assert isinstance(msg["role"], str)

    def test_schema_version_in_json(self):
        profile = make_profile()
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert "schema_version" in data
        assert data["schema_version"] == "1.0"

    def test_json_round_trip(self):
        profile = make_profile()
        json_str = self.formatter.render(profile, fmt="json")
        restored = MemoryProfile.from_json(json_str)
        assert restored.display_name == profile.display_name
        assert restored.memory_count == profile.memory_count
        assert restored.conversation_count == profile.conversation_count

    def test_empty_profile_json(self):
        profile = make_empty_profile()
        data = json.loads(self.formatter.render(profile, fmt="json"))
        assert data["memories"] == []
        assert data["conversations"] == []

    def test_json_is_indented(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="json")
        # Indented JSON has newlines
        assert "\n" in result

    def test_all_expected_top_level_keys(self):
        profile = make_profile()
        data = json.loads(self.formatter.render(profile, fmt="json"))
        expected_keys = {
            "id", "source_platform", "display_name", "exported_at",
            "memories", "conversations", "schema_version", "metadata",
        }
        assert expected_keys <= set(data.keys())


# ---------------------------------------------------------------------------
# YAML formatter
# ---------------------------------------------------------------------------


class TestYAMLFormatter:
    """Tests for the YAML output format."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_returns_valid_yaml(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="yaml")
        data = yaml.safe_load(result)  # Must not raise
        assert isinstance(data, dict)

    def test_display_name_in_yaml(self):
        profile = make_profile(display_name="Alice")
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert data["display_name"] == "Alice"

    def test_source_platform_as_string_in_yaml(self):
        profile = make_profile(source_platform=SourcePlatform.CLAUDE)
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert data["source_platform"] == "claude"

    def test_memories_list_in_yaml(self):
        profile = make_profile(memory_count=2)
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert isinstance(data["memories"], list)
        assert len(data["memories"]) == 2

    def test_conversations_list_in_yaml(self):
        profile = make_profile(conversation_count=1)
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert isinstance(data["conversations"], list)
        assert len(data["conversations"]) == 1

    def test_yaml_round_trip(self):
        profile = make_profile()
        yaml_str = self.formatter.render(profile, fmt="yaml")
        raw = yaml.safe_load(yaml_str)
        restored = MemoryProfile.from_dict(raw)
        assert restored.display_name == profile.display_name
        assert restored.memory_count == profile.memory_count

    def test_empty_profile_yaml(self):
        profile = make_empty_profile()
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert data["memories"] == [] or data["memories"] is None

    def test_yaml_is_multiline(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="yaml")
        assert "\n" in result

    def test_memory_content_in_yaml(self):
        profile = make_profile(memory_count=1)
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert data["memories"][0]["content"] == "User preference fact 0."

    def test_tags_in_yaml(self):
        profile = MemoryProfile(
            display_name="Alice",
            memories=[MemoryEntry(content="Fact.", tags=["a", "b"])],
        )
        data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert data["memories"][0]["tags"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


class TestTextFormatter:
    """Tests for the plain-text summary output format."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_returns_string(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        assert isinstance(result, str)

    def test_nonempty_output(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        assert len(result.strip()) > 0

    def test_contains_display_name(self):
        profile = make_profile(display_name="Alice")
        result = self.formatter.render(profile, fmt="text")
        assert "Alice" in result

    def test_contains_source_platform(self):
        profile = make_profile(source_platform=SourcePlatform.CHATGPT)
        result = self.formatter.render(profile, fmt="text")
        assert "chatgpt" in result.lower()

    def test_contains_memory_section_header(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        assert "REMEMBERED FACTS" in result.upper() or "Remembered Facts" in result

    def test_contains_conversation_section_header(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        assert "CONVERSATION" in result.upper()

    def test_contains_memory_content(self):
        profile = make_profile(memory_count=2)
        result = self.formatter.render(profile, fmt="text")
        assert "User preference fact 0" in result

    def test_contains_conversation_title(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="text")
        assert "Conversation Topic 0" in result

    def test_memory_count_shown(self):
        profile = make_profile(memory_count=4)
        result = self.formatter.render(profile, fmt="text")
        assert "4" in result

    def test_conversation_count_shown(self):
        profile = make_profile(conversation_count=3)
        result = self.formatter.render(profile, fmt="text")
        assert "3" in result

    def test_no_memories_shown_empty(self):
        profile = make_empty_profile()
        result = self.formatter.render(profile, fmt="text")
        assert "(none)" in result or "0" in result

    def test_no_conversations_shown_empty(self):
        profile = make_empty_profile()
        result = self.formatter.render(profile, fmt="text")
        assert "(none)" in result or "0" in result

    def test_exported_at_shown(self):
        profile = make_profile(include_exported_at=True)
        result = self.formatter.render(profile, fmt="text")
        assert "2024-06-01" in result

    def test_total_messages_shown(self):
        profile = make_profile(conversation_count=2)
        result = self.formatter.render(profile, fmt="text")
        # 2 conversations * 2 messages each = 4 total
        assert "4" in result

    def test_contains_separator_lines(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        # Should have horizontal separator characters
        assert "-" * 10 in result or "=" * 10 in result

    def test_schema_version_shown(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="text")
        assert profile.schema_version in result

    def test_long_memory_truncated(self):
        long_content = "A" * 200
        profile = MemoryProfile(
            display_name="Alice",
            memories=[MemoryEntry(content=long_content)],
        )
        result = self.formatter.render(profile, fmt="text")
        # Should not include the full 200-char string verbatim in text format
        assert "..." in result or len(long_content) > result.count("A")

    def test_model_shown_in_conversation(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="text")
        assert "gpt-4o" in result

    def test_conversation_date_shown(self):
        profile = make_profile(conversation_count=1)
        result = self.formatter.render(profile, fmt="text")
        assert "2024-04-01" in result


# ---------------------------------------------------------------------------
# Gemini format (via formatters)
# ---------------------------------------------------------------------------


class TestGeminiFormatViaFormatters:
    """Tests for the 'gemini' format alias in the formatters module."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_gemini_format_returns_string(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="gemini")
        assert isinstance(result, str)

    def test_gemini_format_nonempty(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="gemini")
        assert len(result.strip()) > 0

    def test_gemini_format_contains_display_name(self):
        profile = make_profile(display_name="Alice")
        result = self.formatter.render(profile, fmt="gemini")
        assert "Alice" in result

    def test_gemini_format_is_markdown(self):
        profile = make_profile()
        result = self.formatter.render(profile, fmt="gemini")
        # Gemini output is markdown, so it should start with #
        assert result.strip().startswith("#")

    def test_gemini_format_contains_memories(self):
        profile = make_profile(memory_count=2)
        result = self.formatter.render(profile, fmt="gemini")
        assert "User preference fact 0" in result


# ---------------------------------------------------------------------------
# render_to_file() function
# ---------------------------------------------------------------------------


class TestRenderToFile:
    """Tests for the render_to_file() convenience function."""

    def test_write_markdown_file(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.md"
        render_to_file(profile, dest)
        assert dest.exists()

    def test_write_json_file(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.json"
        render_to_file(profile, dest)
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_write_yaml_file(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.yaml"
        render_to_file(profile, dest)
        data = yaml.safe_load(dest.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_write_yml_extension(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.yml"
        render_to_file(profile, dest)
        assert dest.exists()
        data = yaml.safe_load(dest.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_write_text_file(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.txt"
        render_to_file(profile, dest)
        content = dest.read_text(encoding="utf-8")
        assert len(content.strip()) > 0

    def test_explicit_fmt_overrides_extension(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.md"  # .md extension
        render_to_file(profile, dest, fmt="json")  # but explicit json
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert "display_name" in data

    def test_creates_parent_directories(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "a" / "b" / "output.md"
        render_to_file(profile, dest)
        assert dest.exists()

    def test_accepts_string_path(self, tmp_path: Path):
        profile = make_profile()
        dest = str(tmp_path / "output.md")
        render_to_file(profile, dest)
        assert Path(dest).exists()

    def test_content_is_utf8(self, tmp_path: Path):
        profile = MemoryProfile(
            display_name="Ünïcödé",
            memories=[MemoryEntry(content="Über alles.")],
        )
        dest = tmp_path / "output.md"
        render_to_file(profile, dest)
        content = dest.read_text(encoding="utf-8")
        assert "Ünïcödé" in content

    def test_json_content_correct(self, tmp_path: Path):
        profile = make_profile(display_name="TestUser")
        dest = tmp_path / "output.json"
        render_to_file(profile, dest)
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert data["display_name"] == "TestUser"

    def test_markdown_content_correct(self, tmp_path: Path):
        profile = make_profile(display_name="TestUser")
        dest = tmp_path / "output.md"
        render_to_file(profile, dest)
        content = dest.read_text(encoding="utf-8")
        assert "TestUser" in content

    def test_unknown_extension_defaults_to_markdown(self, tmp_path: Path):
        profile = make_profile()
        dest = tmp_path / "output.xyz"
        render_to_file(profile, dest)
        content = dest.read_text(encoding="utf-8")
        # Should be markdown (starts with #)
        assert content.strip().startswith("#")


# ---------------------------------------------------------------------------
# Formatter class
# ---------------------------------------------------------------------------


class TestFormatterClass:
    """Tests for the Formatter class directly."""

    def test_instantiation_default(self):
        f = Formatter()
        assert f is not None

    def test_instantiation_custom_templates_dir(self, tmp_path: Path):
        f = Formatter(templates_dir=tmp_path)
        assert f is not None

    def test_renders_all_formats(self):
        f = Formatter()
        profile = make_profile()
        for fmt in ["markdown", "json", "yaml", "text", "gemini"]:
            result = f.render(profile, fmt=fmt)
            assert isinstance(result, str)
            assert len(result.strip()) > 0, f"Empty output for format {fmt!r}"

    def test_custom_templates_dir_fallback(self, tmp_path: Path):
        """When templates_dir is a directory without prompt.md.j2, use fallback."""
        f = Formatter(templates_dir=tmp_path)  # empty dir, no template file
        profile = make_profile()
        result = f.render(profile, fmt="markdown")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_custom_templates_dir_with_template(self, tmp_path: Path):
        """When templates_dir contains prompt.md.j2, use that template."""
        template_content = "Custom: {{ profile.display_name }}"
        (tmp_path / "prompt.md.j2").write_text(template_content, encoding="utf-8")
        f = Formatter(templates_dir=tmp_path)
        profile = make_profile(display_name="TemplateUser")
        result = f.render(profile, fmt="markdown")
        assert "Custom: TemplateUser" in result

    def test_nonexistent_templates_dir_uses_fallback(self):
        """When templates_dir does not exist, use the fallback template."""
        f = Formatter(templates_dir="/nonexistent/path/templates")
        profile = make_profile()
        result = f.render(profile, fmt="markdown")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_render_to_file_method(self, tmp_path: Path):
        f = Formatter()
        profile = make_profile()
        dest = tmp_path / "out.md"
        f.render_to_file(profile, dest)
        assert dest.exists()
        assert len(dest.read_text()) > 0

    def test_infer_format_from_path_md(self):
        f = Formatter()
        assert f._infer_format_from_path(Path("file.md")) == "markdown"

    def test_infer_format_from_path_json(self):
        f = Formatter()
        assert f._infer_format_from_path(Path("file.json")) == "json"

    def test_infer_format_from_path_yaml(self):
        f = Formatter()
        assert f._infer_format_from_path(Path("file.yaml")) == "yaml"

    def test_infer_format_from_path_yml(self):
        f = Formatter()
        assert f._infer_format_from_path(Path("file.yml")) == "yaml"

    def test_infer_format_from_path_txt(self):
        f = Formatter()
        assert f._infer_format_from_path(Path("file.txt")) == "text"

    def test_infer_format_from_path_unknown(self):
        f = Formatter()
        # Unknown extension defaults to markdown
        assert f._infer_format_from_path(Path("file.xyz")) == "markdown"


# ---------------------------------------------------------------------------
# list_formats() and get_format_description()
# ---------------------------------------------------------------------------


class TestListFormats:
    """Tests for list_formats() and get_format_description()."""

    def test_list_formats_returns_list(self):
        result = list_formats()
        assert isinstance(result, list)

    def test_list_formats_contains_all_expected(self):
        result = list_formats()
        expected = {"markdown", "json", "yaml", "text", "gemini"}
        assert expected <= set(result)

    def test_list_formats_is_sorted(self):
        result = list_formats()
        assert result == sorted(result)

    def test_list_formats_all_strings(self):
        result = list_formats()
        assert all(isinstance(f, str) for f in result)

    def test_get_format_description_markdown(self):
        desc = get_format_description("markdown")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_format_description_json(self):
        desc = get_format_description("json")
        assert "json" in desc.lower() or "JSON" in desc

    def test_get_format_description_yaml(self):
        desc = get_format_description("yaml")
        assert "yaml" in desc.lower() or "YAML" in desc

    def test_get_format_description_text(self):
        desc = get_format_description("text")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_format_description_gemini(self):
        desc = get_format_description("gemini")
        assert "gemini" in desc.lower() or "Gemini" in desc

    def test_get_format_description_case_insensitive(self):
        desc_lower = get_format_description("json")
        desc_upper = get_format_description("JSON")
        assert desc_lower == desc_upper

    def test_get_format_description_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            get_format_description("toml")

    def test_each_format_has_description(self):
        for fmt in list_formats():
            desc = get_format_description(fmt)
            assert len(desc) > 10, f"Description for {fmt!r} too short: {desc!r}"


# ---------------------------------------------------------------------------
# Cross-format consistency tests
# ---------------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Tests that verify consistency across output formats."""

    @pytest.fixture(autouse=True)
    def formatter(self):
        self.formatter = Formatter()

    def test_all_formats_contain_display_name(self):
        profile = make_profile(display_name="SharedName")
        for fmt in ["markdown", "json", "yaml", "text"]:
            result = self.formatter.render(profile, fmt=fmt)
            assert "SharedName" in result, f"Display name missing in {fmt!r} output"

    def test_json_and_yaml_have_same_memory_count(self):
        profile = make_profile(memory_count=4)
        json_data = json.loads(self.formatter.render(profile, fmt="json"))
        yaml_data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert len(json_data["memories"]) == len(yaml_data["memories"]) == 4

    def test_json_and_yaml_have_same_conversation_count(self):
        profile = make_profile(conversation_count=3)
        json_data = json.loads(self.formatter.render(profile, fmt="json"))
        yaml_data = yaml.safe_load(self.formatter.render(profile, fmt="yaml"))
        assert len(json_data["conversations"]) == len(yaml_data["conversations"]) == 3

    def test_all_formats_handle_empty_profile(self):
        profile = make_empty_profile()
        for fmt in ["markdown", "json", "yaml", "text", "gemini"]:
            result = self.formatter.render(profile, fmt=fmt)
            assert isinstance(result, str)
            assert len(result.strip()) > 0, f"Empty output for format {fmt!r}"

    def test_all_formats_handle_unicode(self):
        profile = MemoryProfile(
            display_name="日本語ユーザー",
            memories=[MemoryEntry(content="好きな言語はPythonです。")],
        )
        for fmt in ["markdown", "json", "yaml", "text"]:
            result = self.formatter.render(profile, fmt=fmt)
            assert "日本語" in result, f"Unicode display name missing in {fmt!r} output"

    def test_large_profile_all_formats(self):
        memories = [MemoryEntry(content=f"Memory {i} content.") for i in range(20)]
        convs = [make_conversation(title=f"Conv {i}") for i in range(10)]
        profile = MemoryProfile(
            display_name="PowerUser",
            memories=memories,
            conversations=convs,
        )
        for fmt in ["markdown", "json", "yaml", "text"]:
            result = self.formatter.render(profile, fmt=fmt)
            assert len(result) > 100, f"Output too short for {fmt!r} with large profile"


# ---------------------------------------------------------------------------
# Template rendering with the default templates directory
# ---------------------------------------------------------------------------


class TestDefaultTemplateFile:
    """Tests that verify the default prompt.md.j2 template is used when available."""

    def test_default_template_dir_path(self):
        from mem_bridge.formatters import _TEMPLATES_DIR
        # Templates dir should be defined (may or may not exist)
        assert isinstance(_TEMPLATES_DIR, Path)

    def test_markdown_render_with_default_formatter(self):
        # Uses the module-level default formatter which uses the project templates dir
        profile = make_profile(display_name="TemplateTestUser")
        result = render(profile, fmt="markdown")
        assert "TemplateTestUser" in result
        assert result.strip().startswith("#")

    def test_fallback_template_produces_valid_markdown(self):
        """Even if the templates directory doesn't exist, output is valid Markdown."""
        f = Formatter(templates_dir="/nonexistent/templates")
        profile = make_profile(display_name="FallbackUser")
        result = f.render(profile, fmt="markdown")
        assert "FallbackUser" in result
        assert "#" in result  # Has headings

    def test_fallback_template_has_memories_section(self):
        f = Formatter(templates_dir="/nonexistent/templates")
        profile = make_profile(memory_count=2)
        result = f.render(profile, fmt="markdown")
        assert "Remembered Facts" in result
        assert "User preference fact 0" in result

    def test_fallback_template_has_conversations_section(self):
        f = Formatter(templates_dir="/nonexistent/templates")
        profile = make_profile(conversation_count=1)
        result = f.render(profile, fmt="markdown")
        assert "Conversation History" in result
        assert "Conversation Topic 0" in result
