"""Unit tests for mem_bridge.differ.

Tests verify:
- DiffResult properties: has_changes, summary, to_dict, to_json, to_yaml
- ProfileDiffer.compare() correctly detects added, removed, and changed memories
- ProfileDiffer.compare() correctly detects added and removed conversations
- ProfileDiffer.compare() detects top-level profile field changes
- ProfileDiffer.render() dispatches correctly to json, yaml, and rich modes
- ProfileDiffer.print_rich() produces non-empty Rich output
- Module-level diff() and compare() convenience functions
- Edge cases: identical profiles, empty profiles, profiles with no memories,
  profiles with no conversations, memories with and without explicit IDs
- Rich output capture via StringIO-backed console
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO

import pytest
import yaml
from rich.console import Console

from mem_bridge.differ import DiffResult, ProfileDiffer, compare, diff
from mem_bridge.models import (
    Conversation,
    ConversationEntry,
    MemoryEntry,
    MemoryProfile,
    Role,
    SourcePlatform,
)


# ---------------------------------------------------------------------------
# Helpers / builders
# ---------------------------------------------------------------------------


def make_memory(
    content: str = "User prefers dark mode.",
    mem_id: str | None = None,
    tags: list[str] | None = None,
    **kwargs,
) -> MemoryEntry:
    """Return a MemoryEntry with sensible defaults."""
    return MemoryEntry(
        content=content,
        id=mem_id,
        tags=tags or [],
        **kwargs,
    )


def make_conversation(
    title: str = "Test Conversation",
    conv_id: str | None = "conv_1",
    message_count: int = 2,
) -> Conversation:
    """Return a Conversation with the requested number of entries."""
    entries = [
        ConversationEntry(
            role=Role.USER if i % 2 == 0 else Role.ASSISTANT,
            content=f"Message {i} content.",
        )
        for i in range(message_count)
    ]
    return Conversation(
        id=conv_id,
        title=title,
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        entries=entries,
    )


def make_profile(
    display_name: str = "Alice",
    source_platform: SourcePlatform = SourcePlatform.CHATGPT,
    memories: list[MemoryEntry] | None = None,
    conversations: list[Conversation] | None = None,
    exported_at: datetime | None = None,
) -> MemoryProfile:
    """Return a MemoryProfile with the supplied or default content."""
    return MemoryProfile(
        source_platform=source_platform,
        display_name=display_name,
        memories=memories or [],
        conversations=conversations or [],
        exported_at=exported_at,
    )


def make_console() -> tuple[Console, StringIO]:
    """Return a Rich Console backed by a StringIO buffer for output capture."""
    buf = StringIO()
    con = Console(file=buf, highlight=False, markup=True)
    return con, buf


# ---------------------------------------------------------------------------
# DiffResult tests
# ---------------------------------------------------------------------------


class TestDiffResult:
    """Tests for the DiffResult data container."""

    def _make_result(
        self,
        added_memories: list | None = None,
        removed_memories: list | None = None,
        changed_memories: list | None = None,
        added_conversations: list | None = None,
        removed_conversations: list | None = None,
        profile_changes: dict | None = None,
    ) -> DiffResult:
        return DiffResult(
            added_memories=added_memories or [],
            removed_memories=removed_memories or [],
            changed_memories=changed_memories or [],
            added_conversations=added_conversations or [],
            removed_conversations=removed_conversations or [],
            profile_changes=profile_changes or {},
            raw_diff={},
            profile_a_name="Profile A",
            profile_b_name="Profile B",
        )

    def test_has_changes_false_when_empty(self):
        result = self._make_result()
        assert result.has_changes is False

    def test_has_changes_true_with_added_memory(self):
        result = self._make_result(
            added_memories=[make_memory("New fact.")]
        )
        assert result.has_changes is True

    def test_has_changes_true_with_removed_memory(self):
        result = self._make_result(
            removed_memories=[make_memory("Old fact.")]
        )
        assert result.has_changes is True

    def test_has_changes_true_with_changed_memory(self):
        old = make_memory("Old content.", mem_id="m1")
        new = make_memory("New content.", mem_id="m1")
        result = self._make_result(changed_memories=[(old, new)])
        assert result.has_changes is True

    def test_has_changes_true_with_added_conversation(self):
        result = self._make_result(
            added_conversations=[{"id": "c1", "title": "New Conv", "message_count": 2}]
        )
        assert result.has_changes is True

    def test_has_changes_true_with_removed_conversation(self):
        result = self._make_result(
            removed_conversations=[{"id": "c1", "title": "Old Conv", "message_count": 2}]
        )
        assert result.has_changes is True

    def test_has_changes_true_with_profile_changes(self):
        result = self._make_result(
            profile_changes={"display_name": ("Alice", "Bob")}
        )
        assert result.has_changes is True

    def test_summary_all_zeros_when_empty(self):
        result = self._make_result()
        s = result.summary
        assert s["added_memories"] == 0
        assert s["removed_memories"] == 0
        assert s["changed_memories"] == 0
        assert s["added_conversations"] == 0
        assert s["removed_conversations"] == 0
        assert s["profile_changes"] == 0

    def test_summary_counts_correct(self):
        old = make_memory("Old.", mem_id="m1")
        new = make_memory("New.", mem_id="m1")
        result = self._make_result(
            added_memories=[make_memory("A."), make_memory("B.")],
            removed_memories=[make_memory("C.")],
            changed_memories=[(old, new)],
            added_conversations=[{"id": "c1", "title": "New", "message_count": 0}],
            profile_changes={"display_name": ("Alice", "Bob")},
        )
        s = result.summary
        assert s["added_memories"] == 2
        assert s["removed_memories"] == 1
        assert s["changed_memories"] == 1
        assert s["added_conversations"] == 1
        assert s["profile_changes"] == 1

    def test_to_dict_has_required_keys(self):
        result = self._make_result()
        d = result.to_dict()
        required = {
            "profiles", "summary", "has_changes",
            "added_memories", "removed_memories", "changed_memories",
            "added_conversations", "removed_conversations", "profile_changes",
        }
        assert required <= set(d.keys())

    def test_to_dict_profiles_section(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["profiles"]["baseline"] == "Profile A"
        assert d["profiles"]["updated"] == "Profile B"

    def test_to_dict_added_memories_structure(self):
        mem = make_memory("Fact.", mem_id="m1", tags=["tag1"])
        result = self._make_result(added_memories=[mem])
        d = result.to_dict()
        assert len(d["added_memories"]) == 1
        assert d["added_memories"][0]["content"] == "Fact."
        assert d["added_memories"][0]["id"] == "m1"
        assert d["added_memories"][0]["tags"] == ["tag1"]

    def test_to_dict_removed_memories_structure(self):
        mem = make_memory("Old fact.", mem_id="m2")
        result = self._make_result(removed_memories=[mem])
        d = result.to_dict()
        assert len(d["removed_memories"]) == 1
        assert d["removed_memories"][0]["content"] == "Old fact."

    def test_to_dict_changed_memories_structure(self):
        old = make_memory("Old content.", mem_id="m1")
        new = make_memory("New content.", mem_id="m1")
        result = self._make_result(changed_memories=[(old, new)])
        d = result.to_dict()
        assert len(d["changed_memories"]) == 1
        assert d["changed_memories"][0]["old"]["content"] == "Old content."
        assert d["changed_memories"][0]["new"]["content"] == "New content."

    def test_to_dict_profile_changes_structure(self):
        result = self._make_result(
            profile_changes={"display_name": ("Alice", "Bob")}
        )
        d = result.to_dict()
        assert "display_name" in d["profile_changes"]
        assert d["profile_changes"]["display_name"]["old"] == "Alice"
        assert d["profile_changes"]["display_name"]["new"] == "Bob"

    def test_to_json_produces_valid_json(self):
        result = self._make_result(
            added_memories=[make_memory("Fact.")]
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "added_memories" in data

    def test_to_json_is_indented(self):
        result = self._make_result()
        json_str = result.to_json()
        assert "\n" in json_str

    def test_to_yaml_produces_valid_yaml(self):
        result = self._make_result(
            removed_memories=[make_memory("Old fact.")]
        )
        yaml_str = result.to_yaml()
        data = yaml.safe_load(yaml_str)
        assert isinstance(data, dict)
        assert "removed_memories" in data

    def test_to_yaml_has_profiles_key(self):
        result = self._make_result()
        data = yaml.safe_load(result.to_yaml())
        assert "profiles" in data


# ---------------------------------------------------------------------------
# ProfileDiffer – memory diff tests
# ---------------------------------------------------------------------------


class TestProfileDifferMemories:
    """Tests for ProfileDiffer memory comparison logic."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_no_changes_identical_profiles(self):
        mem = make_memory("User likes Python.", mem_id="m1")
        profile_a = make_profile(memories=[mem])
        profile_b = make_profile(memories=[make_memory("User likes Python.", mem_id="m1")])
        result = self.differ.compare(profile_a, profile_b)
        assert result.added_memories == []
        assert result.removed_memories == []
        assert result.changed_memories == []

    def test_detect_added_memory_by_content_key(self):
        profile_a = make_profile(memories=[
            make_memory("Fact A."),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact A."),
            make_memory("Fact B."),  # new
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_memories) == 1
        assert result.added_memories[0].content == "Fact B."

    def test_detect_removed_memory_by_content_key(self):
        profile_a = make_profile(memories=[
            make_memory("Fact A."),
            make_memory("Fact B."),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact A."),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.removed_memories) == 1
        assert result.removed_memories[0].content == "Fact B."

    def test_detect_added_memory_by_id_key(self):
        profile_a = make_profile(memories=[
            make_memory("Fact A.", mem_id="m1"),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact A.", mem_id="m1"),
            make_memory("Fact B.", mem_id="m2"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_memories) == 1
        assert result.added_memories[0].id == "m2"

    def test_detect_removed_memory_by_id_key(self):
        profile_a = make_profile(memories=[
            make_memory("Fact A.", mem_id="m1"),
            make_memory("Fact B.", mem_id="m2"),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact A.", mem_id="m1"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.removed_memories) == 1
        assert result.removed_memories[0].id == "m2"

    def test_detect_changed_memory_content(self):
        profile_a = make_profile(memories=[
            make_memory("Original content.", mem_id="m1"),
        ])
        profile_b = make_profile(memories=[
            make_memory("Updated content.", mem_id="m1"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.changed_memories) == 1
        old, new = result.changed_memories[0]
        assert old.content == "Original content."
        assert new.content == "Updated content."

    def test_detect_changed_memory_tags(self):
        profile_a = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=["old_tag"]),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=["new_tag"]),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.changed_memories) == 1
        old, new = result.changed_memories[0]
        assert old.tags == ["old_tag"]
        assert new.tags == ["new_tag"]

    def test_no_changes_when_memories_identical_with_ids(self):
        profile_a = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=["t1"]),
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=["t1"]),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert not result.has_changes

    def test_multiple_additions(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[
            make_memory(f"Fact {i}.") for i in range(5)
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_memories) == 5
        assert result.removed_memories == []

    def test_multiple_removals(self):
        profile_a = make_profile(memories=[
            make_memory(f"Fact {i}.") for i in range(5)
        ])
        profile_b = make_profile(memories=[])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.removed_memories) == 5
        assert result.added_memories == []

    def test_mixed_add_remove_change(self):
        profile_a = make_profile(memories=[
            make_memory("Keep me.", mem_id="m1"),
            make_memory("Change me.", mem_id="m2"),
            make_memory("Remove me.", mem_id="m3"),
        ])
        profile_b = make_profile(memories=[
            make_memory("Keep me.", mem_id="m1"),
            make_memory("Changed content.", mem_id="m2"),
            make_memory("Add me.", mem_id="m4"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_memories) == 1
        assert result.added_memories[0].id == "m4"
        assert len(result.removed_memories) == 1
        assert result.removed_memories[0].id == "m3"
        assert len(result.changed_memories) == 1

    def test_memories_without_ids_use_content_key(self):
        profile_a = make_profile(memories=[
            make_memory("Content-keyed fact."),
        ])
        profile_b = make_profile(memories=[
            make_memory("Different content."),  # same lack of id, different content
        ])
        result = self.differ.compare(profile_a, profile_b)
        # Without matching IDs, this looks like remove + add
        assert len(result.removed_memories) == 1
        assert len(result.added_memories) == 1

    def test_empty_both_profiles_no_changes(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[])
        result = self.differ.compare(profile_a, profile_b)
        assert not result.has_changes
        assert result.added_memories == []
        assert result.removed_memories == []


# ---------------------------------------------------------------------------
# ProfileDiffer – conversation diff tests
# ---------------------------------------------------------------------------


class TestProfileDifferConversations:
    """Tests for ProfileDiffer conversation comparison logic."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_detect_added_conversation(self):
        conv_a = make_conversation(title="Conv A", conv_id="c1")
        profile_a = make_profile(conversations=[conv_a])
        profile_b = make_profile(conversations=[
            conv_a,
            make_conversation(title="Conv B", conv_id="c2"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_conversations) == 1
        assert result.added_conversations[0]["id"] == "c2"
        assert result.added_conversations[0]["title"] == "Conv B"

    def test_detect_removed_conversation(self):
        conv_a = make_conversation(title="Conv A", conv_id="c1")
        conv_b = make_conversation(title="Conv B", conv_id="c2")
        profile_a = make_profile(conversations=[conv_a, conv_b])
        profile_b = make_profile(conversations=[conv_a])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.removed_conversations) == 1
        assert result.removed_conversations[0]["id"] == "c2"

    def test_no_conversation_changes_when_identical(self):
        conv = make_conversation(title="Conv A", conv_id="c1")
        profile_a = make_profile(conversations=[conv])
        # Re-create a fresh Conversation with same id
        conv_b = make_conversation(title="Conv A", conv_id="c1")
        profile_b = make_profile(conversations=[conv_b])
        result = self.differ.compare(profile_a, profile_b)
        assert result.added_conversations == []
        assert result.removed_conversations == []

    def test_empty_conversations_both_sides(self):
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=[])
        result = self.differ.compare(profile_a, profile_b)
        assert result.added_conversations == []
        assert result.removed_conversations == []

    def test_all_conversations_added(self):
        convs = [make_conversation(title=f"Conv {i}", conv_id=f"c{i}") for i in range(3)]
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=convs)
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.added_conversations) == 3
        assert result.removed_conversations == []

    def test_all_conversations_removed(self):
        convs = [make_conversation(title=f"Conv {i}", conv_id=f"c{i}") for i in range(3)]
        profile_a = make_profile(conversations=convs)
        profile_b = make_profile(conversations=[])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.removed_conversations) == 3
        assert result.added_conversations == []

    def test_added_conversation_has_title_field(self):
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=[
            make_conversation(title="New Convo", conv_id="c99")
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert result.added_conversations[0]["title"] == "New Convo"

    def test_removed_conversation_has_id_field(self):
        profile_a = make_profile(conversations=[
            make_conversation(title="Old Convo", conv_id="c_old")
        ])
        profile_b = make_profile(conversations=[])
        result = self.differ.compare(profile_a, profile_b)
        assert result.removed_conversations[0]["id"] == "c_old"


# ---------------------------------------------------------------------------
# ProfileDiffer – profile field diff tests
# ---------------------------------------------------------------------------


class TestProfileDifferFields:
    """Tests for top-level profile field comparison."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_detect_display_name_change(self):
        profile_a = make_profile(display_name="Alice")
        profile_b = make_profile(display_name="Bob")
        result = self.differ.compare(profile_a, profile_b)
        assert "display_name" in result.profile_changes
        old, new = result.profile_changes["display_name"]
        assert old == "Alice"
        assert new == "Bob"

    def test_detect_source_platform_change(self):
        profile_a = make_profile(source_platform=SourcePlatform.CHATGPT)
        profile_b = make_profile(source_platform=SourcePlatform.CLAUDE)
        result = self.differ.compare(profile_a, profile_b)
        assert "source_platform" in result.profile_changes

    def test_detect_exported_at_change(self):
        ts_a = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts_b = datetime(2024, 6, 1, tzinfo=timezone.utc)
        profile_a = make_profile(exported_at=ts_a)
        profile_b = make_profile(exported_at=ts_b)
        result = self.differ.compare(profile_a, profile_b)
        assert "exported_at" in result.profile_changes

    def test_no_field_changes_identical_profiles(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        profile_a = make_profile(display_name="Alice", exported_at=ts)
        profile_b = make_profile(display_name="Alice", exported_at=ts)
        result = self.differ.compare(profile_a, profile_b)
        assert result.profile_changes == {}

    def test_no_field_changes_both_none_exported_at(self):
        profile_a = make_profile(display_name="Alice", exported_at=None)
        profile_b = make_profile(display_name="Alice", exported_at=None)
        result = self.differ.compare(profile_a, profile_b)
        assert "exported_at" not in result.profile_changes

    def test_field_change_exported_at_none_to_value(self):
        profile_a = make_profile(exported_at=None)
        profile_b = make_profile(exported_at=datetime(2024, 6, 1, tzinfo=timezone.utc))
        result = self.differ.compare(profile_a, profile_b)
        assert "exported_at" in result.profile_changes


# ---------------------------------------------------------------------------
# ProfileDiffer – has_changes and summary
# ---------------------------------------------------------------------------


class TestProfileDifferHasChanges:
    """Tests for the has_changes and summary aggregation."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_identical_profiles_no_changes(self):
        profile = make_profile(
            memories=[make_memory("Fact.", mem_id="m1")],
            conversations=[make_conversation(conv_id="c1")],
        )
        # Create a new profile with the same data
        profile_b = make_profile(
            memories=[make_memory("Fact.", mem_id="m1")],
            conversations=[make_conversation(conv_id="c1")],
        )
        result = self.differ.compare(profile, profile_b)
        assert not result.has_changes

    def test_summary_reflects_all_change_types(self):
        profile_a = make_profile(
            display_name="Alice",
            memories=[
                make_memory("Keep.", mem_id="m1"),
                make_memory("Change.", mem_id="m2"),
                make_memory("Remove.", mem_id="m3"),
            ],
            conversations=[make_conversation(conv_id="c1")],
        )
        profile_b = make_profile(
            display_name="Bob",  # field change
            memories=[
                make_memory("Keep.", mem_id="m1"),
                make_memory("Changed.", mem_id="m2"),
                make_memory("Added.", mem_id="m4"),
            ],
            conversations=[
                make_conversation(conv_id="c2"),  # different conv
            ],
        )
        result = self.differ.compare(profile_a, profile_b)
        s = result.summary
        assert s["added_memories"] == 1  # m4
        assert s["removed_memories"] == 1  # m3
        assert s["changed_memories"] == 1  # m2
        assert s["added_conversations"] == 1  # c2
        assert s["removed_conversations"] == 1  # c1
        assert s["profile_changes"] >= 1  # display_name changed

    def test_only_addition_detected(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[make_memory("New fact.")])
        result = self.differ.compare(profile_a, profile_b)
        assert result.has_changes
        assert result.summary["added_memories"] == 1
        assert result.summary["removed_memories"] == 0

    def test_only_removal_detected(self):
        profile_a = make_profile(memories=[make_memory("Old fact.")])
        profile_b = make_profile(memories=[])
        result = self.differ.compare(profile_a, profile_b)
        assert result.has_changes
        assert result.summary["removed_memories"] == 1
        assert result.summary["added_memories"] == 0


# ---------------------------------------------------------------------------
# ProfileDiffer – render() method
# ---------------------------------------------------------------------------


class TestProfileDifferRender:
    """Tests for the render() dispatch method."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def _make_changed_result(self) -> DiffResult:
        profile_a = make_profile(
            display_name="Alice",
            memories=[make_memory("Old fact.", mem_id="m1")],
        )
        profile_b = make_profile(
            display_name="Bob",
            memories=[
                make_memory("New fact.", mem_id="m1"),
                make_memory("Another fact."),
            ],
        )
        return self.differ.compare(profile_a, profile_b)

    def test_render_json_returns_string(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="json")
        assert isinstance(output, str)

    def test_render_json_is_valid_json(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="json")
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_render_yaml_returns_string(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="yaml")
        assert isinstance(output, str)

    def test_render_yaml_is_valid_yaml(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="yaml")
        data = yaml.safe_load(output)
        assert isinstance(data, dict)

    def test_render_rich_returns_none_for_default_console(self):
        result = self._make_changed_result()
        console, buf = make_console()
        output = self.differ.render(result, fmt="rich", console=console)
        # May return string or None depending on implementation
        # Either way the buffer should have content
        assert len(buf.getvalue()) > 0

    def test_render_unsupported_format_raises(self):
        result = self._make_changed_result()
        with pytest.raises(ValueError, match="Unsupported diff format"):
            self.differ.render(result, fmt="toml")

    def test_render_json_case_insensitive(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="JSON")
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_render_yaml_case_insensitive(self):
        result = self._make_changed_result()
        output = self.differ.render(result, fmt="YAML")
        data = yaml.safe_load(output)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# ProfileDiffer – print_rich() output tests
# ---------------------------------------------------------------------------


class TestProfileDifferPrintRich:
    """Tests for the Rich-formatted output."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def _render_to_string(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> str:
        """Run print_rich and capture output to a string."""
        result = self.differ.compare(profile_a, profile_b)
        console, buf = make_console()
        self.differ.print_rich(result, console=console)
        return buf.getvalue()

    def test_output_is_nonempty(self):
        profile_a = make_profile(display_name="Alice")
        profile_b = make_profile(display_name="Bob")
        output = self._render_to_string(profile_a, profile_b)
        assert len(output.strip()) > 0

    def test_output_contains_profile_names(self):
        profile_a = make_profile(display_name="AliceProfile")
        profile_b = make_profile(display_name="BobProfile")
        output = self._render_to_string(profile_a, profile_b)
        assert "AliceProfile" in output
        assert "BobProfile" in output

    def test_output_contains_diff_report_title(self):
        profile_a = make_profile()
        profile_b = make_profile()
        output = self._render_to_string(profile_a, profile_b)
        assert "Diff Report" in output or "diff" in output.lower()

    def test_output_contains_summary_table(self):
        profile_a = make_profile()
        profile_b = make_profile()
        output = self._render_to_string(profile_a, profile_b)
        # Summary table should mention memory or conversation categories
        assert "Memories" in output or "memories" in output.lower()

    def test_no_changes_message_shown(self):
        profile_a = make_profile(memories=[make_memory("Fact.", mem_id="m1")])
        profile_b = make_profile(memories=[make_memory("Fact.", mem_id="m1")])
        output = self._render_to_string(profile_a, profile_b)
        assert "No differences" in output or "no differences" in output.lower()

    def test_added_memory_shown_in_output(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[make_memory("Brand new fact.")])
        output = self._render_to_string(profile_a, profile_b)
        assert "Brand new fact." in output

    def test_removed_memory_shown_in_output(self):
        profile_a = make_profile(memories=[make_memory("Deleted fact.")])
        profile_b = make_profile(memories=[])
        output = self._render_to_string(profile_a, profile_b)
        assert "Deleted fact." in output

    def test_changed_memory_shows_old_and_new(self):
        profile_a = make_profile(memories=[
            make_memory("Before change.", mem_id="m1")
        ])
        profile_b = make_profile(memories=[
            make_memory("After change.", mem_id="m1")
        ])
        output = self._render_to_string(profile_a, profile_b)
        assert "Before change." in output
        assert "After change." in output

    def test_added_conversation_shown_in_output(self):
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=[
            make_conversation(title="Brand New Convo", conv_id="c99")
        ])
        output = self._render_to_string(profile_a, profile_b)
        assert "Brand New Convo" in output

    def test_removed_conversation_shown_in_output(self):
        profile_a = make_profile(conversations=[
            make_conversation(title="Old Convo", conv_id="c_old")
        ])
        profile_b = make_profile(conversations=[])
        output = self._render_to_string(profile_a, profile_b)
        assert "Old Convo" in output

    def test_profile_field_change_shown(self):
        profile_a = make_profile(display_name="OldName")
        profile_b = make_profile(display_name="NewName")
        output = self._render_to_string(profile_a, profile_b)
        assert "OldName" in output or "NewName" in output

    def test_rich_output_with_empty_both_profiles(self):
        profile_a = make_profile(memories=[], conversations=[])
        profile_b = make_profile(memories=[], conversations=[])
        output = self._render_to_string(profile_a, profile_b)
        assert isinstance(output, str)
        assert len(output.strip()) > 0

    def test_rich_output_large_diff(self):
        profile_a = make_profile(memories=[
            make_memory(f"Old fact {i}.", mem_id=f"m{i}") for i in range(10)
        ])
        profile_b = make_profile(memories=[
            make_memory(f"New fact {i}.", mem_id=f"m{i}") for i in range(10)
        ])
        output = self._render_to_string(profile_a, profile_b)
        assert len(output) > 200

    def test_memory_diff_section_header(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[make_memory("Added.")])
        output = self._render_to_string(profile_a, profile_b)
        assert "Memory Diff" in output or "memory" in output.lower()

    def test_conversation_diff_section_header(self):
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=[
            make_conversation(title="New Conv", conv_id="c1")
        ])
        output = self._render_to_string(profile_a, profile_b)
        assert "Conversation" in output


# ---------------------------------------------------------------------------
# Module-level compare() and diff() convenience functions
# ---------------------------------------------------------------------------


class TestModuleLevelFunctions:
    """Tests for the compare() and diff() module-level convenience functions."""

    def test_compare_returns_diff_result(self):
        profile_a = make_profile()
        profile_b = make_profile()
        result = compare(profile_a, profile_b)
        assert isinstance(result, DiffResult)

    def test_compare_detects_added_memory(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[make_memory("New fact.")])
        result = compare(profile_a, profile_b)
        assert len(result.added_memories) == 1

    def test_compare_detects_removed_memory(self):
        profile_a = make_profile(memories=[make_memory("Old fact.")])
        profile_b = make_profile(memories=[])
        result = compare(profile_a, profile_b)
        assert len(result.removed_memories) == 1

    def test_diff_json_returns_string(self):
        profile_a = make_profile()
        profile_b = make_profile()
        output = diff(profile_a, profile_b, fmt="json")
        assert isinstance(output, str)
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_diff_yaml_returns_string(self):
        profile_a = make_profile()
        profile_b = make_profile()
        output = diff(profile_a, profile_b, fmt="yaml")
        assert isinstance(output, str)
        data = yaml.safe_load(output)
        assert isinstance(data, dict)

    def test_diff_rich_with_console(self):
        profile_a = make_profile()
        profile_b = make_profile()
        console, buf = make_console()
        diff(profile_a, profile_b, fmt="rich", console=console)
        assert len(buf.getvalue()) > 0

    def test_diff_default_fmt_is_rich(self):
        profile_a = make_profile()
        profile_b = make_profile()
        console, buf = make_console()
        diff(profile_a, profile_b, console=console)
        assert len(buf.getvalue()) > 0

    def test_diff_unsupported_fmt_raises(self):
        profile_a = make_profile()
        profile_b = make_profile()
        with pytest.raises(ValueError, match="Unsupported diff format"):
            diff(profile_a, profile_b, fmt="xml")

    def test_diff_json_contains_summary(self):
        profile_a = make_profile(memories=[make_memory("Fact.")])
        profile_b = make_profile(memories=[])
        output = diff(profile_a, profile_b, fmt="json")
        data = json.loads(output)
        assert "summary" in data
        assert data["summary"]["removed_memories"] == 1

    def test_diff_yaml_contains_profiles_section(self):
        profile_a = make_profile(display_name="Alpha")
        profile_b = make_profile(display_name="Beta")
        output = diff(profile_a, profile_b, fmt="yaml")
        data = yaml.safe_load(output)
        assert "profiles" in data

    def test_diff_json_has_changes_true_when_different(self):
        profile_a = make_profile(memories=[make_memory("A.")])
        profile_b = make_profile(memories=[make_memory("B.")])
        output = diff(profile_a, profile_b, fmt="json")
        data = json.loads(output)
        assert data["has_changes"] is True

    def test_diff_json_has_changes_false_when_identical(self):
        profile_a = make_profile(memories=[make_memory("Same.", mem_id="m1")])
        profile_b = make_profile(memories=[make_memory("Same.", mem_id="m1")])
        output = diff(profile_a, profile_b, fmt="json")
        data = json.loads(output)
        assert data["has_changes"] is False


# ---------------------------------------------------------------------------
# DiffResult serialisation tests
# ---------------------------------------------------------------------------


class TestDiffResultSerialisation:
    """Tests for DiffResult.to_json() and to_yaml() with ProfileDiffer output."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_json_round_trip_preserves_summary(self):
        profile_a = make_profile(
            memories=[make_memory("Old.", mem_id="m1"), make_memory("Remove.")]
        )
        profile_b = make_profile(
            memories=[make_memory("Old.", mem_id="m1"), make_memory("New.")]
        )
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert data["summary"]["added_memories"] == result.summary["added_memories"]
        assert data["summary"]["removed_memories"] == result.summary["removed_memories"]

    def test_yaml_round_trip_preserves_has_changes(self):
        profile_a = make_profile(display_name="Alice")
        profile_b = make_profile(display_name="Bob")
        result = self.differ.compare(profile_a, profile_b)
        data = yaml.safe_load(result.to_yaml())
        assert data["has_changes"] == result.has_changes

    def test_json_added_memories_list(self):
        profile_a = make_profile(memories=[])
        profile_b = make_profile(memories=[
            make_memory("Fact A.", mem_id="mA"),
            make_memory("Fact B.", mem_id="mB"),
        ])
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert len(data["added_memories"]) == 2
        contents = {m["content"] for m in data["added_memories"]}
        assert "Fact A." in contents
        assert "Fact B." in contents

    def test_json_removed_memories_list(self):
        profile_a = make_profile(memories=[
            make_memory("Gone.", mem_id="mZ")
        ])
        profile_b = make_profile(memories=[])
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert len(data["removed_memories"]) == 1
        assert data["removed_memories"][0]["content"] == "Gone."

    def test_json_changed_memories_list(self):
        profile_a = make_profile(memories=[make_memory("Before.", mem_id="m1")])
        profile_b = make_profile(memories=[make_memory("After.", mem_id="m1")])
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert len(data["changed_memories"]) == 1
        assert data["changed_memories"][0]["old"]["content"] == "Before."
        assert data["changed_memories"][0]["new"]["content"] == "After."

    def test_json_added_conversations_list(self):
        profile_a = make_profile(conversations=[])
        profile_b = make_profile(conversations=[
            make_conversation(title="New Chat", conv_id="c_new")
        ])
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert len(data["added_conversations"]) == 1
        assert data["added_conversations"][0]["title"] == "New Chat"

    def test_json_removed_conversations_list(self):
        profile_a = make_profile(conversations=[
            make_conversation(title="Old Chat", conv_id="c_old")
        ])
        profile_b = make_profile(conversations=[])
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert len(data["removed_conversations"]) == 1
        assert data["removed_conversations"][0]["title"] == "Old Chat"

    def test_json_profile_changes(self):
        profile_a = make_profile(display_name="OriginalName")
        profile_b = make_profile(display_name="UpdatedName")
        result = self.differ.compare(profile_a, profile_b)
        data = json.loads(result.to_json())
        assert "display_name" in data["profile_changes"]
        assert data["profile_changes"]["display_name"]["old"] == "OriginalName"
        assert data["profile_changes"]["display_name"]["new"] == "UpdatedName"


# ---------------------------------------------------------------------------
# ProfileDiffer – configuration options
# ---------------------------------------------------------------------------


class TestProfileDifferConfiguration:
    """Tests for ProfileDiffer constructor configuration options."""

    def test_ignore_order_true_default(self):
        """With ignore_order=True (default), reordered memories are not changes."""
        memories = [
            make_memory(f"Fact {i}.", mem_id=f"m{i}") for i in range(3)
        ]
        profile_a = make_profile(memories=memories)
        # Reverse the order
        profile_b = make_profile(memories=list(reversed(memories)))
        differ = ProfileDiffer(ignore_order=True)
        result = differ.compare(profile_a, profile_b)
        # Should not detect added/removed memories (same content, just reordered)
        assert result.added_memories == []
        assert result.removed_memories == []

    def test_custom_differ_instance(self):
        """ProfileDiffer can be instantiated with custom parameters."""
        differ = ProfileDiffer(
            ignore_order=False,
            ignore_numeric_type_changes=False,
        )
        profile_a = make_profile()
        profile_b = make_profile()
        result = differ.compare(profile_a, profile_b)
        assert isinstance(result, DiffResult)

    def test_significant_digits_parameter(self):
        """ProfileDiffer accepts significant_digits parameter without error."""
        differ = ProfileDiffer(significant_digits=3)
        profile_a = make_profile()
        profile_b = make_profile()
        result = differ.compare(profile_a, profile_b)
        assert isinstance(result, DiffResult)


# ---------------------------------------------------------------------------
# Edge cases and regression tests
# ---------------------------------------------------------------------------


class TestDifferEdgeCases:
    """Edge case and regression tests for the diff engine."""

    @pytest.fixture(autouse=True)
    def differ(self):
        self.differ = ProfileDiffer()

    def test_both_empty_profiles(self):
        profile_a = MemoryProfile()
        profile_b = MemoryProfile()
        result = self.differ.compare(profile_a, profile_b)
        assert not result.has_changes

    def test_profile_with_very_long_memory_content(self):
        long_content = "A" * 500
        profile_a = make_profile(memories=[make_memory(long_content, mem_id="m1")])
        profile_b = make_profile(memories=[make_memory("B" * 500, mem_id="m1")])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.changed_memories) == 1

    def test_memory_with_unicode_content(self):
        profile_a = make_profile(memories=[
            make_memory("日本語のテスト", mem_id="m1")
        ])
        profile_b = make_profile(memories=[
            make_memory("Updated 日本語", mem_id="m1")
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.changed_memories) == 1
        old, new = result.changed_memories[0]
        assert old.content == "日本語のテスト"
        assert new.content == "Updated 日本語"

    def test_memory_with_many_tags(self):
        profile_a = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=[f"tag_{i}" for i in range(10)])
        ])
        profile_b = make_profile(memories=[
            make_memory("Fact.", mem_id="m1", tags=[f"new_tag_{i}" for i in range(10)])
        ])
        result = self.differ.compare(profile_a, profile_b)
        assert len(result.changed_memories) == 1

    def test_diff_result_raw_diff_is_dict(self):
        profile_a = make_profile()
        profile_b = make_profile()
        result = self.differ.compare(profile_a, profile_b)
        assert isinstance(result.raw_diff, dict)

    def test_print_rich_with_explicitly_created_console(self):
        """print_rich() should not crash when given a custom Console."""
        profile_a = make_profile(memories=[make_memory("Old.")])
        profile_b = make_profile(memories=[make_memory("New.")])
        result = self.differ.compare(profile_a, profile_b)
        console, buf = make_console()
        self.differ.print_rich(result, console=console)
        output = buf.getvalue()
        assert len(output.strip()) > 0

    def test_large_profile_comparison(self):
        """Diff engine handles large profiles without error."""
        memories_a = [make_memory(f"Fact {i}.", mem_id=f"m{i}") for i in range(50)]
        memories_b = [
            make_memory(f"Updated fact {i}.", mem_id=f"m{i}") for i in range(40)
        ] + [
            make_memory(f"New fact {i}.", mem_id=f"new_m{i}") for i in range(20)
        ]
        convs_a = [make_conversation(conv_id=f"c{i}", title=f"Conv {i}") for i in range(5)]
        convs_b = [make_conversation(conv_id=f"c{i}", title=f"Conv {i}") for i in range(3)] + [
            make_conversation(conv_id=f"new_c{i}", title=f"New Conv {i}") for i in range(4)
        ]
        profile_a = make_profile(memories=memories_a, conversations=convs_a)
        profile_b = make_profile(memories=memories_b, conversations=convs_b)
        result = self.differ.compare(profile_a, profile_b)
        assert isinstance(result, DiffResult)
        assert result.has_changes
        # Verify summary is non-negative
        for count in result.summary.values():
            assert count >= 0

    def test_conversation_without_id_uses_title_as_key(self):
        """Conversations without explicit IDs fall back to title as the key."""
        conv_a = Conversation(
            id=None,
            title="Unique Title Conversation",
            entries=[ConversationEntry(role=Role.USER, content="Hello.")],
        )
        conv_b = Conversation(
            id=None,
            title="Different Title Conversation",
            entries=[ConversationEntry(role=Role.USER, content="Hi.")],
        )
        profile_a = make_profile(conversations=[conv_a])
        profile_b = make_profile(conversations=[conv_b])
        result = self.differ.compare(profile_a, profile_b)
        assert result.added_conversations or result.removed_conversations

    def test_diff_result_profiles_names_stored_correctly(self):
        profile_a = make_profile(display_name="ProfileAlpha")
        profile_b = make_profile(display_name="ProfileBeta")
        result = self.differ.compare(profile_a, profile_b)
        assert result.profile_a_name == "ProfileAlpha"
        assert result.profile_b_name == "ProfileBeta"

    def test_to_dict_has_changes_boolean(self):
        profile_a = make_profile(memories=[make_memory("A.")])
        profile_b = make_profile(memories=[make_memory("B.")])
        result = self.differ.compare(profile_a, profile_b)
        d = result.to_dict()
        assert isinstance(d["has_changes"], bool)

    def test_memory_id_none_handled_gracefully(self):
        """Memories with id=None should not cause errors."""
        profile_a = make_profile(memories=[
            make_memory("A fact.", mem_id=None)
        ])
        profile_b = make_profile(memories=[
            make_memory("A fact.", mem_id=None)
        ])
        result = self.differ.compare(profile_a, profile_b)
        # Same content, no id → same key → no changes
        assert result.added_memories == []
        assert result.removed_memories == []
