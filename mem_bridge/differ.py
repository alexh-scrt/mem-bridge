"""Diff engine for mem_bridge.

This module provides the :class:`ProfileDiffer` class and the module-level
:func:`diff` function that compare two :class:`~mem_bridge.models.MemoryProfile`
objects and produce human-readable diff reports.

The diff engine uses :mod:`deepdiff` to compute structural differences between
the canonical profile representations and then renders the results using
:mod:`rich` for terminal output or returns structured diff data for programmatic
use.

Three output modes are supported:

* **rich** – A colourised, human-readable :class:`rich.console.Console` output
  with tables and colour-coded additions, deletions, and modifications.
* **json** – A machine-readable JSON string of the raw diff data.
* **yaml** – A machine-readable YAML string of the raw diff data.

Example usage::

    from mem_bridge.differ import diff
    from mem_bridge.models import MemoryProfile

    profile_a = MemoryProfile.from_file("backup_jan.json")
    profile_b = MemoryProfile.from_file("backup_jun.json")

    # Print a Rich-formatted diff to the terminal
    diff(profile_a, profile_b, fmt="rich")

    # Get a JSON diff report
    json_report = diff(profile_a, profile_b, fmt="json")
    print(json_report)
"""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from typing import Any

import yaml
from deepdiff import DeepDiff
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from mem_bridge.models import MemoryEntry, MemoryProfile


# ---------------------------------------------------------------------------
# Data classes for structured diff results
# ---------------------------------------------------------------------------


class DiffResult:
    """Container for the result of comparing two :class:`MemoryProfile` objects.

    Attributes
    ----------
    added_memories:
        Memory entries present in *profile_b* but not in *profile_a*.
    removed_memories:
        Memory entries present in *profile_a* but not in *profile_b*.
    changed_memories:
        Memory entries that exist in both profiles but with differing content
        or metadata.  Each element is a tuple of ``(old_entry, new_entry)``.
    added_conversations:
        Conversation IDs/titles present in *profile_b* but not in *profile_a*.
    removed_conversations:
        Conversation IDs/titles present in *profile_a* but not in *profile_b*.
    profile_changes:
        Dictionary of top-level profile field changes (excluding memories and
        conversations), mapping field name to ``(old_value, new_value)``.
    raw_diff:
        The raw :class:`deepdiff.DeepDiff` result dictionary for advanced use.
    profile_a_name:
        Display name of the baseline profile.
    profile_b_name:
        Display name of the updated profile.
    """

    def __init__(
        self,
        added_memories: list[MemoryEntry],
        removed_memories: list[MemoryEntry],
        changed_memories: list[tuple[MemoryEntry, MemoryEntry]],
        added_conversations: list[dict[str, Any]],
        removed_conversations: list[dict[str, Any]],
        profile_changes: dict[str, tuple[Any, Any]],
        raw_diff: dict[str, Any],
        profile_a_name: str,
        profile_b_name: str,
    ) -> None:
        self.added_memories = added_memories
        self.removed_memories = removed_memories
        self.changed_memories = changed_memories
        self.added_conversations = added_conversations
        self.removed_conversations = removed_conversations
        self.profile_changes = profile_changes
        self.raw_diff = raw_diff
        self.profile_a_name = profile_a_name
        self.profile_b_name = profile_b_name

    @property
    def has_changes(self) -> bool:
        """Return ``True`` if any differences were detected between the profiles."""
        return bool(
            self.added_memories
            or self.removed_memories
            or self.changed_memories
            or self.added_conversations
            or self.removed_conversations
            or self.profile_changes
        )

    @property
    def summary(self) -> dict[str, int]:
        """Return a counts summary of the diff result.

        Returns
        -------
        dict[str, int]
            Dictionary with keys:
            ``'added_memories'``, ``'removed_memories'``,
            ``'changed_memories'``, ``'added_conversations'``,
            ``'removed_conversations'``, ``'profile_changes'``.
        """
        return {
            "added_memories": len(self.added_memories),
            "removed_memories": len(self.removed_memories),
            "changed_memories": len(self.changed_memories),
            "added_conversations": len(self.added_conversations),
            "removed_conversations": len(self.removed_conversations),
            "profile_changes": len(self.profile_changes),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the diff result to a plain Python dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation suitable for JSON/YAML serialisation.
        """
        return {
            "profiles": {
                "baseline": self.profile_a_name,
                "updated": self.profile_b_name,
            },
            "summary": self.summary,
            "has_changes": self.has_changes,
            "added_memories": [
                {"content": m.content, "id": m.id, "tags": m.tags}
                for m in self.added_memories
            ],
            "removed_memories": [
                {"content": m.content, "id": m.id, "tags": m.tags}
                for m in self.removed_memories
            ],
            "changed_memories": [
                {
                    "old": {"content": old.content, "id": old.id, "tags": old.tags},
                    "new": {"content": new.content, "id": new.id, "tags": new.tags},
                }
                for old, new in self.changed_memories
            ],
            "added_conversations": self.added_conversations,
            "removed_conversations": self.removed_conversations,
            "profile_changes": {
                field: {"old": str(old_val), "new": str(new_val)}
                for field, (old_val, new_val) in self.profile_changes.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the diff result to a JSON string.

        Parameters
        ----------
        indent:
            Number of spaces for JSON indentation.

        Returns
        -------
        str
            Pretty-printed JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Serialise the diff result to a YAML string.

        Returns
        -------
        str
            YAML string.
        """
        return yaml.dump(
            self.to_dict(),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


# ---------------------------------------------------------------------------
# ProfileDiffer class
# ---------------------------------------------------------------------------


class ProfileDiffer:
    """Compares two :class:`~mem_bridge.models.MemoryProfile` objects.

    Uses :mod:`deepdiff` for structural comparison and :mod:`rich` for
    terminal output rendering.

    Parameters
    ----------
    ignore_order:
        If ``True`` (default), list order differences in memories and
        conversations are ignored.  Set to ``False`` to treat order
        changes as differences.
    ignore_numeric_type_changes:
        If ``True`` (default), numeric type differences (e.g. int vs float
        for timestamps) are ignored.
    significant_digits:
        Number of significant digits to consider when comparing floating-point
        values.  ``None`` (default) means exact comparison.

    Example
    -------
    ::

        from mem_bridge.differ import ProfileDiffer

        differ = ProfileDiffer()
        result = differ.compare(profile_a, profile_b)
        differ.print_rich(result)
    """

    def __init__(
        self,
        ignore_order: bool = True,
        ignore_numeric_type_changes: bool = True,
        significant_digits: int | None = None,
    ) -> None:
        self.ignore_order = ignore_order
        self.ignore_numeric_type_changes = ignore_numeric_type_changes
        self.significant_digits = significant_digits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compare(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> DiffResult:
        """Compare *profile_a* (baseline) with *profile_b* (updated).

        Parameters
        ----------
        profile_a:
            Baseline :class:`~mem_bridge.models.MemoryProfile`.
        profile_b:
            Updated :class:`~mem_bridge.models.MemoryProfile`.

        Returns
        -------
        DiffResult
            Structured diff result containing added, removed, and changed
            memories and conversations, plus top-level profile field changes.
        """
        dict_a = profile_a.to_dict()
        dict_b = profile_b.to_dict()

        # Compute raw deepdiff on the full profile dicts
        deepdiff_kwargs: dict[str, Any] = {
            "ignore_order": self.ignore_order,
            "ignore_numeric_type_changes": self.ignore_numeric_type_changes,
        }
        if self.significant_digits is not None:
            deepdiff_kwargs["significant_digits"] = self.significant_digits

        raw = DeepDiff(dict_a, dict_b, **deepdiff_kwargs)
        raw_dict: dict[str, Any] = raw.to_dict()

        # --- Memory diff ---
        added_memories = self._diff_memories_added(profile_a, profile_b)
        removed_memories = self._diff_memories_removed(profile_a, profile_b)
        changed_memories = self._diff_memories_changed(profile_a, profile_b)

        # --- Conversation diff ---
        added_convs = self._diff_conversations_added(profile_a, profile_b)
        removed_convs = self._diff_conversations_removed(profile_a, profile_b)

        # --- Profile-level field diff ---
        profile_changes = self._diff_profile_fields(profile_a, profile_b)

        return DiffResult(
            added_memories=added_memories,
            removed_memories=removed_memories,
            changed_memories=changed_memories,
            added_conversations=added_convs,
            removed_conversations=removed_convs,
            profile_changes=profile_changes,
            raw_diff=raw_dict,
            profile_a_name=profile_a.display_name,
            profile_b_name=profile_b.display_name,
        )

    def render(
        self,
        result: DiffResult,
        fmt: str = "rich",
        console: Console | None = None,
    ) -> str | None:
        """Render a :class:`DiffResult` in the requested format.

        Parameters
        ----------
        result:
            The :class:`DiffResult` to render.
        fmt:
            Output format.  One of ``'rich'``, ``'json'``, or ``'yaml'``.
            Case-insensitive.
        console:
            Optional :class:`rich.console.Console` to use for ``'rich'``
            output.  If ``None``, output is printed to ``stdout``.

        Returns
        -------
        str | None
            For ``'json'`` and ``'yaml'`` formats, returns the rendered string.
            For ``'rich'`` format, prints to the console and returns ``None``
            unless *console* is a :class:`~io.StringIO`-backed console,
            in which case the captured string is returned.

        Raises
        ------
        ValueError
            If *fmt* is not a recognised format string.
        """
        fmt_lower = fmt.lower().strip()

        if fmt_lower == "json":
            return result.to_json()
        elif fmt_lower == "yaml":
            return result.to_yaml()
        elif fmt_lower == "rich":
            return self.print_rich(result, console=console)
        else:
            raise ValueError(
                f"Unsupported diff format {fmt!r}. "
                f"Choose 'rich', 'json', or 'yaml'."
            )

    def print_rich(
        self,
        result: DiffResult,
        console: Console | None = None,
    ) -> str | None:
        """Print a Rich-formatted diff report to *console*.

        Parameters
        ----------
        result:
            The :class:`DiffResult` to render.
        console:
            :class:`rich.console.Console` instance to write to.  If ``None``,
            a new console writing to ``stdout`` is created.

        Returns
        -------
        str | None
            If *console* records output (i.e. has a ``file`` that is a
            :class:`~io.StringIO`), returns the captured string.
            Otherwise returns ``None``.
        """
        capture_output = False
        if console is None:
            console = Console()
        else:
            # Detect if the console is backed by a StringIO for capture
            capture_output = isinstance(getattr(console, "file", None), StringIO)

        self._print_header(result, console)
        self._print_summary_table(result, console)

        if result.profile_changes:
            self._print_profile_changes(result, console)

        if result.added_memories or result.removed_memories or result.changed_memories:
            self._print_memory_diff(result, console)

        if result.added_conversations or result.removed_conversations:
            self._print_conversation_diff(result, console)

        if not result.has_changes:
            console.print(
                Panel(
                    "[bold green]✓ No differences found between the two profiles.[/bold green]",
                    title="Diff Result",
                    border_style="green",
                )
            )

        if capture_output:
            return getattr(console, "file").getvalue()
        return None

    # ------------------------------------------------------------------
    # Memory diff helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_key(entry: MemoryEntry) -> str:
        """Return a stable key for a memory entry (prefer id, fall back to content)."""
        return entry.id if entry.id else entry.content

    def _diff_memories_added(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> list[MemoryEntry]:
        """Return memories present in *profile_b* but not in *profile_a*."""
        keys_a = {self._memory_key(m) for m in profile_a.memories}
        return [
            m for m in profile_b.memories
            if self._memory_key(m) not in keys_a
        ]

    def _diff_memories_removed(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> list[MemoryEntry]:
        """Return memories present in *profile_a* but not in *profile_b*."""
        keys_b = {self._memory_key(m) for m in profile_b.memories}
        return [
            m for m in profile_a.memories
            if self._memory_key(m) not in keys_b
        ]

    def _diff_memories_changed(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> list[tuple[MemoryEntry, MemoryEntry]]:
        """Return (old, new) pairs for memories that share an ID but differ in content.

        Only considers memories that have an explicit ``id`` set, since
        content-keyed memories that differ would appear as add/remove pairs.
        """
        # Build id-keyed index for each profile (only entries with explicit ids)
        index_a: dict[str, MemoryEntry] = {
            m.id: m for m in profile_a.memories if m.id
        }
        index_b: dict[str, MemoryEntry] = {
            m.id: m for m in profile_b.memories if m.id
        }

        changed: list[tuple[MemoryEntry, MemoryEntry]] = []
        for mem_id, entry_a in index_a.items():
            entry_b = index_b.get(mem_id)
            if entry_b is None:
                continue
            # Compare content and tags
            if entry_a.content != entry_b.content or entry_a.tags != entry_b.tags:
                changed.append((entry_a, entry_b))

        return changed

    # ------------------------------------------------------------------
    # Conversation diff helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conversation_key(conv_dict: dict[str, Any]) -> str:
        """Return a stable key for a raw conversation dict."""
        return str(conv_dict.get("id") or conv_dict.get("title") or "")

    def _diff_conversations_added(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> list[dict[str, Any]]:
        """Return conversations in *profile_b* not found in *profile_a*."""
        keys_a = {
            self._conversation_key(c)
            for c in (profile_a.to_dict().get("conversations") or [])
        }
        result: list[dict[str, Any]] = []
        for conv in profile_b.to_dict().get("conversations") or []:
            key = self._conversation_key(conv)
            if key not in keys_a:
                result.append({
                    "id": conv.get("id"),
                    "title": conv.get("title", "Untitled"),
                    "message_count": len(conv.get("entries") or []),
                })
        return result

    def _diff_conversations_removed(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> list[dict[str, Any]]:
        """Return conversations in *profile_a* not found in *profile_b*."""
        keys_b = {
            self._conversation_key(c)
            for c in (profile_b.to_dict().get("conversations") or [])
        }
        result: list[dict[str, Any]] = []
        for conv in profile_a.to_dict().get("conversations") or []:
            key = self._conversation_key(conv)
            if key not in keys_b:
                result.append({
                    "id": conv.get("id"),
                    "title": conv.get("title", "Untitled"),
                    "message_count": len(conv.get("entries") or []),
                })
        return result

    # ------------------------------------------------------------------
    # Profile-level field diff helpers
    # ------------------------------------------------------------------

    _PROFILE_FIELDS_TO_COMPARE = (
        "display_name",
        "source_platform",
        "exported_at",
        "schema_version",
    )

    def _diff_profile_fields(
        self,
        profile_a: MemoryProfile,
        profile_b: MemoryProfile,
    ) -> dict[str, tuple[Any, Any]]:
        """Compare top-level scalar fields between two profiles.

        Returns
        -------
        dict[str, tuple[Any, Any]]
            Mapping of field name to ``(old_value, new_value)`` for each field
            that differs between the profiles.
        """
        changes: dict[str, tuple[Any, Any]] = {}
        dict_a = profile_a.to_dict()
        dict_b = profile_b.to_dict()

        for field in self._PROFILE_FIELDS_TO_COMPARE:
            val_a = dict_a.get(field)
            val_b = dict_b.get(field)
            if val_a != val_b:
                changes[field] = (val_a, val_b)

        return changes

    # ------------------------------------------------------------------
    # Rich rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header(result: DiffResult, console: Console) -> None:
        """Print the diff report header panel."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header_lines = [
            f"  Baseline : [bold cyan]{result.profile_a_name}[/bold cyan]",
            f"  Updated  : [bold magenta]{result.profile_b_name}[/bold magenta]",
            f"  Generated: [dim]{ts}[/dim]",
        ]
        console.print(
            Panel(
                "\n".join(header_lines),
                title="[bold]mem_bridge Diff Report[/bold]",
                border_style="blue",
                padding=(1, 2),
            )
        )

    @staticmethod
    def _print_summary_table(result: DiffResult, console: Console) -> None:
        """Print a summary table of change counts."""
        summary = result.summary
        table = Table(
            title="Change Summary",
            show_header=True,
            header_style="bold white on dark_blue",
            border_style="dim",
            show_lines=True,
        )
        table.add_column("Category", style="bold", width=30)
        table.add_column("Count", justify="right", width=10)
        table.add_column("Status", width=15)

        rows = [
            ("Memories Added",       summary["added_memories"],       "green",   "+ Added"),
            ("Memories Removed",     summary["removed_memories"],     "red",     "− Removed"),
            ("Memories Changed",     summary["changed_memories"],     "yellow",  "~ Modified"),
            ("Conversations Added",  summary["added_conversations"],  "green",   "+ Added"),
            ("Conversations Removed",summary["removed_conversations"],"red",     "− Removed"),
            ("Profile Field Changes",summary["profile_changes"],      "yellow",  "~ Modified"),
        ]
        for label, count, colour, status_label in rows:
            count_str = str(count) if count > 0 else "[dim]0[/dim]"
            status_str = (
                f"[{colour}]{status_label}[/{colour}]" if count > 0 else "[dim]—[/dim]"
            )
            table.add_row(label, count_str, status_str)

        console.print(table)
        console.print()

    @staticmethod
    def _print_profile_changes(
        result: DiffResult, console: Console
    ) -> None:
        """Print a table of top-level profile field changes."""
        table = Table(
            title="[yellow]Profile Field Changes[/yellow]",
            show_header=True,
            header_style="bold yellow",
            border_style="yellow",
            show_lines=True,
        )
        table.add_column("Field", style="bold", width=20)
        table.add_column("Old Value", style="red", width=30)
        table.add_column("New Value", style="green", width=30)

        for field, (old_val, new_val) in result.profile_changes.items():
            table.add_row(
                field,
                str(old_val) if old_val is not None else "[dim]null[/dim]",
                str(new_val) if new_val is not None else "[dim]null[/dim]",
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_memory_diff(
        result: DiffResult, console: Console
    ) -> None:
        """Print a table of memory additions, removals, and modifications."""
        table = Table(
            title="[bold]Memory Diff[/bold]",
            show_header=True,
            header_style="bold white",
            border_style="dim",
            show_lines=True,
            expand=True,
        )
        table.add_column("Change", width=10, justify="center")
        table.add_column("ID", width=15)
        table.add_column("Content", ratio=1)
        table.add_column("Tags", width=20)

        for mem in result.added_memories:
            table.add_row(
                "[bold green]+ ADD[/bold green]",
                mem.id or "[dim]—[/dim]",
                f"[green]{_truncate(mem.content)}[/green]",
                ", ".join(mem.tags) if mem.tags else "[dim]—[/dim]",
            )

        for mem in result.removed_memories:
            table.add_row(
                "[bold red]− DEL[/bold red]",
                mem.id or "[dim]—[/dim]",
                f"[red]{_truncate(mem.content)}[/red]",
                ", ".join(mem.tags) if mem.tags else "[dim]—[/dim]",
            )

        for old_mem, new_mem in result.changed_memories:
            # Show old content in red, new content in green
            old_line = Text()
            old_line.append("OLD: ", style="bold red")
            old_line.append(_truncate(old_mem.content), style="red")

            new_line = Text()
            new_line.append("NEW: ", style="bold green")
            new_line.append(_truncate(new_mem.content), style="green")

            combined = Text()
            combined.append_text(old_line)
            combined.append("\n")
            combined.append_text(new_line)

            old_tags = ", ".join(old_mem.tags) if old_mem.tags else "—"
            new_tags = ", ".join(new_mem.tags) if new_mem.tags else "—"
            tags_str = (
                f"[red]{old_tags}[/red] →\n[green]{new_tags}[/green]"
                if old_tags != new_tags
                else (old_tags or "[dim]—[/dim]")
            )

            table.add_row(
                "[bold yellow]~ MOD[/bold yellow]",
                old_mem.id or "[dim]—[/dim]",
                combined,
                tags_str,
            )

        console.print(table)
        console.print()

    @staticmethod
    def _print_conversation_diff(
        result: DiffResult, console: Console
    ) -> None:
        """Print a table of conversation additions and removals."""
        table = Table(
            title="[bold]Conversation Diff[/bold]",
            show_header=True,
            header_style="bold white",
            border_style="dim",
            show_lines=True,
            expand=True,
        )
        table.add_column("Change", width=10, justify="center")
        table.add_column("ID", width=20)
        table.add_column("Title", ratio=1)
        table.add_column("Messages", width=12, justify="right")

        for conv in result.added_conversations:
            table.add_row(
                "[bold green]+ ADD[/bold green]",
                str(conv.get("id") or "[dim]—[/dim]"),
                f"[green]{_truncate(str(conv.get('title', 'Untitled')))}[/green]",
                str(conv.get("message_count", "?")) ,
            )

        for conv in result.removed_conversations:
            table.add_row(
                "[bold red]− DEL[/bold red]",
                str(conv.get("id") or "[dim]—[/dim]"),
                f"[red]{_truncate(str(conv.get('title', 'Untitled')))}[/red]",
                str(conv.get("message_count", "?")),
            )

        console.print(table)
        console.print()


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _truncate(text: str, length: int = 100, suffix: str = "...") -> str:
    """Truncate *text* to *length* characters, appending *suffix* if trimmed.

    Parameters
    ----------
    text:
        Input string.
    length:
        Maximum character count (including *suffix*).
    suffix:
        String appended when *text* is truncated.

    Returns
    -------
    str
        Truncated string.
    """
    if len(text) <= length:
        return text
    return text[: length - len(suffix)].rstrip() + suffix


# ---------------------------------------------------------------------------
# Module-level default differ and convenience functions
# ---------------------------------------------------------------------------

_default_differ = ProfileDiffer()


def compare(
    profile_a: MemoryProfile,
    profile_b: MemoryProfile,
) -> DiffResult:
    """Compare *profile_a* and *profile_b* and return a :class:`DiffResult`.

    This is a convenience wrapper around :class:`ProfileDiffer` using the
    default configuration.

    Parameters
    ----------
    profile_a:
        Baseline :class:`~mem_bridge.models.MemoryProfile`.
    profile_b:
        Updated :class:`~mem_bridge.models.MemoryProfile`.

    Returns
    -------
    DiffResult
        Structured diff result.
    """
    return _default_differ.compare(profile_a, profile_b)


def diff(
    profile_a: MemoryProfile,
    profile_b: MemoryProfile,
    fmt: str = "rich",
    console: Console | None = None,
) -> str | None:
    """Compare two profiles and render the diff in the requested format.

    This is the primary convenience entry point for the diff engine.

    Parameters
    ----------
    profile_a:
        Baseline :class:`~mem_bridge.models.MemoryProfile`.
    profile_b:
        Updated :class:`~mem_bridge.models.MemoryProfile`.
    fmt:
        Output format.  One of ``'rich'``, ``'json'``, or ``'yaml'``.
        Case-insensitive.  Defaults to ``'rich'``.
    console:
        Optional :class:`rich.console.Console` for Rich output.  If ``None``
        and *fmt* is ``'rich'``, output is printed to ``stdout``.

    Returns
    -------
    str | None
        For ``'json'`` and ``'yaml'`` formats, returns the rendered string.
        For ``'rich'`` format, returns ``None`` (or captured string if
        *console* is backed by a :class:`~io.StringIO`).

    Raises
    ------
    ValueError
        If *fmt* is not a recognised format string.

    Examples
    --------
    ::

        from mem_bridge.differ import diff

        # Print Rich diff to terminal
        diff(profile_a, profile_b, fmt="rich")

        # Get JSON diff string
        json_str = diff(profile_a, profile_b, fmt="json")
    """
    result = _default_differ.compare(profile_a, profile_b)
    return _default_differ.render(result, fmt=fmt, console=console)


__all__ = [
    "DiffResult",
    "ProfileDiffer",
    "compare",
    "diff",
]
