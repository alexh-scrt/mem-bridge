"""Typer-based CLI entry point for mem_bridge.

This module defines all user-facing commands:

- ``convert``     – Read a memory export from a source platform and write it
                    in a target format.
- ``show``        – Display a human-readable summary of a memory profile.
- ``diff``        – Compare two memory profile JSON files side-by-side.
- ``list-formats``– Print all registered reader/writer adapters and output
                    formats.
- ``export``      – Export a memory profile to a file in the specified format.

Typical usage::

    mem_bridge convert --from chatgpt --input export.zip --to gemini --output gem.md
    mem_bridge show --from claude --input claude.json
    mem_bridge diff old.json new.json
    mem_bridge list-formats
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mem_bridge import __version__
from mem_bridge.adapters import (
    AdapterNotFoundError,
    get_adapter,
    get_adapter_info,
    list_adapters,
    list_platforms,
)
from mem_bridge.differ import ProfileDiffer, diff as run_diff
from mem_bridge.formatters import (
    Formatter,
    get_format_description,
    list_formats,
    render,
    render_to_file,
)
from mem_bridge.models import MemoryProfile, OutputFormat, SourcePlatform


# ---------------------------------------------------------------------------
# Typer app setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="mem_bridge",
    help=(
        "Export, convert, and diff AI memory profiles across "
        "ChatGPT, Claude, and Gemini — from a single CLI."
    ),
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True, style="bold red")


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print the mem_bridge version and exit."""
    if value:
        console.print(f"mem_bridge version [bold cyan]{__version__}[/bold cyan]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show the mem_bridge version and exit.",
        ),
    ] = None,
) -> None:
    """mem_bridge – AI memory profile converter and manager."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _read_profile(platform: str, input_path: Path) -> MemoryProfile:
    """Read a MemoryProfile from *input_path* using the named *platform* adapter.

    Parameters
    ----------
    platform:
        Platform name (e.g. ``'chatgpt'``, ``'claude'``).
    input_path:
        Path to the export file or directory.

    Returns
    -------
    MemoryProfile
        The parsed canonical profile.

    Raises
    ------
    typer.Exit
        On any error, a formatted error message is printed and the process
        exits with code 1.
    """
    try:
        adapter = get_adapter(platform, mode="read")
    except AdapterNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        _print_available_readers()
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not input_path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Input path does not exist: {input_path}"
        )
        raise typer.Exit(code=1)

    try:
        profile = adapter.read(input_path)
    except FileNotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] File not found: {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            f"[bold red]Error:[/bold red] Failed to parse {platform!r} export: {exc}"
        )
        raise typer.Exit(code=1) from exc

    return profile


def _load_profile_from_json(path: Path) -> MemoryProfile:
    """Load a :class:`MemoryProfile` from a JSON or YAML file.

    Parameters
    ----------
    path:
        Path to the JSON/YAML profile file.

    Returns
    -------
    MemoryProfile
        The loaded profile.

    Raises
    ------
    typer.Exit
        On any error.
    """
    if not path.exists():
        err_console.print(
            f"[bold red]Error:[/bold red] Profile file not found: {path}"
        )
        raise typer.Exit(code=1)
    try:
        return MemoryProfile.from_file(path)
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            f"[bold red]Error:[/bold red] Could not load profile from {path}: {exc}"
        )
        raise typer.Exit(code=1) from exc


def _print_available_readers() -> None:
    """Print a hint listing available reader platforms."""
    adapters = list_adapters()
    readers = ", ".join(adapters["readers"]) or "(none)"
    console.print(f"  Available reader platforms: [cyan]{readers}[/cyan]")


def _print_available_writers() -> None:
    """Print a hint listing available writer platforms."""
    adapters = list_adapters()
    writers = ", ".join(adapters["writers"]) or "(none)"
    console.print(f"  Available writer platforms: [cyan]{writers}[/cyan]")


def _print_available_formats() -> None:
    """Print a hint listing available output formats."""
    fmts = ", ".join(list_formats())
    console.print(f"  Available output formats: [cyan]{fmts}[/cyan]")


# ---------------------------------------------------------------------------
# convert command
# ---------------------------------------------------------------------------


@app.command("convert")
def cmd_convert(
    from_platform: Annotated[
        str,
        typer.Option(
            "--from",
            help="Source platform name (e.g. chatgpt, claude).",
            show_default=False,
        ),
    ],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the source export file or directory.",
            show_default=False,
        ),
    ],
    to_format: Annotated[
        str,
        typer.Option(
            "--to",
            help=(
                "Target output format or platform "
                "(e.g. gemini, markdown, json, yaml, text)."
            ),
            show_default=False,
        ),
    ],
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Destination file path. If omitted, output is printed to stdout.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """[bold]Convert[/bold] a memory export to a different format or platform.

    Read a memory export from a source platform (ChatGPT, Claude) and write
    it in a target format (Gemini, Markdown, JSON, YAML, plain text).

    \b
    Examples:
      mem_bridge convert --from chatgpt --input export.zip --to gemini --output gem.md
      mem_bridge convert --from claude --input claude.json --to markdown
      mem_bridge convert --from chatgpt --input export.zip --to json --output profile.json
    """
    from_platform = from_platform.lower().strip()
    to_format = to_format.lower().strip()

    # Step 1: Read the source profile
    console.print(
        f"[dim]Reading [cyan]{from_platform}[/cyan] export from [white]{input_path}[/white]...[/dim]"
    )
    profile = _read_profile(from_platform, input_path)
    console.print(
        f"[green]✓[/green] Loaded profile: "
        f"[bold]{profile.display_name}[/bold] "
        f"({profile.memory_count} memories, {profile.conversation_count} conversations)"
    )

    # Step 2: Render / write output
    # Determine whether the target is a platform writer or a format renderer
    adapters = list_adapters()
    is_writer_platform = to_format in adapters["writers"]
    is_formatter = to_format in list_formats() or to_format in ("md",)

    if not is_writer_platform and not is_formatter:
        err_console.print(
            f"[bold red]Error:[/bold red] Unknown target format or platform: {to_format!r}"
        )
        _print_available_writers()
        _print_available_formats()
        raise typer.Exit(code=1)

    if output_path is not None:
        # Write to file
        try:
            if is_writer_platform:
                # Use the platform's dedicated writer adapter
                writer = get_adapter(to_format, mode="write")
                writer.write(profile, output_path)
            else:
                # Use the formatter
                render_to_file(profile, output_path, fmt=to_format)
        except Exception as exc:  # noqa: BLE001
            err_console.print(
                f"[bold red]Error:[/bold red] Failed to write output: {exc}"
            )
            raise typer.Exit(code=1) from exc

        console.print(
            f"[green]✓[/green] Output written to [bold white]{output_path}[/bold white]"
        )
    else:
        # Print to stdout
        try:
            if is_writer_platform:
                writer = get_adapter(to_format, mode="write")
                content = writer.render(profile, fmt="markdown")
            else:
                content = render(profile, fmt=to_format)
        except Exception as exc:  # noqa: BLE001
            err_console.print(
                f"[bold red]Error:[/bold red] Failed to render output: {exc}"
            )
            raise typer.Exit(code=1) from exc

        # Print directly without rich markup interpretation
        print(content)  # noqa: T201


# ---------------------------------------------------------------------------
# show command
# ---------------------------------------------------------------------------


@app.command("show")
def cmd_show(
    from_platform: Annotated[
        str,
        typer.Option(
            "--from",
            help="Source platform name (e.g. chatgpt, claude).",
            show_default=False,
        ),
    ],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the source export file or directory.",
            show_default=False,
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Display format: table, json, yaml, text, markdown.",
        ),
    ] = "table",
) -> None:
    """[bold]Show[/bold] a human-readable summary of a memory profile.

    Display the contents of a memory export in the terminal.

    \b
    Examples:
      mem_bridge show --from chatgpt --input export.zip
      mem_bridge show --from claude --input claude.json --format json
      mem_bridge show --from chatgpt --input export.zip --format yaml
    """
    from_platform = from_platform.lower().strip()
    fmt = fmt.lower().strip()

    console.print(
        f"[dim]Loading [cyan]{from_platform}[/cyan] export from [white]{input_path}[/white]...[/dim]"
    )
    profile = _read_profile(from_platform, input_path)

    if fmt == "table":
        _show_table(profile)
    elif fmt in ("json", "yaml", "text", "markdown", "md", "gemini"):
        try:
            content = render(profile, fmt=fmt)
        except ValueError as exc:
            err_console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        print(content)  # noqa: T201
    else:
        err_console.print(
            f"[bold red]Error:[/bold red] Unknown display format: {fmt!r}\n"
            f"  Use: table, json, yaml, text, markdown"
        )
        raise typer.Exit(code=1)


def _show_table(profile: MemoryProfile) -> None:
    """Render a Rich table summary for the given profile."""
    # Header panel
    header = (
        f"  [bold]Profile:[/bold] {profile.display_name}\n"
        f"  [bold]Platform:[/bold] {profile.source_platform.value}\n"
        f"  [bold]Exported:[/bold] "
        + (
            profile.exported_at.strftime("%Y-%m-%d %H:%M UTC")
            if profile.exported_at
            else "unknown"
        )
        + "\n"
        f"  [bold]Schema:[/bold] {profile.schema_version}\n"
        f"  [bold]Memories:[/bold] {profile.memory_count}  "
        f"[bold]Conversations:[/bold] {profile.conversation_count}  "
        f"[bold]Messages:[/bold] {profile.total_message_count}"
    )
    console.print(
        Panel(header, title="[bold blue]mem_bridge Profile Summary[/bold blue]", border_style="blue")
    )

    # Memories table
    if profile.memories:
        mem_table = Table(
            title="[bold]Remembered Facts[/bold]",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            show_lines=True,
            expand=True,
        )
        mem_table.add_column("#", width=4, justify="right")
        mem_table.add_column("ID", width=16)
        mem_table.add_column("Content", ratio=2)
        mem_table.add_column("Tags", width=25)
        mem_table.add_column("Created", width=12)

        for i, mem in enumerate(profile.memories, start=1):
            tags_str = ", ".join(mem.tags) if mem.tags else "[dim]—[/dim]"
            created_str = (
                mem.created_at.strftime("%Y-%m-%d") if mem.created_at else "[dim]—[/dim]"
            )
            mem_table.add_row(
                str(i),
                mem.id or "[dim]—[/dim]",
                _truncate_str(mem.content, 120),
                tags_str,
                created_str,
            )
        console.print(mem_table)
    else:
        console.print(Panel("[dim]No memory facts in this profile.[/dim]", border_style="dim"))

    # Conversations table
    if profile.conversations:
        conv_table = Table(
            title="[bold]Conversations[/bold]",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
            show_lines=True,
            expand=True,
        )
        conv_table.add_column("#", width=4, justify="right")
        conv_table.add_column("ID", width=18)
        conv_table.add_column("Title", ratio=2)
        conv_table.add_column("Model", width=16)
        conv_table.add_column("Msgs", width=6, justify="right")
        conv_table.add_column("Date", width=12)

        for i, conv in enumerate(profile.conversations, start=1):
            date_str = (
                conv.created_at.strftime("%Y-%m-%d") if conv.created_at else "[dim]—[/dim]"
            )
            conv_table.add_row(
                str(i),
                _truncate_str(conv.id or "", 16) or "[dim]—[/dim]",
                _truncate_str(conv.title, 60),
                conv.model or "[dim]—[/dim]",
                str(conv.message_count),
                date_str,
            )
        console.print(conv_table)
    else:
        console.print(
            Panel("[dim]No conversations in this profile.[/dim]", border_style="dim")
        )


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------


@app.command("diff")
def cmd_diff(
    file_a: Annotated[
        Path,
        typer.Argument(
            help="Path to the baseline (old) memory profile JSON/YAML file.",
            show_default=False,
        ),
    ],
    file_b: Annotated[
        Path,
        typer.Argument(
            help="Path to the updated (new) memory profile JSON/YAML file.",
            show_default=False,
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: rich, json, yaml.",
        ),
    ] = "rich",
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Write diff output to this file instead of stdout.",
            show_default=False,
        ),
    ] = None,
) -> None:
    """[bold]Diff[/bold] two memory profile snapshots.

    Compare FILE_A (baseline) against FILE_B (updated) and display a
    Rich-formatted diff highlighting added, removed, and changed memory
    entries and conversations.

    Both files must be mem_bridge JSON or YAML profile files (produced by
    the [cyan]convert[/cyan] or [cyan]export[/cyan] commands with
    [cyan]--to json[/cyan] / [cyan]--to yaml[/cyan]).

    \b
    Examples:
      mem_bridge diff old_profile.json new_profile.json
      mem_bridge diff backup_jan.json backup_jun.json --format json
      mem_bridge diff a.json b.json --format yaml --output diff_report.yaml
    """
    fmt = fmt.lower().strip()

    profile_a = _load_profile_from_json(file_a)
    profile_b = _load_profile_from_json(file_b)

    if fmt not in ("rich", "json", "yaml"):
        err_console.print(
            f"[bold red]Error:[/bold red] Unknown diff format: {fmt!r}\n"
            f"  Use: rich, json, yaml"
        )
        raise typer.Exit(code=1)

    if fmt == "rich" and output_path is None:
        # Render directly to console
        try:
            run_diff(profile_a, profile_b, fmt="rich", console=console)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]Error:[/bold red] Diff failed: {exc}")
            raise typer.Exit(code=1) from exc
    else:
        # Produce a string result
        try:
            if fmt == "rich":
                from io import StringIO
                buf = StringIO()
                cap_console = Console(file=buf, highlight=False, markup=True)
                run_diff(profile_a, profile_b, fmt="rich", console=cap_console)
                content = buf.getvalue()
            else:
                content = run_diff(profile_a, profile_b, fmt=fmt)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[bold red]Error:[/bold red] Diff failed: {exc}")
            raise typer.Exit(code=1) from exc

        if output_path is not None:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
                console.print(
                    f"[green]✓[/green] Diff report written to [bold white]{output_path}[/bold white]"
                )
            except OSError as exc:
                err_console.print(
                    f"[bold red]Error:[/bold red] Could not write diff output: {exc}"
                )
                raise typer.Exit(code=1) from exc
        else:
            print(content)  # noqa: T201


# ---------------------------------------------------------------------------
# list-formats command
# ---------------------------------------------------------------------------


@app.command("list-formats")
def cmd_list_formats(
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show descriptions for each adapter and format.",
        ),
    ] = False,
) -> None:
    """[bold]List[/bold] all registered adapters and supported output formats.

    \b
    Examples:
      mem_bridge list-formats
      mem_bridge list-formats --verbose
    """
    adapters = list_adapters()
    fmts = list_formats()

    # --- Adapters table ---
    adapter_table = Table(
        title="[bold]Registered Adapters[/bold]",
        show_header=True,
        header_style="bold white on dark_blue",
        border_style="blue",
        show_lines=True,
    )
    adapter_table.add_column("Mode", width=10)
    adapter_table.add_column("Platform", width=16)
    if verbose:
        adapter_table.add_column("Description", ratio=2)
        adapter_table.add_column("File Formats", width=28)

    for platform in adapters["readers"]:
        if verbose:
            try:
                info = get_adapter_info(platform)
                adapter_table.add_row(
                    "[green]Read[/green]",
                    f"[cyan]{platform}[/cyan]",
                    info.get("description", ""),
                    info.get("formats", ""),
                )
            except Exception:  # noqa: BLE001
                adapter_table.add_row("[green]Read[/green]", f"[cyan]{platform}[/cyan]", "", "")
        else:
            adapter_table.add_row("[green]Read[/green]", f"[cyan]{platform}[/cyan]")

    for platform in adapters["writers"]:
        if verbose:
            try:
                info = get_adapter_info(platform)
                adapter_table.add_row(
                    "[magenta]Write[/magenta]",
                    f"[yellow]{platform}[/yellow]",
                    info.get("description", ""),
                    info.get("formats", ""),
                )
            except Exception:  # noqa: BLE001
                adapter_table.add_row(
                    "[magenta]Write[/magenta]", f"[yellow]{platform}[/yellow]", "", ""
                )
        else:
            adapter_table.add_row("[magenta]Write[/magenta]", f"[yellow]{platform}[/yellow]")

    console.print(adapter_table)
    console.print()

    # --- Output formats table ---
    fmt_table = Table(
        title="[bold]Output Formats[/bold]",
        show_header=True,
        header_style="bold white on dark_green",
        border_style="green",
        show_lines=True,
    )
    fmt_table.add_column("Format", width=16)
    if verbose:
        fmt_table.add_column("Description", ratio=1)

    for fmt in fmts:
        if verbose:
            try:
                desc = get_format_description(fmt)
            except ValueError:
                desc = ""
            fmt_table.add_row(f"[green]{fmt}[/green]", desc)
        else:
            fmt_table.add_row(f"[green]{fmt}[/green]")

    console.print(fmt_table)
    console.print()

    # Quick-start hint
    console.print(
        Panel(
            "[dim]Quick start:[/dim]\n"
            "  [cyan]mem_bridge convert --from chatgpt --input export.zip "
            "--to gemini --output gem.md[/cyan]\n"
            "  [cyan]mem_bridge diff old.json new.json[/cyan]\n"
            "  [cyan]mem_bridge show --from claude --input claude.json[/cyan]",
            title="[bold]Usage Examples[/bold]",
            border_style="dim",
        )
    )


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


@app.command("export")
def cmd_export(
    from_platform: Annotated[
        str,
        typer.Option(
            "--from",
            help="Source platform name (e.g. chatgpt, claude).",
            show_default=False,
        ),
    ],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Path to the source export file or directory.",
            show_default=False,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Destination file path (.json or .yaml).",
            show_default=False,
        ),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: json or yaml.",
        ),
    ] = "json",
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty/--compact",
            help="Pretty-print JSON output (default: True).",
        ),
    ] = True,
) -> None:
    """[bold]Export[/bold] a raw platform export to a canonical mem_bridge profile file.

    Reads a platform export and saves it as a canonical mem_bridge JSON or YAML
    profile that can later be diffed, shown, or converted.

    \b
    Examples:
      mem_bridge export --from chatgpt --input export.zip --output profile.json
      mem_bridge export --from claude --input claude.json --output profile.yaml --format yaml
    """
    from_platform = from_platform.lower().strip()
    fmt = fmt.lower().strip()

    if fmt not in ("json", "yaml", "yml"):
        err_console.print(
            f"[bold red]Error:[/bold red] Unsupported export format {fmt!r}.\n"
            f"  Use: json, yaml"
        )
        raise typer.Exit(code=1)

    console.print(
        f"[dim]Reading [cyan]{from_platform}[/cyan] export from [white]{input_path}[/white]...[/dim]"
    )
    profile = _read_profile(from_platform, input_path)
    console.print(
        f"[green]✓[/green] Parsed profile: "
        f"[bold]{profile.display_name}[/bold] "
        f"({profile.memory_count} memories, {profile.conversation_count} conversations)"
    )

    try:
        profile.save(output_path, fmt=fmt)
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            f"[bold red]Error:[/bold red] Could not save profile to {output_path}: {exc}"
        )
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]✓[/green] Profile saved to [bold white]{output_path}[/bold white] "
        f"([cyan]{fmt}[/cyan])"
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _truncate_str(text: str, length: int = 60, suffix: str = "...") -> str:
    """Truncate *text* to *length* characters with *suffix* if trimmed."""
    if len(text) <= length:
        return text
    return text[: length - len(suffix)].rstrip() + suffix


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
