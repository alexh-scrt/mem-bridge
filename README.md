# mem_bridge

> Export, convert, and diff AI memory profiles across ChatGPT, Claude, and Gemini — from a single CLI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

`mem_bridge` is a command-line tool that automates the tedious copy-paste workflow of moving your AI conversation history and memory profiles between platforms. Whether you are migrating from ChatGPT to Gemini, backing up your Claude conversations, or comparing two snapshots of your AI memory state, `mem_bridge` has you covered.

### Supported Platforms

| Platform | Read (import) | Write (export) |
|----------|:---:|:---:|
| **ChatGPT** | ✅ ZIP / JSON | — |
| **Claude** | ✅ JSON | — |
| **Gemini** | — | ✅ Gem Markdown / JSON |

### Output Formats

| Format | Description |
|--------|-------------|
| `markdown` | Structured system prompt (Jinja2 template) |
| `json` | Pretty-printed JSON of the canonical profile |
| `yaml` | YAML representation of the canonical profile |
| `text` | Concise plain-text terminal summary |
| `gemini` | Gemini Gem Markdown instruction document |

---

## Quickstart

### Installation

```bash
# From PyPI (once published)
pip install mem_bridge

# From source
git clone https://github.com/example/mem_bridge.git
cd mem_bridge
pip install -e .
```

### Basic usage

```bash
# Convert a ChatGPT export to a Gemini Gem instruction file
mem_bridge convert --from chatgpt --input export.zip --to gemini --output my_gem.md

# Convert a Claude export to a structured system-prompt markdown file
mem_bridge convert --from claude --input claude_export.json --to markdown --output prompt.md

# Export a ChatGPT archive to a canonical JSON profile (for diffing later)
mem_bridge export --from chatgpt --input export.zip --output profile.json

# Show a formatted summary of a memory profile in the terminal
mem_bridge show --from chatgpt --input export.zip

# Show a Claude export as JSON
mem_bridge show --from claude --input claude.json --format json

# Diff two memory snapshots
mem_bridge diff old_profile.json new_profile.json

# Get a JSON diff report
mem_bridge diff backup_jan.json backup_jun.json --format json --output diff.json

# List all supported input/output formats
mem_bridge list-formats

# List formats with descriptions
mem_bridge list-formats --verbose

# Show the version
mem_bridge --version
```

---

## Commands

### `convert`

Read a memory export from a source platform and write it in a target format.

```
mem_bridge convert [OPTIONS]

Options:
  --from   TEXT  Source platform name (chatgpt, claude)            [required]
  --input  PATH  Path to the source export file or directory       [required]
  --to     TEXT  Target format (gemini, markdown, json, yaml, text)[required]
  --output PATH  Destination file path. If omitted, prints to stdout.
  --help         Show this message and exit.
```

**Examples:**

```bash
# To a Gemini Gem Markdown file
mem_bridge convert --from chatgpt --input ~/Downloads/chatgpt_export.zip \
                   --to gemini --output ~/gems/my_memory.md

# To a JSON file
mem_bridge convert --from claude --input claude.json --to json --output profile.json

# Print markdown to stdout
mem_bridge convert --from chatgpt --input export.zip --to markdown
```

---

### `export`

Parse a platform export and save it as a canonical mem_bridge profile file. Use this to create a snapshot that can later be loaded with `diff` or `show`.

```
mem_bridge export [OPTIONS]

Options:
  --from    TEXT  Source platform name (chatgpt, claude)  [required]
  --input   PATH  Path to the source export file          [required]
  --output  PATH  Destination profile file (.json/.yaml)  [required]
  --format  TEXT  Output format: json or yaml             [default: json]
  --help          Show this message and exit.
```

**Examples:**

```bash
# Export ChatGPT archive to canonical JSON
mem_bridge export --from chatgpt --input export.zip --output profile_jan.json

# Export Claude conversation to YAML
mem_bridge export --from claude --input claude.json --output profile.yaml --format yaml
```

---

### `show`

Display a human-readable summary of a memory profile in the terminal.

```
mem_bridge show [OPTIONS]

Options:
  --from    TEXT  Source platform (chatgpt, claude)         [required]
  --input   PATH  Path to the source export file            [required]
  --format  TEXT  Display format: table, json, yaml, text   [default: table]
  --help          Show this message and exit.
```

**Examples:**

```bash
# Rich table summary (default)
mem_bridge show --from chatgpt --input export.zip

# JSON output
mem_bridge show --from claude --input claude.json --format json

# Plain text summary
mem_bridge show --from chatgpt --input export.zip --format text
```

---

### `diff`

Compare two memory profiles and display a Rich-formatted diff highlighting added, removed, and changed memory entries and conversations.

Both files must be canonical mem_bridge profile files (JSON or YAML), as produced by the `export` or `convert --to json` commands.

```
mem_bridge diff [OPTIONS] FILE_A FILE_B

Arguments:
  FILE_A  Path to the baseline (old) memory profile JSON/YAML.
  FILE_B  Path to the updated (new) memory profile JSON/YAML.

Options:
  --format  TEXT  Output format: rich, json, yaml   [default: rich]
  --output  PATH  Write diff report to this file instead of stdout.
  --help          Show this message and exit.
```

**Examples:**

```bash
# Rich terminal diff
mem_bridge diff backup_jan.json backup_jun.json

# JSON diff report saved to file
mem_bridge diff old.json new.json --format json --output diff_report.json

# YAML diff report
mem_bridge diff old.json new.json --format yaml
```

---

### `list-formats`

Print all registered reader and writer adapters and supported output formats.

```
mem_bridge list-formats [OPTIONS]

Options:
  --verbose  -v  Show descriptions for each adapter and format.
  --help         Show this message and exit.
```

**Examples:**

```bash
mem_bridge list-formats
mem_bridge list-formats --verbose
```

Sample output:

```
                Registered Adapters
 ──────────────────────────────────────────
  Mode    │ Platform
 ─────────┼────────────────────────────────
  Read    │ chatgpt
  Read    │ claude
  Write   │ gemini
 ──────────────────────────────────────────

          Output Formats
 ──────────────────────────────
  Format
 ──────────────────────────────
  gemini
  json
  markdown
  text
  yaml
 ──────────────────────────────
```

---

## Typical Workflow

### Migrate ChatGPT memory to Gemini

```bash
# 1. Download your ChatGPT data export (Settings → Data controls → Export data)
# 2. Convert to a Gemini Gem instruction file
mem_bridge convert --from chatgpt --input ~/Downloads/chatgpt_export.zip \
                   --to gemini --output ~/gems/my_memory.md

# 3. Paste the contents of my_memory.md into a Gemini Gem's instruction field
```

### Create periodic snapshots and diff them

```bash
# Save a baseline snapshot
mem_bridge export --from chatgpt --input export_jan.zip --output snapshot_jan.json

# Six months later, save another snapshot
mem_bridge export --from chatgpt --input export_jun.zip --output snapshot_jun.json

# Compare the two snapshots
mem_bridge diff snapshot_jan.json snapshot_jun.json
```

### Convert Claude export to a system prompt for any LLM API

```bash
mem_bridge convert --from claude --input claude_export.json \
                   --to markdown --output system_prompt.md

# Use system_prompt.md as the system message in your LLM API call
```

---

## Export Formats

### ChatGPT Export

Request your data export from **Settings → Data controls → Export data** in ChatGPT. You will receive a `.zip` file containing:

- `conversations.json` – full conversation history
- `memory.json` – your saved memories (if Memory is enabled)

Pass the `.zip` file (or the extracted directory, or either JSON file directly) to `mem_bridge`.

```bash
# All of these work:
mem_bridge show --from chatgpt --input chatgpt_export.zip
mem_bridge show --from chatgpt --input /path/to/extracted/
mem_bridge show --from chatgpt --input conversations.json
mem_bridge show --from chatgpt --input memory.json
```

### Claude Export

Claude exports conversations as JSON via **Settings → Privacy → Export data**. Pass the downloaded `.json` file to `mem_bridge`.

```bash
mem_bridge show --from claude --input conversations.json
```

### Gemini Output

`mem_bridge` writes Gemini-compatible output in two sub-formats:

1. **Gem Markdown** (`.md`) – A structured system-prompt you can paste directly into a Gem's instruction field in Google Gemini AI Studio.
2. **JSON** (`.json`) – A machine-readable import file with a `gem_format_version` schema.

```bash
# Gem Markdown (default for .md extension)
mem_bridge convert --from chatgpt --input export.zip --to gemini --output my_gem.md

# JSON format
mem_bridge convert --from chatgpt --input export.zip --to gemini --output my_gem.json
```

---

## Architecture

```
mem_bridge/
├── cli.py            ← Typer CLI entry point (all commands)
├── models.py         ← Canonical Pydantic data models
├── formatters.py     ← Jinja2-based output renderers
├── differ.py         ← deepdiff-powered diff engine
└── adapters/
    ├── __init__.py   ← Adapter registry
    ├── chatgpt.py    ← ChatGPT reader adapter
    ├── claude.py     ← Claude reader adapter
    └── gemini.py     ← Gemini writer adapter
templates/
└── prompt.md.j2      ← Jinja2 template for Markdown output
```

### Internal Data Flow

```
Platform export file
        │
        ▼
  Reader Adapter          (chatgpt.py / claude.py)
        │   .read(path) → MemoryProfile
        ▼
  MemoryProfile           (models.py)  ← canonical interchange format
        │
        ├──────────────────────────────────────────────────────┐
        │                                                       │
        ▼                                                       ▼
  Writer Adapter          (gemini.py)       Output Formatter   (formatters.py)
  .write(profile, dest)                     .render(profile, fmt)
        │                                         │
        ▼                                         ▼
  Gemini .md / .json                  markdown / json / yaml / text
```

The `MemoryProfile` model (defined in `models.py`) is the **single internal interchange format**. All readers produce a `MemoryProfile`; all writers and formatters consume one. Adding a new platform requires only implementing the appropriate reader or writer class.

---

## Extending mem_bridge

### Adding a Custom Reader Adapter

```python
# my_provider.py
from pathlib import Path
from mem_bridge.models import MemoryProfile, SourcePlatform

class MyproviderAdapter:
    def read(self, source: str | Path) -> MemoryProfile:
        path = Path(source)
        # ... parse the file ...
        return MemoryProfile(
            source_platform=SourcePlatform.UNKNOWN,
            display_name="My User",
            memories=[],
            conversations=[],
        )
```

```python
# Register at runtime
from mem_bridge.adapters import register_reader
register_reader("myprovider", "my_provider")

# Now use it
from mem_bridge.adapters import get_adapter
adapter = get_adapter("myprovider", mode="read")
profile = adapter.read("my_export.json")
```

### Adding a Custom Writer Adapter

```python
# my_writer.py
from pathlib import Path
from mem_bridge.models import MemoryProfile

class MytargetAdapter:
    def write(self, profile: MemoryProfile, dest: str | Path, fmt=None) -> None:
        Path(dest).write_text(f"# {profile.display_name}\n", encoding="utf-8")
```

```python
from mem_bridge.adapters import register_writer
register_writer("mytarget", "my_writer")
```

### Using a Custom Jinja2 Template

Place a `prompt.md.j2` file in any directory and pass it to the `Formatter`:

```python
from mem_bridge.formatters import Formatter
from mem_bridge.models import MemoryProfile

formatter = Formatter(templates_dir="/path/to/my/templates")
output = formatter.render(profile, fmt="markdown")
```

The template receives a `profile` variable of type `MemoryProfile`, plus helper
functions `wrap`, `truncate`, `format_dt`, and `role_label`.

---

## Data Models

### `MemoryProfile` (top-level container)

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Optional opaque identifier |
| `source_platform` | `SourcePlatform` | `chatgpt`, `claude`, `gemini`, `unknown` |
| `display_name` | `str` | Human-readable label for this profile |
| `exported_at` | `datetime \| None` | When the export was generated |
| `memories` | `list[MemoryEntry]` | List of atomic memory facts |
| `conversations` | `list[Conversation]` | List of full conversation threads |
| `schema_version` | `str` | mem_bridge internal schema version |
| `metadata` | `dict` | Extra source-specific fields |

### `MemoryEntry`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Platform-assigned identifier |
| `content` | `str` | Text of the memory fact |
| `created_at` | `datetime \| None` | Creation timestamp |
| `updated_at` | `datetime \| None` | Last-updated timestamp |
| `tags` | `list[str]` | Topic tags |
| `metadata` | `dict` | Extra source-specific fields |

### `Conversation`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Platform-assigned identifier |
| `title` | `str` | Display title |
| `created_at` | `datetime \| None` | Conversation start time |
| `updated_at` | `datetime \| None` | Last message time |
| `entries` | `list[ConversationEntry]` | Ordered message turns |
| `model` | `str \| None` | Primary AI model used |
| `metadata` | `dict` | Extra source-specific fields |

### `ConversationEntry`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str \| None` | Platform-assigned message ID |
| `role` | `Role` | `user`, `assistant`, `system`, `tool`, `unknown` |
| `content` | `str` | Text content of the message |
| `timestamp` | `datetime \| None` | Message creation timestamp |
| `model` | `str \| None` | AI model identifier for assistant turns |
| `metadata` | `dict` | Extra source-specific fields |

---

## Using mem_bridge as a Library

```python
from mem_bridge.adapters import get_adapter
from mem_bridge.formatters import render
from mem_bridge.differ import diff
from mem_bridge.models import MemoryProfile

# Read a ChatGPT export
adapter = get_adapter("chatgpt", mode="read")
profile = adapter.read("chatgpt_export.zip")

print(f"Loaded {profile.memory_count} memories, {profile.conversation_count} conversations")

# Render as markdown
markdown = render(profile, fmt="markdown")

# Render as JSON
json_str = render(profile, fmt="json")

# Save canonical profile
profile.save("profile.json", fmt="json")

# Load a saved profile
profile_b = MemoryProfile.from_file("profile_new.json")

# Diff two profiles
json_diff = diff(profile, profile_b, fmt="json")
```

---

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/example/mem_bridge.git
cd mem_bridge
pip install -e .

# Run the test suite
pytest

# Run the test suite with coverage
pytest --tb=short -q

# Run linting (requires ruff)
ruff check mem_bridge tests

# Run a quick smoke test
mem_bridge --version
mem_bridge list-formats
```

### Project structure

```
mem_bridge/
├── __init__.py           # Package version and metadata
├── cli.py                # Typer CLI commands
├── models.py             # Canonical Pydantic models
├── formatters.py         # Output formatters (Jinja2, JSON, YAML, text)
├── differ.py             # Diff engine (deepdiff + Rich)
└── adapters/
    ├── __init__.py       # Adapter registry
    ├── chatgpt.py        # ChatGPT reader
    ├── claude.py         # Claude reader
    └── gemini.py         # Gemini writer
templates/
└── prompt.md.j2          # Default Markdown prompt template
tests/
├── fixtures/
│   ├── chatgpt_export.json
│   └── claude_export.json
├── test_models.py
├── test_adapters.py
├── test_formatters.py
└── test_differ.py
pyproject.toml
README.md
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-new-adapter`
3. Implement your changes and add tests.
4. Ensure all tests pass: `pytest`
5. Open a pull request.

---

## License

MIT © mem_bridge contributors
