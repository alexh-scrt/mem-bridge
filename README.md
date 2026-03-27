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
mem_bridge convert --from chatgpt export.zip --to gemini --output my_gem.md

# Convert a Claude export to a structured system-prompt markdown file
mem_bridge convert --from claude claude_export.json --to markdown --output prompt.md

# Show a formatted summary of a memory profile
mem_bridge show --from chatgpt export.zip

# Diff two memory snapshots
mem_bridge diff old_profile.json new_profile.json

# List all supported input/output formats
mem_bridge list-formats
```

---

## Commands

### `convert`

Read a memory export from a source platform and write it in a target format.

```
mem_bridge convert [OPTIONS]

Options:
  --from   TEXT  Source platform name (chatgpt, claude)       [required]
  --input  PATH  Path to the source export file               [required]
  --to     TEXT  Target format (gemini, markdown, json, yaml) [required]
  --output PATH  Destination file path (default: stdout)
```

**Example:**

```bash
mem_bridge convert --from chatgpt --input ~/Downloads/chatgpt_export.zip \
                   --to gemini --output ~/gems/my_memory.md
```

---

### `show`

Display a human-readable summary of a memory profile in the terminal.

```
mem_bridge show [OPTIONS]

Options:
  --from   TEXT  Source platform (chatgpt, claude) [required]
  --input  PATH  Path to the source export file    [required]
  --format TEXT  Output format: table, json, yaml  [default: table]
```

---

### `diff`

Compare two memory profiles and display a Rich-formatted diff highlighting
added, removed, and changed memory entries.

```
mem_bridge diff [OPTIONS] FILE_A FILE_B

Arguments:
  FILE_A  Path to the first (baseline) memory profile JSON.
  FILE_B  Path to the second (updated) memory profile JSON.

Options:
  --format TEXT  Output format: rich, json, yaml [default: rich]
```

**Example:**

```bash
mem_bridge diff backup_2024_01.json backup_2024_06.json
```

---

### `list-formats`

Print all registered reader and writer adapters.

```bash
mem_bridge list-formats
```

Sample output:

```
╭──────────────────────────────────────╮
│          mem_bridge Adapters         │
├──────────────┬───────────────────────┤
│ Mode         │ Platforms             │
├──────────────┼───────────────────────┤
│ Readers      │ chatgpt, claude       │
│ Writers      │ gemini                │
╰──────────────┴───────────────────────╯
```

---

## Export Formats

### ChatGPT Export

Request your data export from **Settings → Data controls → Export data** in ChatGPT. You will receive a `.zip` file containing:

- `conversations.json` – full conversation history
- `memory.json` – your saved memories (if Memory is enabled)

Pass the `.zip` file (or the extracted directory) directly to `mem_bridge`.

### Claude Export

Claude exports conversations as JSON via **Settings → Export data**. Pass the downloaded `.json` file to `mem_bridge`.

### Gemini Output

`mem_bridge` writes Gemini-compatible output in two sub-formats:

1. **Gem Markdown** (`.md`) – A structured system-prompt you can paste into a Gem's instruction field.
2. **JSON** (`.json`) – A machine-readable import file for programmatic use.

---

## Architecture

```
mem_bridge/
├── cli.py            ← Typer CLI entry point
├── models.py         ← Canonical Pydantic data models
├── formatters.py     ← Jinja2-based output renderers
├── differ.py         ← deepdiff-powered diff engine
└── adapters/
    ├── __init__.py   ← Adapter registry
    ├── chatgpt.py    ← ChatGPT reader
    ├── claude.py     ← Claude reader
    └── gemini.py     ← Gemini writer
```

The internal interchange format is `MemoryProfile` (defined in `models.py`). All readers produce a `MemoryProfile`; all writers and formatters consume one. Adding a new platform requires only implementing the appropriate reader or writer class.

### Adding a Custom Adapter

```python
# my_provider.py
from pathlib import Path
from mem_bridge.models import MemoryProfile

class MyproviderAdapter:
    def read(self, source: str | Path) -> MemoryProfile:
        ...

# Register at runtime
from mem_bridge.adapters import register_reader
register_reader("myprovider", "my_provider")
```

---

## Development

```bash
# Clone and install in editable mode with dev extras
git clone https://github.com/example/mem_bridge.git
cd mem_bridge
pip install -e ".[dev]"

# Run the test suite
pytest

# Run linting
ruff check mem_bridge tests
```

---

## License

MIT © mem_bridge contributors
