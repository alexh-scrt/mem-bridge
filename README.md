# mem_bridge

> Port your AI memory anywhere — export, convert, and diff profiles across ChatGPT, Claude, and Gemini from a single CLI.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

`mem_bridge` is a command-line tool that exports conversation history and memory profiles from AI assistants like ChatGPT and Claude, then reformats them into structured prompts or import files compatible with other AI platforms. It eliminates the tedious copy-paste workflow of migrating your AI context between providers. With a single command you can convert, diff, and manage your AI memory states across ChatGPT, Claude, and Gemini using a clean CLI and an extensible adapter architecture.

---

## Quick Start

**Install**

```bash
pip install mem_bridge
```

Or install from source:

```bash
git clone https://github.com/your-org/mem_bridge.git
cd mem_bridge
pip install -e .
```

**Convert a ChatGPT export to a Gemini Gem file**

```bash
mem_bridge convert --from chatgpt --input chatgpt_export.zip --to gemini --output my_gem.md
```

**Show a summary of a Claude export**

```bash
mem_bridge show --from claude --input claude_export.json
```

---

## Features

- **Multi-platform ingestion** — Parse ChatGPT data-export ZIP/JSON files and Claude conversation exports into a unified canonical `MemoryProfile` format.
- **Flexible output formats** — Convert profiles to structured system prompts (Markdown), JSON, YAML, or Gemini Gem-compatible instruction files.
- **Side-by-side diff** — Compare two memory profile snapshots with Rich-formatted terminal output highlighting added, removed, and changed entries.
- **Extensible adapter registry** — Add support for new AI platforms by dropping in a standalone reader or writer adapter module.
- **Single-command workflow** — Go from raw export to ready-to-use Gem file in one command with no intermediate steps.

---

## Usage Examples

### Convert ChatGPT export → Gemini Gem Markdown

```bash
mem_bridge convert --from chatgpt --input chatgpt_export.zip --to gemini --output my_gem.md
```

### Convert Claude export → structured system prompt

```bash
mem_bridge convert --from claude --input claude_export.json --to markdown --output prompt.md
```

### Export a profile to YAML for archival

```bash
mem_bridge export --from chatgpt --input chatgpt_export.zip --format yaml --output profile.yaml
```

### Show a human-readable summary in the terminal

```bash
mem_bridge show --from claude --input claude_export.json
```

### Diff two memory profile snapshots

```bash
mem_bridge diff profile_old.json profile_new.json
```

Example diff output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Memory Profile Diff Report       ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

  + [ADDED]    Prefers dark mode in all editors
  - [REMOVED]  Uses Windows as primary OS
  ~ [CHANGED]  Primary language: Python 3.9 → Python 3.12
```

### List all supported platforms and formats

```bash
mem_bridge list-formats
```

### Use as a Python library

```python
from mem_bridge.adapters import get_adapter
from mem_bridge.formatters import render

adapter = get_adapter("chatgpt", mode="read")
profile = adapter.read("chatgpt_export.zip")

markdown_prompt = render(profile, fmt="markdown")
print(markdown_prompt)
```

---

## Supported Platforms

| Platform    | Read (import)     | Write (export)          |
|-------------|:-----------------:|:-----------------------:|
| **ChatGPT** | ✅ ZIP / JSON     | —                       |
| **Claude**  | ✅ JSON           | —                       |
| **Gemini**  | —                 | ✅ Gem Markdown / JSON  |

**Output formats:** `markdown`, `json`, `yaml`, `text`, `gemini`

---

## Project Structure

```
mem_bridge/
├── pyproject.toml                  # Project metadata, dependencies, and CLI entry point
├── templates/
│   └── prompt.md.j2                # Jinja2 template for system prompt rendering
├── mem_bridge/
│   ├── __init__.py                 # Package init, version, top-level symbols
│   ├── cli.py                      # Typer CLI: convert, show, diff, export, list-formats
│   ├── models.py                   # Pydantic models: MemoryProfile, ConversationEntry
│   ├── formatters.py               # Render profiles to markdown, JSON, YAML, text
│   ├── differ.py                   # Diff engine with Rich-formatted output
│   └── adapters/
│       ├── __init__.py             # Adapter registry
│       ├── chatgpt.py              # ChatGPT ZIP/JSON reader
│       ├── claude.py               # Claude JSON reader
│       └── gemini.py               # Gemini Gem Markdown/JSON writer
└── tests/
    ├── test_models.py
    ├── test_adapters.py
    ├── test_gemini_adapter.py
    ├── test_formatters.py
    ├── test_differ.py
    └── fixtures/
        ├── chatgpt_export.json
        └── claude_export.json
```

---

## Configuration

`mem_bridge` requires no configuration file for basic use. All options are passed as CLI flags.

| Flag | Description | Default |
|------|-------------|---------|
| `--from` | Source platform (`chatgpt`, `claude`) | required |
| `--input` | Path to the source export file or ZIP | required |
| `--to` | Target platform or format (`gemini`, `markdown`, `json`, `yaml`, `text`) | required |
| `--output` | Output file path. Omit to print to stdout. | stdout |
| `--format` | Explicit format override (used with `export`) | auto-detected from extension |

**Format auto-detection:** When using `--output`, the format is inferred from the file extension (`.md` → markdown/gemini, `.json` → JSON, `.yaml`/`.yml` → YAML). Override with `--format` if needed.

**Adding a custom adapter:**

Create a new file in `mem_bridge/adapters/` implementing `read()` or `write()`, then register it in `mem_bridge/adapters/__init__.py`:

```python
from mem_bridge.adapters.my_platform import MyPlatformAdapter

REGISTRY["my_platform"] = {"read": MyPlatformAdapter, "write": MyPlatformAdapter}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
