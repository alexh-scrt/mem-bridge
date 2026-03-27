"""mem_bridge: Export, convert, and manage AI memory profiles across platforms.

This package provides a CLI tool and Python library for ingesting conversation
history and memory profiles from AI assistants (ChatGPT, Claude) and converting
them into structured prompts or import files compatible with other AI platforms
(Gemini, etc.).

Typical usage via CLI::

    mem_bridge convert --from chatgpt export.zip --to gemini --output gem.md
    mem_bridge diff profile_a.json profile_b.json
    mem_bridge list-formats

Typical usage as a library::

    from mem_bridge.adapters import get_adapter
    from mem_bridge.formatters import render

    adapter = get_adapter("chatgpt")
    profile = adapter.read("export.zip")
    output = render(profile, fmt="markdown")
"""

__version__ = "0.1.0"
__author__ = "mem_bridge contributors"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
