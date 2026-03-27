"""Adapter registry for mem_bridge platform readers and writers.

This module maintains a central registry mapping platform names to their
corresponding adapter classes. Each adapter implements a standard interface:

- Reader adapters expose a ``read(source: str | Path) -> MemoryProfile`` method.
- Writer adapters expose a ``write(profile: MemoryProfile, dest: str | Path) -> None`` method.

Registered platforms
--------------------
- ``chatgpt``  : Reads ChatGPT data-export ZIP/JSON files.
- ``claude``   : Reads Claude conversation export JSON files.
- ``gemini``   : Writes Gemini-compatible Gem instruction files.

Example usage::

    from mem_bridge.adapters import get_adapter, list_adapters

    reader = get_adapter("chatgpt", mode="read")
    profile = reader.read("chatgpt_export.zip")

    writer = get_adapter("gemini", mode="write")
    writer.write(profile, "my_gem.md")
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Lazy import helpers – the individual adapter modules are only imported when
# actually requested so that missing optional dependencies don't break the
# entire package at import time.
# ---------------------------------------------------------------------------

_READER_REGISTRY: dict[str, str] = {
    "chatgpt": "mem_bridge.adapters.chatgpt",
    "claude": "mem_bridge.adapters.claude",
}

_WRITER_REGISTRY: dict[str, str] = {
    "gemini": "mem_bridge.adapters.gemini",
}

# Canonical adapter class names per platform
_ADAPTER_CLASS_NAMES: dict[str, str] = {
    "chatgpt": "ChatGPTAdapter",
    "claude": "ClaudeAdapter",
    "gemini": "GeminiAdapter",
}

# Human-readable descriptions for each platform and mode
_ADAPTER_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "chatgpt": {
        "mode": "read",
        "description": "Reads ChatGPT data-export ZIP archives and JSON files.",
        "formats": ".zip, conversations.json, memory.json",
    },
    "claude": {
        "mode": "read",
        "description": "Reads Claude conversation export JSON files.",
        "formats": ".json",
    },
    "gemini": {
        "mode": "write",
        "description": "Writes Gemini-compatible Gem Markdown and JSON files.",
        "formats": ".md, .json",
    },
}


class AdapterNotFoundError(KeyError):
    """Raised when an adapter for a given platform or mode is not registered."""

    def __init__(self, platform: str, mode: str) -> None:
        self.platform = platform
        self.mode = mode
        super().__init__(
            f"No {mode!r} adapter registered for platform {platform!r}. "
            f"Available readers: {sorted(_READER_REGISTRY)}. "
            f"Available writers: {sorted(_WRITER_REGISTRY)}."
        )


def _import_module(module_path: str) -> Any:
    """Dynamically import a module by its dotted path string.

    Parameters
    ----------
    module_path:
        Fully-qualified module path, e.g. ``'mem_bridge.adapters.chatgpt'``.

    Returns
    -------
    Any
        The imported module object.

    Raises
    ------
    ImportError
        If the module cannot be found or imported.
    """
    import importlib

    return importlib.import_module(module_path)


def get_adapter(platform: str, mode: str = "read") -> Any:
    """Retrieve an instantiated adapter for the given platform and mode.

    Parameters
    ----------
    platform:
        Case-insensitive platform identifier, e.g. ``'chatgpt'``, ``'claude'``,
        or ``'gemini'``.
    mode:
        Either ``'read'`` (to obtain a reader adapter) or ``'write'`` (to
        obtain a writer adapter). Defaults to ``'read'``.

    Returns
    -------
    Any
        An instantiated adapter object with either a ``read`` or ``write``
        method, depending on the requested *mode*.

    Raises
    ------
    AdapterNotFoundError
        If no adapter is registered for the requested *platform* / *mode*
        combination.
    ValueError
        If *mode* is not one of ``'read'`` or ``'write'``.
    """
    platform = platform.lower().strip()

    if mode not in ("read", "write"):
        raise ValueError(f"mode must be 'read' or 'write', got {mode!r}")

    registry = _READER_REGISTRY if mode == "read" else _WRITER_REGISTRY

    if platform not in registry:
        raise AdapterNotFoundError(platform, mode)

    module = _import_module(registry[platform])

    # Look up the canonical class name first, then fall back to convention
    class_name = _ADAPTER_CLASS_NAMES.get(platform)
    if class_name is None:
        # Convention: PlatformAdapter (capitalise first letter only)
        class_name = platform.capitalize() + "Adapter"

    cls = getattr(module, class_name, None)

    if cls is None:
        raise AdapterNotFoundError(platform, mode)

    return cls()


def list_adapters() -> dict[str, list[str]]:
    """Return a summary of all registered adapters grouped by mode.

    Returns
    -------
    dict[str, list[str]]
        A dictionary with keys ``'readers'`` and ``'writers'``, each mapping
        to a sorted list of registered platform names.

    Example
    -------
    >>> from mem_bridge.adapters import list_adapters
    >>> list_adapters()
    {'readers': ['chatgpt', 'claude'], 'writers': ['gemini']}
    """
    return {
        "readers": sorted(_READER_REGISTRY.keys()),
        "writers": sorted(_WRITER_REGISTRY.keys()),
    }


def list_platforms() -> list[str]:
    """Return a deduplicated sorted list of all known platform names.

    Returns
    -------
    list[str]
        Sorted list of platform identifiers that have at least one adapter
        (reader or writer) registered.
    """
    all_platforms: set[str] = set(_READER_REGISTRY.keys()) | set(_WRITER_REGISTRY.keys())
    return sorted(all_platforms)


def get_adapter_info(platform: str) -> dict[str, str]:
    """Return human-readable metadata about a registered adapter.

    Parameters
    ----------
    platform:
        Platform identifier, e.g. ``'chatgpt'``, ``'claude'``, ``'gemini'``.

    Returns
    -------
    dict[str, str]
        Dictionary containing ``'mode'``, ``'description'``, and
        ``'formats'`` keys.

    Raises
    ------
    AdapterNotFoundError
        If *platform* is not registered in either the reader or writer registry.
    """
    platform = platform.lower().strip()
    if platform not in _ADAPTER_DESCRIPTIONS:
        # Determine which registries were searched for a useful error message
        if platform not in _READER_REGISTRY and platform not in _WRITER_REGISTRY:
            raise AdapterNotFoundError(platform, "read/write")
    return _ADAPTER_DESCRIPTIONS.get(
        platform,
        {"mode": "unknown", "description": "No description available.", "formats": ""},
    )


def register_reader(platform: str, module_path: str) -> None:
    """Register a custom reader adapter for a platform at runtime.

    Parameters
    ----------
    platform:
        Platform identifier string, e.g. ``'myprovider'``.
    module_path:
        Fully-qualified module path that contains a class named
        ``'<Platform>Adapter'`` with a ``read`` method.

    Raises
    ------
    ValueError
        If *platform* is an empty string.
    """
    if not platform:
        raise ValueError("platform must be a non-empty string")
    _READER_REGISTRY[platform.lower().strip()] = module_path


def register_writer(platform: str, module_path: str) -> None:
    """Register a custom writer adapter for a platform at runtime.

    Parameters
    ----------
    platform:
        Platform identifier string, e.g. ``'myprovider'``.
    module_path:
        Fully-qualified module path that contains a class named
        ``'<Platform>Adapter'`` with a ``write`` method.

    Raises
    ------
    ValueError
        If *platform* is an empty string.
    """
    if not platform:
        raise ValueError("platform must be a non-empty string")
    _WRITER_REGISTRY[platform.lower().strip()] = module_path


__all__ = [
    "AdapterNotFoundError",
    "get_adapter",
    "get_adapter_info",
    "list_adapters",
    "list_platforms",
    "register_reader",
    "register_writer",
]
