"""Folder roots for file search: Windows known-folder shortcuts and hints for missing folders."""

from __future__ import annotations

import difflib
import os
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

KNOWN_FOLDERS: dict[str, str] = {
    "desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}
"""Shortcut names that can be used in tools.file_roots. Windows asks the shell where each lives,
so folders moved by OneDrive or a drive change are still found."""


def is_known_folder_name(text: str) -> bool:
    return text.strip().lower() in KNOWN_FOLDERS


def known_folder(name: str) -> Path:
    """The path of a Windows known folder. Falls back to the usual name under the home folder."""
    key = name.strip().lower()
    if key not in KNOWN_FOLDERS:
        raise ValueError(f"unknown folder shortcut '{name}'")
    shell_path = _shell_known_folder(KNOWN_FOLDERS[key]) if sys.platform == "win32" else None
    return Path(shell_path) if shell_path else Path.home() / key.capitalize()


def _shell_known_folder(guid: str) -> str | None:
    """Ask the Windows shell for a known folder path. Returns None if the shell cannot say."""
    import ctypes

    try:
        identifier = (ctypes.c_char * 16).from_buffer_copy(uuid.UUID(guid).bytes_le)
        out = ctypes.c_wchar_p()
        status = ctypes.windll.shell32.SHGetKnownFolderPath(  # type: ignore[attr-defined]
            ctypes.byref(identifier), 0, None, ctypes.byref(out)
        )
        if status != 0 or not out.value:
            return None
        value = out.value
        ctypes.windll.ole32.CoTaskMemFree(out)  # type: ignore[attr-defined]
        return value
    except (AttributeError, OSError, ValueError):
        return None


def resolve_root(text: str, project_root: Path) -> Path:
    """Turn one tools.file_roots entry into a path: a shortcut name, or a path with ~ or %VARS%."""
    cleaned = text.strip()
    if is_known_folder_name(cleaned):
        return known_folder(cleaned)
    path = Path(os.path.expandvars(cleaned)).expanduser()
    return path if path.is_absolute() else project_root / path


def root_problems(roots: Iterable[Path]) -> list[str]:
    """One message per configured folder that does not exist, with a suggestion when one is close."""
    problems = []
    for root in roots:
        if root.is_dir():
            continue
        suggestion = _suggest(root)
        hint = f" Did you mean {suggestion}?" if suggestion else " It is skipped."
        problems.append(f"File folder not found: {root}.{hint}")
    return problems


def _suggest(missing: Path) -> Path | None:
    parent = missing.parent
    if not parent.is_dir():
        return None
    try:
        names = {child.name.lower(): child.name for child in parent.iterdir() if child.is_dir()}
    except OSError:
        return None
    close = difflib.get_close_matches(missing.name.lower(), list(names), n=1, cutoff=0.8)
    return parent / names[close[0]] if close else None
