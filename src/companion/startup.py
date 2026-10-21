"""Start the companion at Windows sign-in, using the per-user Run key (no admin rights needed).

Usage:
    python -m companion.startup --enable
    python -m companion.startup --disable
    python -m companion.startup --status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from companion.config import PROJECT_ROOT

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "PCCompanion"
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_companion.pyw"


def launch_command(python: str | None = None, launcher: Path = LAUNCHER) -> str:
    """Command line for the Run key. Uses pythonw.exe next to the interpreter so no console appears."""
    exe = Path(python or sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = exe
    return f'"{pythonw}" "{launcher}"'


def is_enabled() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, VALUE_NAME)
    except FileNotFoundError:
        return False
    return True


def enable(command: str | None = None) -> str:
    """Register the companion to start at sign-in. Returns the command that was stored."""
    import winreg

    value = command or launch_command()
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, value)
    return value


def disable() -> bool:
    """Remove the sign-in entry. Returns False if it was not set."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start PC Companion at Windows sign-in.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enable", action="store_true", help="start at sign-in")
    group.add_argument("--disable", action="store_true", help="do not start at sign-in")
    group.add_argument("--status", action="store_true", help="show whether it starts at sign-in")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("Start at sign-in is only supported on Windows.", file=sys.stderr)
        return 1

    if args.status:
        print("Start at sign-in: " + ("on" if is_enabled() else "off"))
        return 0
    if args.disable:
        removed = disable()
        print("Start at sign-in: off" if removed else "Start at sign-in was already off.")
        return 0

    if not LAUNCHER.is_file():
        print(f"Launcher not found: {LAUNCHER}", file=sys.stderr)
        return 1
    command = enable()
    print(f"Start at sign-in: on\nCommand: {command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
