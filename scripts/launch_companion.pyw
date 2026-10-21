"""Windows sign-in launcher. Starts the companion in tray mode with no console window.

Registered by `python -m companion.startup --enable`. Run through pythonw.exe.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
LOG_DIR = ROOT / "logs"


def _redirect_missing_stdio() -> None:
    """pythonw.exe has no console, so sys.stdout and sys.stderr are None. Send them to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / "startup-console.log", "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    if sys.stdin is None:
        sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115


if __name__ == "__main__":
    sys.path.insert(0, str(SRC))
    os.chdir(ROOT)
    _redirect_missing_stdio()
    from companion.main import main

    sys.exit(main(["--tray", "--mode", "continuous", "--background"]))
