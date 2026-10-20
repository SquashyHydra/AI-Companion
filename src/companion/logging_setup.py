"""File logging: the app log and the tool-call audit log under logs/."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from companion.config import PROJECT_ROOT
from companion.tools.registry import audit_logger

LOG_DIR = PROJECT_ROOT / "logs"
_MAX_BYTES = 5_000_000
_BACKUPS = 5


def configure_logging(log_dir: Path = LOG_DIR) -> None:
    """Write companion.log and tool_calls.jsonl to log_dir. Call once at startup.

    Warnings also go to the console. Tool calls are kept out of the console and the app log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    text_format = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(text_format)

    app_file = RotatingFileHandler(
        log_dir / "companion.log", maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
    )
    app_file.setFormatter(text_format)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(app_file)

    audit_file = RotatingFileHandler(
        log_dir / "tool_calls.jsonl", maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
    )
    audit_file.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_file)
    audit_logger.propagate = False
