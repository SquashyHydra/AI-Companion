"""Auto-discovery of user tool files in the plugins folder."""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from companion.config import PROJECT_ROOT
from companion.tools.registry import default_registry

logger = logging.getLogger(__name__)

PLUGINS_DIR = PROJECT_ROOT / "plugins"


@dataclass
class PluginReport:
    loaded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def load_plugins(directory: Path = PLUGINS_DIR) -> PluginReport:
    """Import every *.py file in directory so its @tool decorators register tools.

    Files starting with '_' are skipped. A file that fails to load is reported and its
    partly registered tools are removed. Startup continues either way.
    """
    report = PluginReport()
    if not directory.is_dir():
        return report

    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        before = {t.name for t in default_registry.tools}
        module_name = f"companion_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create an import spec for this file")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            for name in {t.name for t in default_registry.tools} - before:
                default_registry.unregister(name)
            logger.warning("Plugin '%s' failed to load", path.name, exc_info=True)
            report.failed[path.name] = f"{type(exc).__name__}: {exc}"
            continue
        report.loaded.append(path.name)
    return report
