"""System tray icon with Open logs and Quit. Runs on its own thread next to the voice loop."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("companion.tray")


def _icon_image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((6, 6, 58, 58), fill=(46, 134, 222, 255))
    return image


def status_text(listening: bool) -> str:
    return "Status: listening" if listening else "Status: paused"


def toggle_text(listening: bool) -> str:
    return "Pause listening" if listening else "Resume listening"


class TrayIcon:
    """Shows the icon in the notification area. Quit calls on_quit; it does not stop the app itself.

    on_toggle (optional) pauses or resumes listening. set_listening shows the current state in
    the menu and the tooltip.
    """

    def __init__(
        self,
        on_quit: Callable[[], None],
        logs_dir: Path,
        on_toggle: Callable[[], None] | None = None,
    ) -> None:
        self._on_quit = on_quit
        self._on_toggle = on_toggle
        self._logs_dir = logs_dir
        self._listening = True
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start the icon. Returns False if pystray or Pillow is not installed."""
        try:
            import pystray
        except ImportError:
            logger.warning("pystray is not installed; the tray icon is off")
            return False
        items: list[Any] = [
            pystray.MenuItem("PC Companion", None, enabled=False),
            pystray.MenuItem(lambda item: status_text(self._listening), None, enabled=False),
        ]
        if self._on_toggle is not None:
            items.append(pystray.MenuItem(lambda item: toggle_text(self._listening), self._toggle))
        items += [
            pystray.MenuItem("Open logs folder", self._open_logs),
            pystray.MenuItem("Quit", self._quit),
        ]
        menu = pystray.Menu(*items)
        self._icon = pystray.Icon("pc-companion", _icon_image(), "PC Companion", menu)
        self._thread = threading.Thread(target=self._icon.run, name="tray", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()

    def set_listening(self, listening: bool) -> None:
        """Show whether listening is on. Called from the voice loop; the menu refreshes on the next open."""
        self._listening = listening
        if self._icon is not None:
            self._icon.title = f"PC Companion: {'listening' if listening else 'paused'}"
            self._icon.update_menu()

    def _toggle(self, icon: Any, item: Any) -> None:
        if self._on_toggle is not None:
            self._on_toggle()

    def _quit(self, icon: Any, item: Any) -> None:
        self._on_quit()
        icon.stop()

    def _open_logs(self, icon: Any, item: Any) -> None:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(self._logs_dir)  # type: ignore[attr-defined]
