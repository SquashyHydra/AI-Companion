"""Volume and media keys. They send the same key presses as a keyboard, so they work in any app."""

from __future__ import annotations

import ctypes
import sys
from typing import Literal

from companion.tools.registry import tool

KEYEVENTF_KEYUP = 0x0002
MAX_VOLUME_STEPS = 20
VOLUME_KEYS = {"up": 0xAF, "down": 0xAE, "mute": 0xAD}
MEDIA_KEYS = {"play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2}


def _press(virtual_key: int) -> None:
    """Press and release one virtual key. Replaced in tests."""
    if sys.platform != "win32":
        raise RuntimeError("media keys only work on Windows")
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.keybd_event(virtual_key, 0, 0, 0)
    user32.keybd_event(virtual_key, 0, KEYEVENTF_KEYUP, 0)


@tool
def volume(action: Literal["up", "down", "mute"], steps: int = 2) -> str:
    """Change the computer volume. Each step is about 2 percent. Mute toggles mute on and off.

    Args:
        action: up, down, or mute.
        steps: How many steps to change by, from 1 to 20. Ignored for mute.
    """
    if action == "mute":
        return _send(VOLUME_KEYS["mute"], 1, "Toggled mute.")
    if not 1 <= steps <= MAX_VOLUME_STEPS:
        return "Error: steps must be from 1 to 20."
    return _send(VOLUME_KEYS[action], steps, f"Volume {action} by {steps} step(s).")


@tool
def media(action: Literal["play_pause", "next", "previous", "stop"]) -> str:
    """Control the music or video that is playing: play or pause, next or previous track, or stop.

    Args:
        action: play_pause, next, previous, or stop.
    """
    if action not in MEDIA_KEYS:
        return "Error: action must be play_pause, next, previous, or stop."
    return _send(MEDIA_KEYS[action], 1, f"Sent media key: {action}.")


def _send(virtual_key: int, times: int, done: str) -> str:
    try:
        for _ in range(times):
            _press(virtual_key)
    except Exception as exc:
        return f"Error: could not send the key press: {exc}"
    return done
