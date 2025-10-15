"""Player detection and spawning helpers."""
from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .playlist import Channel
from .logging_utils import get_logger


DEFAULT_PLAYER_CANDIDATES: Sequence[str] = ("mpv", "vlc", "ffplay")


log = get_logger(__name__)

@dataclass(slots=True)
class PlayerCommand:
    """Describe a player invocation."""

    executable: str
    args: list[str]

    def as_sequence(self) -> list[str]:
        return [self.executable, *self.args]


def detect_player(preferred: Optional[str] = None, *, candidates: Iterable[str] = DEFAULT_PLAYER_CANDIDATES) -> Optional[str]:
    """Return the path to the first available player executable."""

    search_order: list[str] = []
    if preferred:
        log.debug("Preferred player requested: %s", preferred)
        search_order.append(preferred)
    for candidate in candidates:
        if candidate not in search_order:
            search_order.append(candidate)
    for executable in search_order:
        path = shutil.which(executable)
        if path:
            log.info("Selected player executable: %s (from candidate %s)", path, executable)
            return path
        log.debug("Player candidate %s not found on PATH", executable)
    return None


def build_player_command(channel: Channel, *, preferred: Optional[str] = None) -> PlayerCommand:
    """Construct a player command for the given channel."""

    executable = detect_player(preferred)
    if executable is None:
        log.error("Unable to locate supported media player")
        raise RuntimeError("No supported media player found (mpv, vlc, ffplay)")
    command = PlayerCommand(executable=executable, args=[channel.url])
    log.info("Built player command for channel %s: %s", channel.name, command.as_sequence())
    return command


async def launch_player(channel: Channel, *, preferred: Optional[str] = None) -> asyncio.subprocess.Process:
    """Launch a media player for the channel."""

    command = build_player_command(channel, preferred=preferred)
    env = os.environ.copy()
    log.info("Launching player process for %s", channel.name)
    process = await asyncio.create_subprocess_exec(*command.as_sequence(), env=env)
    log.debug("Spawned process PID %s", getattr(process, "pid", "unknown"))
    return process


__all__ = [
    "PlayerCommand",
    "detect_player",
    "build_player_command",
    "launch_player",
]
