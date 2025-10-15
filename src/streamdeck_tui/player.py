"""Player detection and spawning helpers."""
from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .playlist import Channel


DEFAULT_PLAYER_CANDIDATES: Sequence[str] = ("mpv", "vlc", "ffplay")


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
        search_order.append(preferred)
    for candidate in candidates:
        if candidate not in search_order:
            search_order.append(candidate)
    for executable in search_order:
        path = shutil.which(executable)
        if path:
            return path
    return None


def build_player_command(channel: Channel, *, preferred: Optional[str] = None) -> PlayerCommand:
    """Construct a player command for the given channel."""

    executable = detect_player(preferred)
    if executable is None:
        raise RuntimeError("No supported media player found (mpv, vlc, ffplay)")
    return PlayerCommand(executable=executable, args=[channel.url])


async def launch_player(channel: Channel, *, preferred: Optional[str] = None) -> asyncio.subprocess.Process:
    """Launch a media player for the channel."""

    command = build_player_command(channel, preferred=preferred)
    env = os.environ.copy()
    return await asyncio.create_subprocess_exec(*command.as_sequence(), env=env)


__all__ = [
    "PlayerCommand",
    "detect_player",
    "build_player_command",
    "launch_player",
]
