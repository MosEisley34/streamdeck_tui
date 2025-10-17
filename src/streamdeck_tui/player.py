"""Player detection and spawning helpers."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from uuid import uuid4

from .playlist import Channel
from .logging_utils import get_logger


PREFERRED_PLAYER_DEFAULT = "/usr/bin/mpv"

DEFAULT_PLAYER_CANDIDATES: Sequence[str] = ("mpv", "vlc", "ffplay")

PLAYER_PROBE_TIMEOUT_ENV = "STREAMDECK_TUI_PLAYER_PROBE_TIMEOUT"
DEFAULT_PLAYER_PROBE_TIMEOUT = 10.0


log = get_logger(__name__)

@dataclass(slots=True)
class PlayerCommand:
    """Describe a player invocation."""

    executable: str
    args: list[str]
    ipc_path: Optional[str] = None
    cleanup_paths: tuple[Path, ...] = ()

    def as_sequence(self) -> list[str]:
        return [self.executable, *self.args]


@dataclass(slots=True)
class PlayerHandle:
    """Return value from :func:`launch_player` containing process metadata."""

    process: asyncio.subprocess.Process
    command: PlayerCommand


def detect_player(
    preferred: Optional[str] = None,
    *,
    candidates: Iterable[str] = DEFAULT_PLAYER_CANDIDATES,
) -> Optional[str]:
    """Return the path to the first available player executable."""

    search_order: list[str] = []
    if preferred:
        preferred_str = str(preferred)
        log.debug("Preferred player requested: %s", preferred_str)
        search_order.append(preferred_str)
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


def _prepare_mpv_ipc() -> tuple[Optional[str], tuple[Path, ...]]:
    """Return an IPC path suitable for mpv along with cleanup targets."""

    if os.name == "nt":
        # Use a unique named pipe for Windows. mpv will create and tear down the
        # pipe automatically, so no cleanup is required.
        return rf"\\\\.\\pipe\\streamdeck_tui_{uuid4().hex}", ()

    temp_dir = Path(tempfile.mkdtemp(prefix="streamdeck_tui_mpv_"))
    ipc_path = temp_dir / "ipc.sock"
    return str(ipc_path), (temp_dir,)


def _player_probe_timeout() -> float:
    """Return the timeout to use for player probes."""

    raw_value = os.getenv(PLAYER_PROBE_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_PLAYER_PROBE_TIMEOUT
    try:
        timeout = float(raw_value)
    except ValueError:
        log.warning(
            "Invalid %s value %r; using default %.1f seconds",
            PLAYER_PROBE_TIMEOUT_ENV,
            raw_value,
            DEFAULT_PLAYER_PROBE_TIMEOUT,
        )
        return DEFAULT_PLAYER_PROBE_TIMEOUT
    if timeout <= 0:
        log.warning(
            "Probe timeout %.1f from %s must be positive; using default",
            timeout,
            PLAYER_PROBE_TIMEOUT_ENV,
        )
        return DEFAULT_PLAYER_PROBE_TIMEOUT
    return timeout


def _detect_display_backend() -> str:
    """Return a string describing the current display backend."""

    platform = sys.platform
    if platform.startswith("win"):
        return "windows"
    if platform == "darwin":
        return "darwin"
    if os.getenv("WAYLAND_DISPLAY"):
        return "wayland"
    if os.getenv("DISPLAY"):
        return "x11"
    return "unknown"


def _mpv_hardware_flags() -> list[str]:
    """Return hardware-accelerated mpv flags appropriate for the environment."""

    backend = _detect_display_backend()
    if backend in {"windows", "darwin"}:
        return ["--hwdec=auto-safe"]
    if backend == "wayland":
        return ["--hwdec=auto-safe", "--vo=gpu", "--gpu-context=wayland"]
    if backend == "x11":
        return ["--hwdec=auto-safe", "--vo=gpu", "--gpu-context=x11"]
    return []


def build_player_command(channel: Channel, *, preferred: Optional[str] = None) -> PlayerCommand:
    """Construct a player command for the given channel."""

    executable = detect_player(preferred)
    if executable is None:
        log.error("Unable to locate supported media player")
        raise RuntimeError("No supported media player found (mpv, vlc, ffplay)")
    args: list[str] = []
    cleanup_paths: tuple[Path, ...] = ()
    ipc_path: Optional[str] = None
    executable_name = Path(executable).name.lower()
    if executable_name == "mpv":
        ipc_path, cleanup_paths = _prepare_mpv_ipc()
        args.extend(
            [
                "--force-window=immediate",
                "--player-operation-mode=pseudo-gui",
                "--no-terminal",
                "--no-interpolation",
                "--video-sync=display-resample",
                "--scale=bilinear",
                "--dscale=bilinear",
                "--cscale=bilinear",
                "--framedrop=vo",
                "--mute",
            ]
        )
        args.extend(_mpv_hardware_flags())
        if ipc_path:
            args.append(f"--input-ipc-server={ipc_path}")
    command = PlayerCommand(
        executable=executable,
        args=[*args, channel.url],
        ipc_path=ipc_path,
        cleanup_paths=cleanup_paths,
    )
    log.info("Built player command for channel %s: %s", channel.name, command.as_sequence())
    return command


async def launch_player(channel: Channel, *, preferred: Optional[str] = None) -> PlayerHandle:
    """Launch a media player for the channel."""

    command = build_player_command(channel, preferred=preferred)
    env = os.environ.copy()
    log.info("Launching player process for %s", channel.name)
    process = await asyncio.create_subprocess_exec(*command.as_sequence(), env=env)
    log.debug("Spawned process PID %s", getattr(process, "pid", "unknown"))
    return PlayerHandle(process=process, command=command)


def probe_player(preferred: Optional[str] = None) -> str:
    """Invoke the preferred player with ``--version`` to verify availability."""

    executable = detect_player(preferred)
    if executable is None:
        raise RuntimeError("No supported media player found (mpv, vlc, ffplay)")
    timeout = _player_probe_timeout()
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            (
                f"{Path(executable).name} --version timed out after {timeout:.1f} seconds. "
                f"Increase the timeout via the {PLAYER_PROBE_TIMEOUT_ENV} environment variable "
                "or run the command manually."
            )
        ) from exc
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to execute {executable}: {exc}") from exc
    except subprocess.SubprocessError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Player probe failed: {exc}") from exc
    if result.returncode != 0:
        output = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"{Path(executable).name} --version exited with {result.returncode}: {output}"
        )
    output = result.stdout.strip() or result.stderr.strip()
    summary = output.splitlines()[0] if output else Path(executable).name
    log.info("Player probe succeeded using %s: %s", executable, summary)
    return summary


__all__ = [
    "PREFERRED_PLAYER_DEFAULT",
    "PlayerCommand",
    "PlayerHandle",
    "detect_player",
    "build_player_command",
    "launch_player",
    "probe_player",
]
