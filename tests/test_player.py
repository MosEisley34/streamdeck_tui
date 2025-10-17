import shutil

import pytest

from streamdeck_tui import player as player_module
from streamdeck_tui.player import (
    PlayerCommand,
    build_player_command,
    detect_player,
    probe_player,
)
from streamdeck_tui.playlist import Channel


def test_detect_player_prefers_preferred(monkeypatch):
    calls = []

    def fake_which(cmd: str):
        calls.append(cmd)
        return "/usr/bin/mpv" if cmd == "mpv" else None

    monkeypatch.setattr(shutil, "which", fake_which)
    assert detect_player("mpv", candidates=["vlc", "mpv"]) == "/usr/bin/mpv"
    assert calls[0] == "mpv"


def test_build_player_command_raises_when_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    channel = Channel(name="Test", url="http://example")
    with pytest.raises(RuntimeError):
        build_player_command(channel)


def test_build_player_command_success(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    channel = Channel(name="Test", url="http://example")
    command = build_player_command(channel, preferred="vlc")
    assert isinstance(command, PlayerCommand)
    assert command.executable == "/usr/bin/vlc"
    assert command.args == ["http://example"]


def test_build_player_command_adds_mpv_flags(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    channel = Channel(name="Test", url="http://example/stream")
    command = build_player_command(channel)
    assert command.executable == "/usr/bin/mpv"
    assert command.args[:3] == [
        "--force-window=immediate",
        "--player-operation-mode=pseudo-gui",
        "--no-terminal",
    ]
    assert command.args[-1] == channel.url


def test_probe_player_success(monkeypatch):
    monkeypatch.setattr(player_module, "detect_player", lambda preferred=None, **kwargs: "/usr/bin/mpv")

    class Result:
        returncode = 0
        stdout = "mpv 0.37.0"
        stderr = ""

    monkeypatch.setattr(player_module.subprocess, "run", lambda *args, **kwargs: Result())
    assert probe_player().startswith("mpv")


def test_probe_player_failure(monkeypatch):
    monkeypatch.setattr(player_module, "detect_player", lambda preferred=None, **kwargs: "/usr/bin/mpv")

    class Result:
        returncode = 1
        stdout = ""
        stderr = "fatal error"

    monkeypatch.setattr(player_module.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError):
        probe_player()
