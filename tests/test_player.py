import shutil
from pathlib import Path

import pytest

from streamdeck_tui.player import build_player_command, detect_player, PlayerCommand
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
