"""Tests for the command line interface helpers."""
from __future__ import annotations

import pytest

from pathlib import Path

from streamdeck_tui import cli
from streamdeck_tui.themes import CUSTOM_THEMES


def test_help_lists_all_themes(capsys: pytest.CaptureFixture[str]) -> None:
    """The --theme help text should reflect the packaged theme catalog."""

    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])

    help_text = capsys.readouterr().out
    for theme_name in sorted(CUSTOM_THEMES):
        assert theme_name in help_text


def test_list_themes_short_circuits_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--list-themes should print the catalog without instantiating the app."""

    def _unexpected_app(*args, **kwargs):  # pragma: no cover - only used when failing
        raise AssertionError("StreamdeckApp should not be constructed when listing themes")

    monkeypatch.setattr(cli, "StreamdeckApp", _unexpected_app)

    cli.main(["--list-themes"])

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == sorted(CUSTOM_THEMES)


def test_provider_logs_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--provider-logs should print log lines without starting the TUI."""

    log_file = tmp_path / "streamdeck.log"
    log_file.write_text(
        "2024-01-01 10:00:00 [INFO] streamdeck_tui.app: Beginning channel load for provider Eagle 4K\n"
        "2024-01-01 10:00:01 [ERROR] streamdeck_tui.providers: Failed to fetch status for Eagle 4K\n",
        encoding="utf8",
    )

    def _unexpected_app(*args, **kwargs):  # pragma: no cover - sanity check
        raise AssertionError("StreamdeckApp should not start when dumping logs")

    monkeypatch.setattr(cli, "StreamdeckApp", _unexpected_app)

    cli.main(["--log-file", str(log_file), "--provider-logs", "Eagle 4K"])

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured[0] == f"Log file: {log_file}"
    assert "provider Eagle 4K" in captured[1]
    assert "Eagle 4K" in captured[2]
