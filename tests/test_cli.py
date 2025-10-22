"""Tests for the command line interface helpers."""
from __future__ import annotations

import pytest

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
