"""Tests for :mod:`streamdeck_tui.logging_utils`."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.mark.parametrize("argv", [()])
def test_cli_main_configures_logging_without_ui(monkeypatch: pytest.MonkeyPatch, argv: Any) -> None:
    """CLI should configure logging safely before the Textual app exists."""

    monkeypatch.setenv("STREAMDECK_TUI_LOG_FILE", "")

    # Ensure we start from a clean logging state before invoking the CLI entry point.
    from streamdeck_tui import logging_utils

    importlib.reload(logging_utils)

    # Reload the CLI module afterwards so it picks up the reset logging helpers.
    cli = importlib.reload(importlib.import_module("streamdeck_tui.cli"))

    # Remove any handler state that may linger from other tests.
    monkeypatch.setattr(logging_utils.configure_logging, "_configured", False, raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_app", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_viewer", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_textual_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_ui_handler", raising=False)

    loaded_configs: list[Any] = []
    created_apps: list[Any] = []
    
    def fake_load_config(path: Any) -> dict[str, Any]:
        loaded_configs.append(path)
        return {"path": path}

    class DummyApp:
        def __init__(self, config: dict[str, Any], *, config_path: Any) -> None:
            self.config = config
            self.config_path = config_path
            self.ran = False
            created_apps.append(self)

        def run(self) -> None:
            self.ran = True

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "StreamdeckApp", DummyApp)
    monkeypatch.setattr(cli, "warn_if_legacy_stylesheet", lambda: None)

    cli.main(argv)

    assert loaded_configs, "load_config should be invoked"
    assert created_apps, "StreamdeckApp should be instantiated"
    # The dummy application should have run without configure_logging raising.
    assert created_apps[-1].ran is True
