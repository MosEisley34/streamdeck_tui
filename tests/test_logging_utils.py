"""Tests for :mod:`streamdeck_tui.logging_utils`."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
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
    monkeypatch.delattr(logging_utils.configure_logging, "_file_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_path", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_level", raising=False)

    logger = logging.getLogger("streamdeck_tui")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    loaded_configs: list[Any] = []
    created_apps: list[Any] = []
    
    def fake_load_config(path: Any) -> dict[str, Any]:
        loaded_configs.append(path)
        return {"path": path}

    class DummyApp:
        def __init__(
            self,
            config: dict[str, Any],
            *,
            config_path: Any,
            preferred_player: Any = None,
        ) -> None:
            self.config = config
            self.config_path = config_path
            self.preferred_player = preferred_player
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


def test_cli_accepts_logging_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ensure ``--log-level`` and ``--log-file`` influence logging setup."""

    monkeypatch.setenv("STREAMDECK_TUI_LOG_FILE", "")

    from streamdeck_tui import logging_utils

    importlib.reload(logging_utils)
    cli = importlib.reload(importlib.import_module("streamdeck_tui.cli"))

    monkeypatch.setattr(logging_utils.configure_logging, "_configured", False, raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_app", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_viewer", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_textual_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_ui_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_file_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_path", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_level", raising=False)

    logger = logging.getLogger("streamdeck_tui")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    captured: list[tuple[Any, Any]] = []

    def fake_configure_logging(**kwargs: Any) -> Any:
        captured.append((kwargs.get("level"), kwargs.get("log_file")))
        return logging_utils.configure_logging(**kwargs)

    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)

    def fake_load_config(path: Any) -> dict[str, Any]:
        return {"path": path}

    class DummyApp:
        def __init__(
            self,
            config: dict[str, Any],
            *,
            config_path: Any,
            preferred_player: Any = None,
        ) -> None:
            self.config = config
            self.config_path = config_path
            self.preferred_player = preferred_player

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "StreamdeckApp", DummyApp)
    monkeypatch.setattr(cli, "warn_if_legacy_stylesheet", lambda: None)

    log_file = tmp_path / "cli.log"
    cli.main(("--log-level", "DEBUG", "--log-file", str(log_file)))

    assert captured, "configure_logging should be invoked"
    assert captured[-1][0] == "DEBUG"
    assert captured[-1][1] == str(log_file)
    assert log_file.exists()


def test_cli_passes_preferred_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should pass --player through to the application."""

    monkeypatch.setenv("STREAMDECK_TUI_LOG_FILE", "")

    from streamdeck_tui import logging_utils

    importlib.reload(logging_utils)
    cli = importlib.reload(importlib.import_module("streamdeck_tui.cli"))

    monkeypatch.setattr(logging_utils.configure_logging, "_configured", False, raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_app", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_viewer", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_textual_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_ui_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_file_handler", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_log_path", raising=False)
    monkeypatch.delattr(logging_utils.configure_logging, "_level", raising=False)

    logger = logging.getLogger("streamdeck_tui")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    def fake_load_config(path: Any) -> dict[str, Any]:
        return {"path": path}

    created_apps: list[Any] = []

    class DummyApp:
        def __init__(
            self,
            config: dict[str, Any],
            *,
            config_path: Any,
            preferred_player: Any = None,
        ) -> None:
            self.config = config
            self.config_path = config_path
            self.preferred_player = preferred_player
            created_apps.append(self)

        def run(self) -> None:
            pass

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "StreamdeckApp", DummyApp)
    monkeypatch.setattr(cli, "warn_if_legacy_stylesheet", lambda: None)

    cli.main(("--player", "custom-mpv"))

    assert created_apps, "StreamdeckApp should be instantiated"
    assert created_apps[-1].preferred_player == "custom-mpv"


def test_configure_logging_updates_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runtime calls should be able to update the log level."""

    monkeypatch.setenv("STREAMDECK_TUI_LOG_FILE", "")
    from streamdeck_tui import logging_utils

    module = importlib.reload(logging_utils)
    monkeypatch.setattr(module.configure_logging, "_configured", False, raising=False)
    monkeypatch.delattr(module.configure_logging, "_app", raising=False)
    monkeypatch.delattr(module.configure_logging, "_log_viewer", raising=False)
    monkeypatch.delattr(module.configure_logging, "_textual_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_ui_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_file_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_log_path", raising=False)
    monkeypatch.delattr(module.configure_logging, "_level", raising=False)

    logger = logging.getLogger("streamdeck_tui")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger = module.configure_logging(level="INFO")
    assert logger.getEffectiveLevel() == logging.INFO

    module.configure_logging(level="DEBUG")
    assert logger.getEffectiveLevel() == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in logger.handlers)


def test_configure_logging_changes_file_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Switching log files should replace the active file handler."""

    monkeypatch.setenv("STREAMDECK_TUI_LOG_FILE", "")
    from streamdeck_tui import logging_utils

    module = importlib.reload(logging_utils)
    monkeypatch.setattr(module.configure_logging, "_configured", False, raising=False)
    monkeypatch.delattr(module.configure_logging, "_app", raising=False)
    monkeypatch.delattr(module.configure_logging, "_log_viewer", raising=False)
    monkeypatch.delattr(module.configure_logging, "_textual_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_ui_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_file_handler", raising=False)
    monkeypatch.delattr(module.configure_logging, "_log_path", raising=False)
    monkeypatch.delattr(module.configure_logging, "_level", raising=False)

    logger = logging.getLogger("streamdeck_tui")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    logger = module.configure_logging(level="INFO")
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    log_path = tmp_path / "runtime.log"
    module.configure_logging(log_file=str(log_path))

    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers and Path(file_handlers[-1].baseFilename) == log_path
    assert log_path.exists()
