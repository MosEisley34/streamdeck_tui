"""Logging helpers for :mod:`streamdeck_tui`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import weakref
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .app import StreamdeckApp
    from .log_viewer import LogViewer

__all__ = ["configure_logging", "get_logger"]

_ENV_LEVEL = "STREAMDECK_TUI_LOG_LEVEL"
_ENV_FILE = "STREAMDECK_TUI_LOG_FILE"
_DEFAULT_LOG_PATH = Path.home() / ".cache" / "streamdeck_tui.log"
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _TextualLogHandler(logging.Handler):
    """Logging handler that forwards records to the Textual UI."""

    def __init__(self, app: "StreamdeckApp", viewer: "LogViewer") -> None:
        super().__init__()
        self._app_ref: "weakref.ReferenceType[StreamdeckApp]" = weakref.ref(app)
        self._viewer_ref: "weakref.ReferenceType[LogViewer]" = weakref.ref(viewer)

    @property
    def viewer(self) -> Optional["LogViewer"]:
        return self._viewer_ref()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        app = self._app_ref()
        viewer = self._viewer_ref()
        if app is None or viewer is None:
            return
        try:
            formatted = self.format(record)
            app_thread_id = getattr(app, "_app_thread_id", getattr(app, "_thread_id", None))
            if app_thread_id is not None and threading.get_ident() == app_thread_id:
                viewer.post_message(record, formatted)
            else:
                app.call_from_thread(viewer.post_message, record, formatted)
        except Exception:  # pragma: no cover - defensive against UI failures
            self.handleError(record)


def _create_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)


def _attach_textual_handler(
    logger: logging.Logger,
    *,
    app: Optional["StreamdeckApp"],
    log_viewer: Optional["LogViewer"],
) -> None:
    if app is None or log_viewer is None:
        return
    existing = getattr(configure_logging, "_textual_handler", None)
    if existing is not None:
        logger.removeHandler(existing)
    handler = _TextualLogHandler(app, log_viewer)
    handler.setLevel(logger.getEffectiveLevel())
    handler.setFormatter(_create_formatter())
    logger.addHandler(handler)
    configure_logging._textual_handler = handler  # type: ignore[attr-defined]


def _coerce_level(value: str) -> int:
    """Return a logging level derived from *value*."""

    normalized = value.strip().upper()
    if normalized.isdigit():
        level = int(normalized)
        if 0 <= level <= logging.CRITICAL:
            return level
    return getattr(logging, normalized, logging.INFO)


def configure_logging(
    *,
    level: Optional[str] = None,
    app: Optional["StreamdeckApp"] = None,
    log_viewer: Optional["LogViewer"] = None,
) -> logging.Logger:
    """Configure the package logger if it hasn't been set up yet."""

    logger = logging.getLogger("streamdeck_tui")
    if getattr(configure_logging, "_configured", False):
        _attach_textual_handler(logger, app=app, log_viewer=log_viewer)
        return logger

    env_level = os.getenv(_ENV_LEVEL)
    log_level = _coerce_level(level or env_level or "INFO")

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = _create_formatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    destination = os.getenv(_ENV_FILE)
    if destination is None:
        destination = str(_DEFAULT_LOG_PATH)
    if destination:
        log_path = Path(destination).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf8")
        except OSError:
            logger.warning("Failed to set up file logging at %s", log_path)
        else:
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            logger.debug("File logging enabled at %s", log_path)

    configure_logging._configured = True  # type: ignore[attr-defined]
    _attach_textual_handler(logger, app=app, log_viewer=log_viewer)
    logger.debug("Logging configured with level %s", logging.getLevelName(log_level))
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-specific logger."""

    base = configure_logging()
    if not name or name == base.name:
        return base
    if name.startswith(base.name + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{base.name}.{name}")
