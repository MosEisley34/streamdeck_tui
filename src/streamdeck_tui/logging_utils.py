"""Logging helpers for :mod:`streamdeck_tui`."""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import threading
import weakref

if TYPE_CHECKING:  # pragma: no cover - only for typing
    from .app import LogViewer

__all__ = ["configure_logging", "get_logger", "register_log_viewer"]

_ENV_LEVEL = "STREAMDECK_TUI_LOG_LEVEL"
_ENV_FILE = "STREAMDECK_TUI_LOG_FILE"
_DEFAULT_LOG_PATH = Path.home() / ".cache" / "streamdeck_tui.log"


def _coerce_level(value: str) -> int:
    """Return a logging level derived from *value*."""

    normalized = value.strip().upper()
    if normalized.isdigit():
        level = int(normalized)
        if 0 <= level <= logging.CRITICAL:
            return level
    return getattr(logging, normalized, logging.INFO)


class _UILogHandler(logging.Handler):
    """Handler that relays log records to the TUI log viewer."""

    def __init__(self, *, capacity: int = 200) -> None:
        super().__init__()
        self._buffer: deque[str] = deque(maxlen=capacity)
        self._viewer: Optional[weakref.ReferenceType["LogViewer"]] = None
        self._lock = threading.RLock()

    def set_viewer(self, viewer: Optional["LogViewer"]) -> None:
        with self._lock:
            self._viewer = weakref.ref(viewer) if viewer else None
            if viewer is not None:
                messages = list(self._buffer)
                app = viewer.app
                if app is None:  # pragma: no cover - defensive
                    return
                try:
                    app.call_from_thread(viewer.replace_messages, messages)
                except RuntimeError:
                    viewer.replace_messages(messages)
                except Exception:  # pragma: no cover - defensive
                    pass

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        with self._lock:
            self._buffer.append(message)
            viewer_ref = self._viewer
        if viewer_ref is None:
            return
        viewer = viewer_ref()
        if viewer is None:
            return
        app = viewer.app
        if app is None:  # pragma: no cover - defensive
            return
        try:
            app.call_from_thread(viewer.append_message, message)
        except RuntimeError:
            viewer.append_message(message)
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


def configure_logging(*, level: Optional[str] = None) -> logging.Logger:
    """Configure the package logger if it hasn't been set up yet."""

    logger = logging.getLogger("streamdeck_tui")
    if getattr(configure_logging, "_configured", False):
        return logger

    env_level = os.getenv(_ENV_LEVEL)
    log_level = _coerce_level(level or env_level or "INFO")

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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

    ui_handler = _UILogHandler()
    ui_handler.setFormatter(formatter)
    ui_handler.setLevel(log_level)
    logger.addHandler(ui_handler)

    configure_logging._ui_handler = ui_handler  # type: ignore[attr-defined]
    configure_logging._configured = True  # type: ignore[attr-defined]
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


def register_log_viewer(viewer: Optional["LogViewer"]) -> None:
    """Attach *viewer* to the in-app log handler."""

    logger = configure_logging()
    handler: Optional[_UILogHandler] = getattr(configure_logging, "_ui_handler", None)
    if handler is None:
        logger.warning("UI log handler is not available")
        return
    handler.set_viewer(viewer)
