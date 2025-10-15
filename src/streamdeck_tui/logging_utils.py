"""Logging helpers for :mod:`streamdeck_tui`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

__all__ = ["configure_logging", "get_logger"]

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
