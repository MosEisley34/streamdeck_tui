"""Command line entry point for Streamdeck TUI."""
from __future__ import annotations

import argparse
import importlib.resources as resources
from pathlib import Path
from typing import Iterable

from .app import StreamdeckApp
from .config import CONFIG_PATH, load_config
from .logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamdeck IPTV management TUI")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override STREAMDECK_TUI_LOG_LEVEL for this invocation",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Write logs to this file instead of the default or"
            " STREAMDECK_TUI_LOG_FILE"
        ),
    )
    parser.add_argument(
        "--player",
        dest="preferred_player",
        default=None,
        help="Preferred media player executable to launch (default: auto-detect)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(
        level=args.log_level,
        log_file=str(args.log_file) if args.log_file is not None else None,
    )
    warn_if_legacy_stylesheet()
    log.info("CLI invoked with config=%s", args.config)
    config = load_config(args.config)
    app = StreamdeckApp(
        config,
        config_path=args.config,
        preferred_player=args.preferred_player,
    )
    log.info("Launching Textual application")
    try:
        app.run()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received; exiting application")
        if app.is_running:
            app.exit()
        raise SystemExit(130) from None


def warn_if_legacy_stylesheet() -> None:
    """Log guidance if an old packaged stylesheet is still present."""

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    try:
        css_text = css_resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    legacy_tokens = ("gap:", "$surface", "$accent")
    if any(token in css_text for token in legacy_tokens):
        log.warning(
            "Detected legacy stylesheet at %s. This file belongs to an older"
            " installation. Reinstall the project with 'pip install -e .'"
            " (or run 'make install') to refresh the console script, or"
            " uninstall the previous package with 'pip uninstall"
            " streamdeck-tui'.",
            css_resource,
        )

if __name__ == "__main__":  # pragma: no cover
    main()
