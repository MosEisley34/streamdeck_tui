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
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    configure_logging()
    warn_if_legacy_stylesheet()
    args = parse_args(argv)
    log.info("CLI invoked with config=%s", args.config)
    config = load_config(args.config)
    app = StreamdeckApp(config, config_path=args.config)
    log.info("Launching Textual application")
    app.run()


def warn_if_legacy_stylesheet() -> None:
    """Log guidance if an old packaged stylesheet is still present."""

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    if css_resource.is_file():
        log.warning(
            "Detected legacy stylesheet at %s. This file is no longer used and"
            " belongs to an older installation. Reinstall the project with"
            " 'pip install -e .' (or run 'make install') to refresh the"
            " console script, or uninstall the previous package with"
            " 'pip uninstall streamdeck-tui'.",
            css_resource,
        )

if __name__ == "__main__":  # pragma: no cover
    main()
