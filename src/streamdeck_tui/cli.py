"""Command line entry point for Streamdeck TUI."""
from __future__ import annotations

import argparse
import importlib.resources as resources
from pathlib import Path
from typing import Iterable

from . import __version__
from .app import StreamdeckApp
from .config import CONFIG_PATH, load_config
from .logging_utils import configure_logging, get_log_file_path, get_logger
from .player import PREFERRED_PLAYER_DEFAULT
from .themes import CUSTOM_THEMES

log = get_logger(__name__)


def _sorted_theme_names() -> list[str]:
    """Return the bundled theme catalog in a consistent order."""

    return sorted(CUSTOM_THEMES)


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
        default=PREFERRED_PLAYER_DEFAULT,
        help=(
            "Preferred media player executable to launch (default: %(default)s;"
            " falls back to auto-detect)"
        ),
    )
    theme_names = ", ".join(_sorted_theme_names())
    parser.add_argument(
        "--theme",
        default=None,
        help=(
            "Select the application theme. Available options: "
            f"{theme_names}."
        ),
    )
    parser.add_argument(
        "--list-themes",
        action="store_true",
        help="List available themes and exit.",
    )
    parser.add_argument(
        "--provider-logs",
        metavar="NAME",
        default=None,
        help="Print log entries mentioning the given provider name and exit.",
    )
    return parser.parse_args(argv)


def _print_provider_logs(provider_name: str) -> None:
    """Write log entries that reference *provider_name* to stdout."""

    log_path = get_log_file_path()
    if log_path is None:
        print("File logging is not enabled; set --log-file or STREAMDECK_TUI_LOG_FILE.")
        return

    if not log_path.exists():
        print(f"No log file found at {log_path}")
        return

    provider_token = provider_name.lower()
    matches = 0

    print(f"Log file: {log_path}")
    with log_path.open("r", encoding="utf8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if provider_token in line.lower():
                print(line)
                matches += 1

    if matches == 0:
        print(f"No log entries mentioning '{provider_name}' were found.")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list_themes:
        for theme_name in _sorted_theme_names():
            print(theme_name)
        return
    configure_logging(
        level=args.log_level,
        log_file=str(args.log_file) if args.log_file is not None else None,
    )
    if args.provider_logs:
        _print_provider_logs(args.provider_logs)
        return
    warn_if_legacy_stylesheet()
    log.info("CLI invoked with config=%s", args.config)
    config = load_config(args.config)
    app = StreamdeckApp(
        config,
        config_path=args.config,
        preferred_player=args.preferred_player,
        theme=args.theme,
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
