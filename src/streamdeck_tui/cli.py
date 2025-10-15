"""Command line entry point for Streamdeck TUI."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .app import StreamdeckApp
from .config import CONFIG_PATH, load_config


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
    args = parse_args(argv)
    config = load_config(args.config)
    app = StreamdeckApp(config, config_path=args.config)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
