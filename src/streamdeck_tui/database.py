from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .playlist import Channel
from .logging_utils import get_logger

log = get_logger(__name__)

CHANNEL_DATABASE_PATH = Path.home() / ".cache" / "streamdeck_tui" / "channels.sqlite"


def _resolve_database_path(path: Optional[Path]) -> Path:
    if path is None:
        path = CHANNEL_DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS channels (
            provider TEXT NOT NULL,
            position INTEGER NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            channel_group TEXT,
            logo TEXT,
            attributes TEXT,
            PRIMARY KEY (provider, position)
        ) WITHOUT ROWID
        """
    )


def load_all_channels(path: Optional[Path] = None) -> Dict[str, List[Channel]]:
    target = _resolve_database_path(path)
    if not target.exists():
        return {}
    connection = sqlite3.connect(target)
    try:
        _ensure_schema(connection)
        cursor = connection.execute(
            """
            SELECT provider, name, url, channel_group, logo, attributes
            FROM channels
            ORDER BY provider, position
            """
        )
        results: Dict[str, List[Channel]] = {}
        for provider, name, url, group, logo, attributes_json in cursor:
            attributes = json.loads(attributes_json) if attributes_json else {}
            results.setdefault(provider, []).append(
                Channel(
                    name=name,
                    url=url,
                    group=group,
                    logo=logo,
                    raw_attributes=attributes,
                )
            )
        return results
    finally:
        connection.close()


def save_channels(
    provider: str, channels: Sequence[Channel], *, path: Optional[Path] = None
) -> None:
    target = _resolve_database_path(path)
    connection = sqlite3.connect(target)
    try:
        _ensure_schema(connection)
        with connection:
            connection.execute(
                "DELETE FROM channels WHERE provider = ?",
                (provider,),
            )
            if not channels:
                return
            connection.executemany(
                """
                INSERT INTO channels(
                    provider, position, name, url, channel_group, logo, attributes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        provider,
                        index,
                        channel.name,
                        channel.url,
                        channel.group,
                        channel.logo,
                        json.dumps(channel.raw_attributes, ensure_ascii=False, sort_keys=True)
                        if channel.raw_attributes
                        else None,
                    )
                    for index, channel in enumerate(channels)
                ],
            )
    finally:
        connection.close()


def remove_provider_channels(provider: str, path: Optional[Path] = None) -> None:
    target = _resolve_database_path(path)
    if not target.exists():
        return
    connection = sqlite3.connect(target)
    try:
        _ensure_schema(connection)
        with connection:
            connection.execute("DELETE FROM channels WHERE provider = ?", (provider,))
    finally:
        connection.close()


def clear_channel_database(path: Optional[Path] = None) -> None:
    target = _resolve_database_path(path)
    if target.exists():
        try:
            target.unlink()
        except OSError as exc:  # pragma: no cover - best effort cleanup
            log.warning("Failed to remove channel database at %s: %s", target, exc)


__all__ = [
    "CHANNEL_DATABASE_PATH",
    "clear_channel_database",
    "load_all_channels",
    "remove_provider_channels",
    "save_channels",
]
