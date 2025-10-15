"""Utilities for parsing IPTV playlists."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional
from urllib import request


@dataclass(slots=True)
class Channel:
    """A parsed IPTV channel entry."""

    name: str
    url: str
    group: Optional[str] = None
    logo: Optional[str] = None
    raw_attributes: dict[str, str] = field(default_factory=dict)

    def matches(self, query: str) -> bool:
        """Return True if the channel matches the search query."""

        tokens = query.lower().split()
        haystack = " ".join(
            filter(
                None,
                [
                    self.name.lower(),
                    (self.group or "").lower(),
                    (self.raw_attributes.get("tvg-id") or "").lower(),
                ],
            )
        )
        return all(token in haystack for token in tokens)


class PlaylistError(RuntimeError):
    """Raised when a playlist cannot be parsed."""


def _parse_extinf(line: str) -> tuple[dict[str, str], str]:
    """Parse an ``#EXTINF`` line."""

    if not line.startswith("#EXTINF:"):
        raise PlaylistError("Expected #EXTINF line")

    payload = line[len("#EXTINF:") :]
    if "," not in payload:
        raise PlaylistError("Invalid #EXTINF line")
    metadata, name = payload.split(",", 1)
    attributes: dict[str, str] = {}
    current_key: Optional[str] = None
    buffer: List[str] = []
    i = 0
    while i < len(metadata):
        char = metadata[i]
        if char == " ":
            i += 1
            continue
        if char == "=":
            if current_key is None:
                raise PlaylistError("Malformed EXTINF attribute")
            if i + 1 >= len(metadata) or metadata[i + 1] != '"':
                raise PlaylistError("Attribute values must be quoted")
            i += 2
            buffer.clear()
            while i < len(metadata) and metadata[i] != '"':
                buffer.append(metadata[i])
                i += 1
            if i >= len(metadata):
                raise PlaylistError("Unterminated attribute value")
            attributes[current_key] = "".join(buffer)
            current_key = None
            i += 1
        elif char == ":":
            # duration; skip until next space or end
            i += 1
            while i < len(metadata) and metadata[i] not in " ":
                i += 1
        else:
            buffer.clear()
            while i < len(metadata) and metadata[i] not in "= ":
                buffer.append(metadata[i])
                i += 1
            current_key = "".join(buffer)
    return attributes, name.strip()


def parse_playlist(lines: Iterable[str]) -> List[Channel]:
    """Parse playlist lines into a list of :class:`Channel` objects."""

    iterator = iter(lines)
    try:
        first = next(iterator).strip()
    except StopIteration as exc:  # pragma: no cover - defensive
        raise PlaylistError("Playlist is empty") from exc
    if not first.startswith("#EXTM3U"):
        raise PlaylistError("Not an extended M3U playlist")

    channels: List[Channel] = []
    current_attrs: Optional[dict[str, str]] = None
    current_name: Optional[str] = None

    for raw_line in iterator:
        line = raw_line.strip()
        if not line or line.startswith("#EXTM3U"):
            continue
        if line.startswith("#EXTINF:"):
            current_attrs, current_name = _parse_extinf(line)
            continue
        if line.startswith("#"):
            # ignore other metadata
            continue
        if current_name is None or current_attrs is None:
            raise PlaylistError("Found stream URL without metadata")
        channels.append(
            Channel(
                name=current_name,
                url=line,
                group=current_attrs.get("group-title"),
                logo=current_attrs.get("tvg-logo"),
                raw_attributes=current_attrs,
            )
        )
        current_attrs = None
        current_name = None

    return channels


def load_playlist(source: str | Path) -> List[Channel]:
    """Load and parse a playlist from a local path or URL."""

    source_str = str(source)
    if source_str.startswith(("http://", "https://")):
        with request.urlopen(source_str, timeout=30.0) as response:
            content = response.read().decode("utf8", errors="replace")
        text = content.splitlines()
        return parse_playlist(text)
    path = Path(source)
    if not path.exists():
        raise PlaylistError(f"Playlist path not found: {path}")
    return parse_playlist(path.read_text(encoding="utf8").splitlines())


def filter_channels(channels: Iterable[Channel], query: str) -> List[Channel]:
    """Return channels matching the given search query."""

    query = query.strip()
    if not query:
        return list(channels)
    return [channel for channel in channels if channel.matches(query)]


__all__ = [
    "Channel",
    "PlaylistError",
    "parse_playlist",
    "load_playlist",
    "filter_channels",
]
