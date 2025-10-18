"""Utilities for parsing IPTV playlists."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence
from typing import Literal, overload
from urllib import request

from .logging_utils import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Channel:
    """A parsed IPTV channel entry."""

    name: str
    url: str
    group: Optional[str] = None
    logo: Optional[str] = None
    raw_attributes: dict[str, str] = field(default_factory=dict)
    _tokens: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parts = [
            self.name,
            self.group or "",
            self.raw_attributes.get("tvg-id") or "",
        ]
        token_set: set[str] = set()
        for part in parts:
            if part:
                token_set.update(normalize_tokens(part))
        self._tokens = frozenset(token_set)

    def matches_tokens(self, tokens: Sequence[str]) -> bool:
        """Return True if all tokens match the precomputed search blob."""

        if not tokens:
            return True
        return all(token in self._tokens for token in tokens)

    def matches(self, query: str) -> bool:
        """Return True if the channel matches the search query."""

        return self.matches_tokens(normalize_tokens(query))


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
            value = "".join(buffer)
            if i >= len(metadata):
                attributes[current_key] = value
                log.warning(
                    "Unterminated attribute value for %s; using remainder '%s'",
                    current_key,
                    value,
                )
                current_key = None
                break
            i += 1
            if i < len(metadata) and not metadata[i].isspace():
                remainder = metadata[i:]
                if remainder:
                    value = f"{value}{remainder}"
                attributes[current_key] = value
                log.warning(
                    "Unterminated attribute value for %s; using remainder '%s'",
                    current_key,
                    value,
                )
                current_key = None
                break
            attributes[current_key] = value
            log.debug("Parsed attribute %s=%s", current_key, value)
            current_key = None
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


def normalize_tokens(text: str) -> list[str]:
    """Normalize ``text`` into a list of unique, lowercase search tokens."""

    if not text:
        return []
    normalized = unicodedata.normalize("NFKD", text)
    without_diacritics = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    raw_tokens = re.findall(r"\w+", without_diacritics.lower())
    seen: set[str] = set()
    tokens: list[str] = []
    for token in raw_tokens:
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def build_search_index(channels: Sequence[Channel]) -> dict[str, set[int]]:
    """Construct an inverted index mapping tokens to channel positions."""

    index: dict[str, set[int]] = {}
    for idx, channel in enumerate(channels):
        for token in channel._tokens:
            index.setdefault(token, set()).add(idx)
    return index


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
        channel = Channel(
            name=current_name,
            url=line,
            group=current_attrs.get("group-title"),
            logo=current_attrs.get("tvg-logo"),
            raw_attributes=current_attrs,
        )
        channels.append(channel)
        log.debug("Added channel %s (%s)", channel.name, channel.url)
        current_attrs = None
        current_name = None

    log.info("Parsed %d channels from playlist", len(channels))
    return channels


def load_playlist(
    source: str | Path,
    *,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> List[Channel]:
    """Load and parse a playlist from a local path or URL."""

    def report(loaded: int, total: Optional[int]) -> None:
        if progress is None:
            return
        try:
            progress(loaded, total)
        except Exception:  # pragma: no cover - diagnostic safeguard
            log.exception("Progress callback failed")

    source_str = str(source)
    log.info("Loading playlist from %s", source_str)
    chunk_size = 64_000
    data = bytearray()

    if source_str.startswith(("http://", "https://")):
        with request.urlopen(source_str, timeout=30.0) as response:
            total = getattr(response, "length", None)
            if total is None:
                length_header = response.headers.get("Content-Length")
                if length_header:
                    try:
                        total = int(length_header)
                    except ValueError:
                        total = None
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                data.extend(chunk)
                report(len(data), total)
        report(len(data), total)
        text = data.decode("utf8", errors="replace").splitlines()
        log.debug("Downloaded playlist bytes: %d", len(data))
        return parse_playlist(text)

    path = Path(source)
    if not path.exists():
        raise PlaylistError(f"Playlist path not found: {path}")
    total = path.stat().st_size
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            data.extend(chunk)
            report(len(data), total)
    report(len(data), total)
    log.debug("Read playlist file %s (%d bytes)", path, len(data))
    return parse_playlist(data.decode("utf8", errors="replace").splitlines())


@overload
def filter_channels(
    channels: Sequence[Channel],
    query: str,
    search_index: Optional[Mapping[str, set[int]]] = None,
    *,
    return_indices: Literal[False] = False,
) -> List[Channel]:
    ...


@overload
def filter_channels(
    channels: Sequence[Channel],
    query: str,
    search_index: Optional[Mapping[str, set[int]]] = None,
    *,
    return_indices: Literal[True],
) -> List[int]:
    ...


def filter_channels(
    channels: Sequence[Channel],
    query: str,
    search_index: Optional[Mapping[str, set[int]]] = None,
    *,
    return_indices: bool = False,
) -> List[Channel] | List[int]:
    """Return channels matching the given search query."""

    tokens = normalize_tokens(query.strip())
    if not tokens:
        results = list(range(len(channels))) if return_indices else list(channels)
        log.debug("Filter query empty; returning %d channels", len(results))
        return results

    candidate_ids: Optional[set[int]] = None
    if search_index is not None:
        for token in tokens:
            postings = search_index.get(token)
            if not postings:
                candidate_ids = set()
                break
            if candidate_ids is None:
                candidate_ids = set(postings)
            else:
                candidate_ids &= postings
            if not candidate_ids:
                break

    channel_results: List[Channel] = []
    index_results: List[int] = []
    for idx, channel in enumerate(channels):
        if candidate_ids is not None and idx not in candidate_ids:
            continue
        if channel.matches_tokens(tokens):
            if return_indices:
                index_results.append(idx)
            else:
                channel_results.append(channel)

    match_count = len(index_results) if return_indices else len(channel_results)
    log.debug("Filter query '%s' matched %d channel(s)", query, match_count)
    return index_results if return_indices else channel_results


__all__ = [
    "Channel",
    "PlaylistError",
    "parse_playlist",
    "load_playlist",
    "normalize_tokens",
    "build_search_index",
    "filter_channels",
]
