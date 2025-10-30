"""Utilities for parsing IPTV playlists."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence
from typing import Literal, overload
from urllib import request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from .logging_utils import get_logger
from .config import build_xtream_urls, _normalize_xtream_base_url

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


class ChannelSearchIndex:
    """High-performance search index for large channel collections."""

    __slots__ = ("_conn", "_size")

    def __init__(self, connection: Optional[sqlite3.Connection], size: int) -> None:
        self._conn = connection
        self._size = size

    @classmethod
    def build(cls, channels: Sequence[Channel]) -> "ChannelSearchIndex":
        """Build an indexed search structure for *channels*."""

        total = len(channels)
        if total == 0:
            return cls(None, 0)

        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-64000")
        connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        connection.execute(
            """
            CREATE TABLE token_index (
                token TEXT NOT NULL,
                channel INTEGER NOT NULL,
                PRIMARY KEY (token, channel)
            ) WITHOUT ROWID
            """
        )

        connection.execute("BEGIN IMMEDIATE")
        batch: list[tuple[str, int]] = []
        batch_size = 50_000
        for index, channel in enumerate(channels):
            for token in channel._tokens:
                batch.append((token, index))
            if len(batch) >= batch_size:
                connection.executemany(
                    "INSERT OR IGNORE INTO token_index(token, channel) VALUES (?, ?)",
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT OR IGNORE INTO token_index(token, channel) VALUES (?, ?)",
                batch,
            )
        connection.commit()
        connection.execute("PRAGMA optimize")
        return cls(connection, total)

    def lookup(self, tokens: Sequence[str]) -> list[int]:
        """Return candidate channel indices that match *tokens*."""

        if not tokens:
            return list(range(self._size))
        if self._conn is None:
            return []
        placeholders = ",".join("?" for _ in tokens)
        query = (
            "SELECT channel FROM token_index WHERE token IN ("
            f"{placeholders}) GROUP BY channel HAVING COUNT(*) = ? ORDER BY channel"
        )
        cursor = self._conn.execute(query, (*tokens, len(tokens)))
        return [row[0] for row in cursor]

    def close(self) -> None:
        """Release resources held by the search index."""

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __len__(self) -> int:
        return self._size


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


def build_search_index(channels: Sequence[Channel]) -> ChannelSearchIndex:
    """Construct a high-performance search index for ``channels``."""

    return ChannelSearchIndex.build(channels)


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
    user_agent: Optional[str] = None,
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
        req = request.Request(source_str)
        if user_agent:
            req.add_header("User-Agent", user_agent)
        with request.urlopen(req, timeout=30.0) as response:
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
    search_index: Optional[Mapping[str, set[int]] | ChannelSearchIndex] = None,
    *,
    return_indices: bool = False,
) -> List[Channel] | List[int]:
    """Return channels matching the given search query."""

    tokens = normalize_tokens(query.strip())
    if not tokens:
        results = list(range(len(channels))) if return_indices else list(channels)
        log.debug("Filter query empty; returning %d channels", len(results))
        return results

    candidate_ids: Optional[Sequence[int] | set[int]] = None
    candidate_lookup: Optional[set[int]] = None
    if search_index is not None:
        if isinstance(search_index, ChannelSearchIndex):
            candidate_list = search_index.lookup(tokens)
            candidate_ids = candidate_list
            candidate_lookup = set(candidate_list)
        else:
            for token in tokens:
                postings = search_index.get(token)
                if not postings:
                    candidate_lookup = set()
                    candidate_ids = []
                    break
                if candidate_lookup is None:
                    candidate_lookup = set(postings)
                else:
                    candidate_lookup &= postings
                if not candidate_lookup:
                    break
            if candidate_ids is None:
                candidate_ids = sorted(candidate_lookup) if candidate_lookup is not None else None

    channel_results: List[Channel] = []
    index_results: List[int] = []
    if candidate_ids is not None:
        iterator = (
            (idx, channels[idx])
            for idx in candidate_ids
            if 0 <= idx < len(channels)
        )
    else:
        iterator = enumerate(channels)
        candidate_lookup = None
    for idx, channel in iterator:
        if candidate_lookup is not None and idx not in candidate_lookup:
            continue
        if channel.matches_tokens(tokens):
            if return_indices:
                index_results.append(idx)
            else:
                channel_results.append(channel)

    match_count = len(index_results) if return_indices else len(channel_results)
    log.debug("Filter query '%s' matched %d channel(s)", query, match_count)
    return index_results if return_indices else channel_results


def _xtream_request(url: str, *, user_agent: Optional[str]) -> object:
    """Return JSON decoded payload from an Xtream Codes endpoint."""

    headers = {"Accept": "application/json"}
    if user_agent:
        headers["User-Agent"] = user_agent
    req = request.Request(url, headers=headers)
    with request.urlopen(req) as response:
        payload = response.read()
    try:
        return json.loads(payload.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(payload.decode("latin-1"))


def _xtream_stream_extension(output: Optional[str]) -> str:
    normalized = (output or "ts").strip().lower()
    if normalized in {"ts", "mpegts"}:
        return "ts"
    if normalized in {"hls", "m3u8"}:
        return "m3u8"
    return normalized or "ts"


def load_xtream_api_channels(
    base_url: str,
    username: str,
    password: str,
    *,
    playlist_type: Optional[str] = None,
    output: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> List[Channel]:
    """Return live channels fetched through the Xtream JSON API."""

    normalized_base = _normalize_xtream_base_url(base_url)
    _, api_url = build_xtream_urls(
        normalized_base,
        username,
        password,
        playlist_type=playlist_type,
        output=output,
    )
    extension = _xtream_stream_extension(output)
    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")

    categories: dict[str, str] = {}
    categories_url = f"{api_url}&action=get_live_categories"
    try:
        category_payload = _xtream_request(categories_url, user_agent=user_agent)
    except (HTTPError, URLError, json.JSONDecodeError):
        log.debug("Failed to load Xtream categories from %s", categories_url)
    else:
        if isinstance(category_payload, list):
            for entry in category_payload:
                if not isinstance(entry, dict):
                    continue
                category_id = entry.get("category_id")
                name = entry.get("category_name") or entry.get("category")
                if category_id is None or not name:
                    continue
                categories[str(category_id)] = str(name)

    streams_url = f"{api_url}&action=get_live_streams"
    streams_payload = _xtream_request(streams_url, user_agent=user_agent)
    if not isinstance(streams_payload, list):
        raise PlaylistError("Unexpected Xtream response while loading channels")

    channels: List[Channel] = []
    for entry in streams_payload:
        if not isinstance(entry, dict):
            continue
        stream_id = entry.get("stream_id")
        name = entry.get("name")
        if stream_id is None or not name:
            continue
        category_name = entry.get("category_name")
        if not category_name and entry.get("category_id") is not None:
            category_name = categories.get(str(entry["category_id"]))
        logo = entry.get("stream_icon") or entry.get("stream_logo")
        epg_id = entry.get("epg_channel_id") or entry.get("epg_id")
        attributes: dict[str, str] = {}
        if category_name:
            attributes["group-title"] = str(category_name)
        if logo:
            attributes["tvg-logo"] = str(logo)
        if epg_id:
            attributes["tvg-id"] = str(epg_id)
        attributes["xtream-stream-id"] = str(stream_id)
        stream_url = (
            f"{normalized_base}/live/{encoded_username}/{encoded_password}/{stream_id}.{extension}"
        )
        channels.append(
            Channel(
                name=str(name),
                url=stream_url,
                group=str(category_name) if category_name else None,
                logo=str(logo) if logo else None,
                raw_attributes=attributes,
            )
        )

    log.info(
        "Loaded %d channel(s) via Xtream API from %s",
        len(channels),
        normalized_base,
    )
    return channels


__all__ = [
    "Channel",
    "ChannelSearchIndex",
    "PlaylistError",
    "parse_playlist",
    "load_playlist",
    "load_xtream_api_channels",
    "normalize_tokens",
    "build_search_index",
    "filter_channels",
]
