import logging
from email.message import Message
from pathlib import Path
from typing import Sequence

import pytest

from streamdeck_tui.playlist import (
    Channel,
    PlaylistError,
    build_search_index,
    filter_channels,
    load_playlist,
    load_xtream_api_channels,
    normalize_tokens,
    parse_playlist,
)


SAMPLE_PLAYLIST = """#EXTM3U
#EXTINF:-1 tvg-id="channel1" group-title="News",Channel One
http://example.com/stream1
#EXTINF:-1 tvg-id="channel2" group-title="Sports" tvg-logo="http://logo",Channel Two
http://example.com/stream2
""".splitlines()


def test_parse_playlist_success():
    channels = parse_playlist(SAMPLE_PLAYLIST)
    assert len(channels) == 2
    first = channels[0]
    assert first.name == "Channel One"
    assert first.group == "News"
    assert first.raw_attributes["tvg-id"] == "channel1"


def test_parse_playlist_requires_extm3u():
    with pytest.raises(PlaylistError):
        parse_playlist(["#EXTINF:-1,Missing header", "http://example.com"])


def test_filter_channels_matches_multiple_fields():
    channels = parse_playlist(SAMPLE_PLAYLIST)
    result = filter_channels(channels, "Sports")
    assert [channel.name for channel in result] == ["Channel Two"]
    result = filter_channels(channels, "channel2")
    assert [channel.name for channel in result] == ["Channel Two"]
    result = filter_channels(channels, "sPoRtS")
    assert [channel.name for channel in result] == ["Channel Two"]


def test_filter_channels_reuses_normalized_tokens(monkeypatch):
    channels = parse_playlist(SAMPLE_PLAYLIST)
    captured_tokens: list[Sequence[str]] = []
    original = Channel.matches_tokens

    def spy(self: Channel, tokens):  # type: ignore[override]
        captured_tokens.append(tokens)
        return original(self, tokens)

    monkeypatch.setattr(Channel, "matches_tokens", spy)

    result = filter_channels(channels, "Sports channel2")

    assert [channel.name for channel in result] == ["Channel Two"]
    assert captured_tokens
    assert all(token == token.lower() for tokens in captured_tokens for token in tokens)
    first_tokens = captured_tokens[0]
    assert all(tokens is first_tokens for tokens in captured_tokens)


def test_filter_channels_can_return_indices() -> None:
    channels = parse_playlist(SAMPLE_PLAYLIST)
    indices = filter_channels(channels, "Channel", return_indices=True)
    assert indices == [0, 1]
    sports_only = filter_channels(channels, "Sports", return_indices=True)
    assert sports_only == [1]
    empty = filter_channels(channels, "", return_indices=True)
    assert empty == [0, 1]


def test_normalize_tokens_handles_punctuation_and_diacritics():
    tokens = normalize_tokens("ESPN 1080 / 1080 ESPN")
    assert tokens == ["espn", "1080"]
    assert normalize_tokens("Café-Éclair") == ["cafe", "eclair"]


def test_filter_channels_uses_inverted_index(monkeypatch):
    channels = parse_playlist(SAMPLE_PLAYLIST)
    index = build_search_index(channels)
    captured: list[str] = []
    original = Channel.matches_tokens

    def spy(self: Channel, tokens):  # type: ignore[override]
        captured.append(self.name)
        return original(self, tokens)

    monkeypatch.setattr(Channel, "matches_tokens", spy)

    result = filter_channels(channels, "Sports", index)

    assert [channel.name for channel in result] == ["Channel Two"]
    assert captured == ["Channel Two"]


def test_parse_playlist_requires_metadata_for_stream_url():
    with pytest.raises(PlaylistError):
        parse_playlist(["#EXTM3U", "http://example.com"])


def test_parse_playlist_recovers_from_unterminated_attribute(monkeypatch):
    malformed_playlist = [
        "#EXTM3U",
        '#EXTINF:-1 tvg-id="channel1" group-title="News" tvg-logo="http://logo,Channel One',
        "http://example.com/stream1",
        '#EXTINF:-1 tvg-id="channel2" group-title="Sports" tvg-logo="http://logo2 tvg-name="Channel Two",Channel Two',
        "http://example.com/stream2",
        '#EXTINF:-1 tvg-id="channel3" group-title="Movies",Channel Three',
        "http://example.com/stream3",
    ]

    warnings: list[str] = []

    def capture_warning(message: str, *args, **kwargs) -> None:
        warnings.append(message % args)

    monkeypatch.setattr("streamdeck_tui.playlist.log.warning", capture_warning)

    channels = parse_playlist(malformed_playlist)

    assert [channel.name for channel in channels] == [
        "Channel One",
        "Channel Two",
        "Channel Three",
    ]
    unterminated_messages = [
        message for message in warnings if "Unterminated attribute value" in message
    ]
    assert len(unterminated_messages) == 2

    second_channel = channels[1]
    assert (
        second_channel.raw_attributes["tvg-logo"]
        == 'http://logo2 tvg-name=Channel Two"'
    )
    assert "tvg-name" not in second_channel.raw_attributes


def test_load_playlist_uses_custom_user_agent(monkeypatch):
    playlist_bytes = """#EXTM3U
#EXTINF:-1,Example
http://stream.example/1
""".encode("utf8")

    class DummyResponse:
        def __init__(self) -> None:
            self._buffer = playlist_bytes
            self._offset = 0
            self.headers = {"Content-Length": str(len(self._buffer))}
            self.length = len(self._buffer)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, size: int) -> bytes:
            if self._offset >= len(self._buffer):
                return b""
            chunk = self._buffer[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    captured: dict[str, str | None] = {}

    def fake_urlopen(req, timeout: float = 0.0):  # pragma: no cover - network shim
        captured["timeout"] = timeout
        captured["user_agent"] = req.get_header("User-agent")
        return DummyResponse()

    monkeypatch.setattr("streamdeck_tui.playlist.request.urlopen", fake_urlopen)

    channels = load_playlist(
        "https://example.com/playlist.m3u",
        user_agent="Streamdeck/3.0",
    )
    assert [channel.name for channel in channels] == ["Example"]
    assert captured["user_agent"] == "Streamdeck/3.0"


def test_load_xtream_api_channels(monkeypatch) -> None:
    base = "https://portal.example"
    username = "user"
    password = "pass"

    categories_payload = [
        {"category_id": "1", "category_name": "News"},
        {"category_id": "2", "category_name": "Sports"},
    ]
    streams_payload = [
        {
            "stream_id": 101,
            "name": "Channel One",
            "category_id": "1",
            "stream_icon": "http://logo1",
            "epg_channel_id": "chan1",
        },
        {
            "stream_id": 202,
            "name": "Channel Two",
            "category_name": "Sports",
        },
    ]

    responses = {
        f"{base}/player_api.php?username={username}&password={password}&action=get_live_categories": categories_payload,
        f"{base}/player_api.php?username={username}&password={password}&action=get_live_streams": streams_payload,
    }

    class DummyResponse:
        def __init__(self, payload: object) -> None:
            self._payload = payload
            self.headers = Message()
            self.headers.add_header("Content-Type", "application/json; charset=utf-8")

        def __enter__(self) -> "DummyResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def read(self) -> bytes:
            import json

            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(req, timeout: float = 0.0):  # pragma: no cover - shim
        url = req.full_url if hasattr(req, "full_url") else req
        if url not in responses:
            raise AssertionError(f"Unexpected URL requested: {url}")
        return DummyResponse(responses[url])

    monkeypatch.setattr("streamdeck_tui.playlist.request.urlopen", fake_urlopen)

    channels = load_xtream_api_channels(
        base,
        username,
        password,
        output="hls",
    )

    assert [channel.name for channel in channels] == ["Channel One", "Channel Two"]
    first, second = channels
    assert first.group == "News"
    assert first.logo == "http://logo1"
    assert first.raw_attributes["tvg-id"] == "chan1"
    assert first.url.endswith("/live/user/pass/101.m3u8")
    assert second.group == "Sports"
