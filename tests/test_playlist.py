import logging
from pathlib import Path
from typing import Sequence

import pytest

from streamdeck_tui.playlist import (
    Channel,
    PlaylistError,
    build_search_index,
    filter_channels,
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
