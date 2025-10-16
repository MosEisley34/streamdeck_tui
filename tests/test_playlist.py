from pathlib import Path

import pytest

from streamdeck_tui.playlist import Channel, PlaylistError, filter_channels, parse_playlist


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


def test_parse_playlist_requires_metadata_for_stream_url():
    with pytest.raises(PlaylistError):
        parse_playlist(["#EXTM3U", "http://example.com"])


def test_parse_playlist_recovers_from_unterminated_attribute(monkeypatch):
    malformed_playlist = [
        "#EXTM3U",
        '#EXTINF:-1 tvg-id="channel1" group-title="News" tvg-logo="http://logo,Channel One',
        "http://example.com/stream1",
        '#EXTINF:-1 tvg-id="channel2" group-title="Sports",Channel Two',
        "http://example.com/stream2",
    ]

    warnings: list[str] = []

    def capture_warning(message: str, *args, **kwargs) -> None:
        warnings.append(message % args)

    monkeypatch.setattr("streamdeck_tui.playlist.log.warning", capture_warning)

    channels = parse_playlist(malformed_playlist)

    assert [channel.name for channel in channels] == ["Channel One", "Channel Two"]
    assert any("Unterminated attribute value" in message for message in warnings)
