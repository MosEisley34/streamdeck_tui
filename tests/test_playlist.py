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


def test_parse_playlist_allows_commas_in_attribute_values():
    playlist = """#EXTM3U
#EXTINF:-1 tvg-id="" tvg-name="Formula 1 Alexander Albon (Williams, Thailand)" tvg-logo="https://example/logo.png" group-title="FORMULA 1",Formula 1 Alexander Albon (Williams, Thailand)
https://example.com/channel
""".splitlines()
    channels = parse_playlist(playlist)
    assert channels[0].name == "Formula 1 Alexander Albon (Williams, Thailand)"
    assert channels[0].raw_attributes["tvg-logo"] == "https://example/logo.png"


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

def test_parse_playlist_without_display_name_uses_tvg_name():
    playlist = """#EXTM3U
#EXTINF:-1 tvg-id="id-1" tvg-name="Flo (FLSP) 943" group-title="Sports"
https://example.com/channel
""".splitlines()
    channels = parse_playlist(playlist)
    assert channels[0].name == "Flo (FLSP) 943"
    assert channels[0].raw_attributes["tvg-id"] == "id-1"


def test_parse_playlist_without_any_name_defaults_to_unknown():
    playlist = """#EXTM3U
#EXTINF:-1 group-title="Misc"
https://example.com/channel
""".splitlines()
    channels = parse_playlist(playlist)
    assert channels[0].name == "Unknown"
    assert channels[0].group == "Misc"
