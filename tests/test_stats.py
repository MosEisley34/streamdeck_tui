import pytest

from streamdeck_tui.app import _format_bitrate, resolution_tag_for_height
from streamdeck_tui.stats import StreamStatsAccumulator


def test_format_bitrate_formats_units() -> None:
    assert _format_bitrate(4_200_000) == "4.20 Mbps"
    assert _format_bitrate(1200) == "1.2 Kbps"
    assert _format_bitrate(None) == "–"


def test_resolution_tag_for_height() -> None:
    assert resolution_tag_for_height(2160) == ("2160p", "bright_magenta")
    assert resolution_tag_for_height(1080) == ("1080p", "green")
    assert resolution_tag_for_height(0) is None


def test_stream_stats_accumulator_tracks_average() -> None:
    accumulator = StreamStatsAccumulator()
    snapshot = accumulator.push_bitrate(2_000_000)
    assert snapshot.live_bitrate == 2_000_000
    assert snapshot.average_bitrate == pytest.approx(2_000_000)

    snapshot = accumulator.push_bitrate(4_000_000)
    assert snapshot.live_bitrate == 4_000_000
    assert snapshot.average_bitrate == pytest.approx(3_000_000)

    snapshot = accumulator.set_resolution(1920, 1080)
    assert snapshot.width == 1920
    assert snapshot.height == 1080
