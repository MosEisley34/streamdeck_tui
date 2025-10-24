from __future__ import annotations

from streamdeck_tui import database
from streamdeck_tui.playlist import Channel


def test_channel_database_round_trip(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "channels.sqlite"
    monkeypatch.setattr(database, "CHANNEL_DATABASE_PATH", db_path)

    provider_one = "Provider One"
    provider_two = "Provider Two"
    channels_one = [
        Channel(
            name="Channel A",
            url="http://example.com/a",
            group="News",
            logo="http://example.com/logo_a.png",
            raw_attributes={"tvg-id": "a"},
        ),
        Channel(
            name="Channel B",
            url="http://example.com/b",
            group=None,
            logo=None,
            raw_attributes={},
        ),
    ]
    channels_two = [
        Channel(
            name="Channel C",
            url="http://example.com/c",
            group="Sports",
            logo=None,
            raw_attributes={"tvg-id": "c"},
        )
    ]

    database.save_channels(provider_one, channels_one)
    database.save_channels(provider_two, channels_two)

    loaded = database.load_all_channels()
    assert set(loaded.keys()) == {provider_one, provider_two}
    assert [channel.name for channel in loaded[provider_one]] == ["Channel A", "Channel B"]
    assert loaded[provider_one][0].raw_attributes["tvg-id"] == "a"
    assert loaded[provider_two][0].group == "Sports"

    database.remove_provider_channels(provider_one)
    loaded_after_removal = database.load_all_channels()
    assert provider_one not in loaded_after_removal
    assert provider_two in loaded_after_removal

    database.clear_channel_database()
    assert not db_path.exists()
