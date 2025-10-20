from pathlib import Path

from streamdeck_tui.config import (
    AppConfig,
    FavoriteChannel,
    ProviderConfig,
    load_config,
    save_config,
)


def test_load_config_returns_empty_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = load_config(config_path)
    assert config.providers == []


def test_load_and_save_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = AppConfig(
        providers=[
            ProviderConfig(
                name="Provider A",
                playlist_url="https://example.com/a.m3u",
                api_url="https://example.com/a/status",
                last_loaded_at="2024-01-01T00:00:00+00:00",
            ),
            ProviderConfig(
                name="Provider B",
                playlist_url="https://example.com/b.m3u",
            ),
        ]
    )
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert [provider.name for provider in loaded.providers] == ["Provider A", "Provider B"]
    assert loaded.providers[0].api_url == "https://example.com/a/status"
    assert loaded.providers[1].api_url is None
    assert loaded.providers[0].last_loaded_at == "2024-01-01T00:00:00+00:00"
    assert loaded.providers[1].last_loaded_at is None
    assert loaded.favorites == []


def test_round_trip_with_favorites(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = AppConfig(
        providers=[
            ProviderConfig(name="Provider", playlist_url="https://example.com/a.m3u"),
        ],
        favorites=[
            FavoriteChannel(
                provider="Provider",
                channel_name="Demo Channel",
                channel_url="https://stream.example/demo",
                group="News",
                logo="https://example.com/logo.png",
            )
        ],
    )
    save_config(config, config_path)
    loaded = load_config(config_path)
    assert loaded.favorites
    favorite = loaded.favorites[0]
    assert favorite.provider == "Provider"
    assert favorite.channel_name == "Demo Channel"
    assert favorite.channel_url == "https://stream.example/demo"
    assert favorite.group == "News"
    assert favorite.logo == "https://example.com/logo.png"


def test_invalid_favorites_are_skipped(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: Demo
    playlist_url: https://example.com/demo.m3u
favorites:
  - provider: Demo
    channel_name: Missing URL
  - provider: Demo
    channel_name: Valid Channel
    channel_url: https://stream.example/valid
""".strip()
    )
    loaded = load_config(config_path)
    assert len(loaded.favorites) == 1
    assert loaded.favorites[0].channel_name == "Valid Channel"


def test_invalid_last_loaded_at_is_ignored(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: Demo
    playlist_url: https://example.com/demo.m3u
    last_loaded_at: invalid timestamp
""".strip()
    )
    loaded = load_config(config_path)
    assert loaded.providers[0].last_loaded_at is None
