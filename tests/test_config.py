from pathlib import Path

from streamdeck_tui.config import AppConfig, ProviderConfig, load_config, save_config


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
