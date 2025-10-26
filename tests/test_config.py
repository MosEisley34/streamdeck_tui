from pathlib import Path

from streamdeck_tui.config import (
    AppConfig,
    FavoriteChannel,
    ProviderConfig,
    build_xtream_playlist_variants,
    build_xtream_urls,
    extract_xtream_credentials,
    load_config,
    save_config,
)


def test_load_config_returns_empty_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = load_config(config_path)
    assert config.providers == []
    assert config.theme is None


def test_load_and_save_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = AppConfig(
        providers=[
            ProviderConfig(
                name="Provider A",
                playlist_url="https://example.com/a.m3u",
                api_url="https://example.com/a/status",
                last_loaded_at="2024-01-01T00:00:00+00:00",
                user_agent="Streamdeck/1.0",
                enable_vod=False,
            ),
            ProviderConfig(
                name="Provider B",
                playlist_url="https://example.com/b.m3u",
            ),
        ],
        theme="solarized-light",
    )
    save_config(config, config_path)
    raw = config_path.read_text()
    assert "theme: solarized-light" in raw.splitlines()[0]
    loaded = load_config(config_path)
    assert [provider.name for provider in loaded.providers] == ["Provider A", "Provider B"]
    assert loaded.providers[0].api_url == "https://example.com/a/status"
    assert loaded.providers[1].api_url is None
    assert loaded.providers[0].last_loaded_at == "2024-01-01T00:00:00+00:00"
    assert loaded.providers[1].last_loaded_at is None
    assert not loaded.providers[0].enable_vod
    assert loaded.providers[0].user_agent == "Streamdeck/1.0"
    assert loaded.providers[1].user_agent is None
    assert loaded.providers[1].enable_vod
    assert loaded.favorites == []
    assert loaded.theme == "solarized-light"


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


def test_xtream_credentials_generate_urls(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: Xtream Demo
    xtream_base_url: https://portal.example.com:8080/sub
    xtream_username: demo user
    xtream_password: secret pass
""".strip()
    )
    loaded = load_config(config_path)
    assert len(loaded.providers) == 1
    provider = loaded.providers[0]
    playlist, api = build_xtream_urls(
        "https://portal.example.com:8080/sub",
        "demo user",
        "secret pass",
    )
    assert provider.playlist_url == playlist
    assert provider.api_url == api
    assert provider.xtream_base_url == "https://portal.example.com:8080/sub"
    assert provider.xtream_username == "demo user"
    assert provider.xtream_password == "secret pass"


def test_build_xtream_playlist_variants_includes_scheme_and_formats() -> None:
    variants = build_xtream_playlist_variants(
        "http://portal.example.com", "user", "pass"
    )
    playlists = {playlist for playlist, *_ in variants}
    assert any(url.startswith("http://") for url in playlists)
    assert any(url.startswith("https://") for url in playlists)
    assert any("type=m3u" in url for url in playlists)
    assert any("output=mpegts" in url for url in playlists)


def test_build_xtream_urls_defaults_to_https_when_scheme_missing() -> None:
    playlist, api = build_xtream_urls("portal.example.com", "user", "pass")
    assert playlist.startswith("https://portal.example.com/")
    assert api.startswith("https://portal.example.com/")


def test_build_xtream_variants_without_scheme_adds_http_fallback() -> None:
    variants = build_xtream_playlist_variants("portal.example.com", "user", "pass")
    playlists = {playlist for playlist, *_ in variants}
    assert any(url.startswith("https://portal.example.com/") for url in playlists)
    assert any(url.startswith("http://portal.example.com/") for url in playlists)


def test_save_config_persists_xtream_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    playlist, api = build_xtream_urls("https://portal.example.com", "user", "pass")
    config = AppConfig(
        providers=[
            ProviderConfig(
                name="Xtream",
                playlist_url=playlist,
                api_url=api,
                xtream_base_url="https://portal.example.com",
                xtream_username="user",
                xtream_password="pass",
                user_agent="Streamdeck/2.0",
                enable_vod=False,
            )
        ]
    )
    save_config(config, config_path)
    raw = config_path.read_text()
    assert "xtream_base_url: https://portal.example.com" in raw
    assert "xtream_username: user" in raw
    assert "xtream_password: pass" in raw
    assert "user_agent: Streamdeck/2.0" in raw
    assert "enable_vod: false" in raw
    reloaded = load_config(config_path)
    assert reloaded.providers[0].xtream_base_url == "https://portal.example.com"
    assert reloaded.providers[0].xtream_username == "user"
    assert reloaded.providers[0].xtream_password == "pass"
    assert not reloaded.providers[0].enable_vod
    assert reloaded.providers[0].user_agent == "Streamdeck/2.0"


def test_extract_xtream_credentials_from_playlist() -> None:
    playlist, _ = build_xtream_urls("https://portal.example.com", "user", "pass")
    base, username, password = extract_xtream_credentials(playlist)
    assert base == "https://portal.example.com"
    assert username == "user"
    assert password == "pass"


def test_load_config_parses_enable_vod_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
providers:
  - name: Demo
    playlist_url: https://example.com/demo.m3u
    enable_vod: false
""".strip()
    )
    loaded = load_config(config_path)
    assert len(loaded.providers) == 1
    assert not loaded.providers[0].enable_vod
