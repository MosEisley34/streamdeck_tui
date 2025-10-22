"""Configuration management for Streamdeck TUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import parse_qs, quote_plus, urlparse, urlunparse

from .logging_utils import get_logger

CONFIG_PATH = Path.home() / ".config" / "streamdeck_tui" / "config.yaml"

log = get_logger(__name__)


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a single IPTV provider."""

    name: str
    playlist_url: str
    api_url: Optional[str] = None
    last_loaded_at: Optional[str] = None
    xtream_base_url: Optional[str] = None
    xtream_username: Optional[str] = None
    xtream_password: Optional[str] = None
    enable_vod: bool = True


@dataclass(slots=True)
class AppConfig:
    """Top level application configuration."""

    providers: list[ProviderConfig] = field(default_factory=list)
    favorites: list["FavoriteChannel"] = field(default_factory=list)

    def add_or_update(self, provider: ProviderConfig) -> None:
        """Insert or update a provider configuration by name."""

        for index, existing in enumerate(self.providers):
            if existing.name == provider.name:
                self.providers[index] = provider
                return
        self.providers.append(provider)

    def remove(self, provider_name: str) -> None:
        """Remove a provider by name if it exists."""

        self.providers = [p for p in self.providers if p.name != provider_name]
        self.favorites = [favorite for favorite in self.favorites if favorite.provider != provider_name]

    def names(self) -> Iterable[str]:
        """Return provider names."""

        return [provider.name for provider in self.providers]

    def find_favorite(self, provider_name: str, channel_url: str) -> Optional["FavoriteChannel"]:
        """Return the matching favorite entry if it exists."""

        for favorite in self.favorites:
            if favorite.provider == provider_name and favorite.channel_url == channel_url:
                return favorite
        return None

    def add_favorite(self, favorite: "FavoriteChannel") -> None:
        """Add a favorite if it is not already present."""

        if self.find_favorite(favorite.provider, favorite.channel_url) is None:
            self.favorites.append(favorite)

    def remove_favorite(self, provider_name: str, channel_url: str) -> None:
        """Remove a favorite channel if present."""

        self.favorites = [
            favorite
            for favorite in self.favorites
            if not (favorite.provider == provider_name and favorite.channel_url == channel_url)
        ]


@dataclass(slots=True)
class FavoriteChannel:
    """A persistent reference to a favorited channel."""

    provider: str
    channel_name: str
    channel_url: str
    group: Optional[str] = None
    logo: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        """Return a serializable representation of the favorite."""

        data: dict[str, Optional[str]] = {
            "provider": self.provider,
            "channel_name": self.channel_name,
            "channel_url": self.channel_url,
        }
        if self.group:
            data["group"] = self.group
        if self.logo:
            data["logo"] = self.logo
        return data


def _normalize_xtream_base_url(base_url: str) -> str:
    base = base_url.strip()
    if not base:
        raise ValueError("Xtream base URL cannot be empty")
    return base.rstrip("/")


def build_xtream_urls(base_url: str, username: str, password: str) -> Tuple[str, str]:
    """Return Xtream Codes playlist and API URLs for the provided credentials."""

    normalized = _normalize_xtream_base_url(base_url)
    user = quote_plus(username)
    passwd = quote_plus(password)
    playlist = (
        f"{normalized}/get.php?username={user}&password={passwd}&type=m3u_plus&output=ts"
    )
    api = f"{normalized}/player_api.php?username={user}&password={passwd}"
    return playlist, api


def build_xtream_playlist_variants(
    base_url: str, username: str, password: str
) -> list[tuple[str, str, str, str, str, str]]:
    """Return alternate Xtream playlist/API URL combinations for recovery attempts.

    The returned list contains tuples of ``(playlist_url, api_url, base_url, type, output, scheme)``
    ordered from most preferred to least preferred options.
    """

    normalized = _normalize_xtream_base_url(base_url)
    user = quote_plus(username)
    passwd = quote_plus(password)

    base_candidates: list[str] = []
    parsed = urlparse(normalized)
    if parsed.scheme:
        base_candidates.append(normalized)
        if parsed.scheme == "http":
            https_variant = urlunparse(parsed._replace(scheme="https"))
            if https_variant:
                base_candidates.append(https_variant.rstrip("/"))
    else:
        base_candidates.append(normalized)

    playlist_params = [
        ("m3u_plus", "ts"),
        ("m3u_plus", "mpegts"),
        ("m3u", "ts"),
        ("m3u", "mpegts"),
    ]

    variants: list[tuple[str, str, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate_base in base_candidates:
        stripped_base = candidate_base.rstrip("/")
        parsed_candidate = urlparse(stripped_base)
        scheme = parsed_candidate.scheme or parsed.scheme or "http"
        for playlist_type, output in playlist_params:
            playlist = (
                f"{stripped_base}/get.php?username={user}&password={passwd}"
                f"&type={playlist_type}&output={output}"
            )
            api = f"{stripped_base}/player_api.php?username={user}&password={passwd}"
            signature = (playlist, api)
            if signature in seen:
                continue
            seen.add(signature)
            variants.append((playlist, api, stripped_base, playlist_type, output, scheme))
    return variants


def extract_xtream_credentials(
    playlist_url: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract Xtream Codes credentials from a playlist URL if present."""

    try:
        parsed = urlparse(playlist_url)
    except ValueError:
        return None, None, None
    if not parsed.scheme or not parsed.netloc:
        return None, None, None
    if not parsed.path.endswith("get.php"):
        return None, None, None
    params = parse_qs(parsed.query)
    usernames = params.get("username")
    passwords = params.get("password")
    if not usernames or not passwords:
        return None, None, None
    username = usernames[0]
    password = passwords[0]
    base_path, _, _ = parsed.path.rpartition("/")
    base = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            base_path.rstrip("/"),
            "",
            "",
            "",
        )
    ).rstrip("/")
    if not base:
        base = f"{parsed.scheme}://{parsed.netloc}"
    return base or None, username or None, password or None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        value = value[1:-1]
    return value


def _parse_bool(value: object, *, default: bool = True) -> bool:
    """Coerce *value* into a boolean flag."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return default


def _parse_config(raw: str) -> dict[str, object]:
    if not raw.strip():
        return {}
    try:
        import json

        return json.loads(raw)
    except (json.JSONDecodeError, ModuleNotFoundError):
        pass

    result: dict[str, object] = {}
    current_list: Optional[list[dict[str, str]]] = None
    current_item: Optional[dict[str, str]] = None
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            key, _, remainder = line.partition(":")
            key = key.strip()
            value = remainder.strip()
            if value:
                result[key] = _clean_scalar(value)
                current_list = None
            else:
                current_list = []
                result[key] = current_list
            current_item = None
            continue
        if line.strip().startswith("-"):
            current_item = {}
            if current_list is not None:
                current_list.append(current_item)
            remainder = line.strip()[1:].strip()
            if remainder and current_item is not None:
                key, _, value = remainder.partition(":")
                current_item[key.strip()] = _clean_scalar(value)
            continue
        if current_item is not None:
            key, _, value = line.strip().partition(":")
            current_item[key.strip()] = _clean_scalar(value)
    return result


def _dump_config(data: AppConfig) -> str:
    lines: list[str] = []
    if data.providers:
        lines.append("providers:")
        for provider in data.providers:
            lines.append("  - name: " + provider.name)
            lines.append("    playlist_url: " + provider.playlist_url)
            if provider.api_url:
                lines.append("    api_url: " + provider.api_url)
            if provider.xtream_base_url:
                lines.append("    xtream_base_url: " + provider.xtream_base_url)
            if provider.xtream_username:
                lines.append("    xtream_username: " + provider.xtream_username)
            if provider.xtream_password:
                lines.append("    xtream_password: " + provider.xtream_password)
            if provider.last_loaded_at:
                lines.append("    last_loaded_at: " + provider.last_loaded_at)
            if not provider.enable_vod:
                lines.append("    enable_vod: false")
    else:
        lines.append("providers: []")
    if data.favorites:
        lines.append("favorites:")
        for favorite in data.favorites:
            lines.append("  - provider: " + favorite.provider)
            lines.append("    channel_name: " + favorite.channel_name)
            lines.append("    channel_url: " + favorite.channel_url)
            if favorite.group:
                lines.append("    group: " + favorite.group)
            if favorite.logo:
                lines.append("    logo: " + favorite.logo)
    else:
        lines.append("favorites: []")
    lines.append("")
    return "\n".join(lines)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from *path* or return an empty configuration."""

    config_path = path or CONFIG_PATH
    if not config_path.exists():
        log.info("Configuration file missing at %s; using defaults", config_path)
        return AppConfig()
    log.debug("Loading configuration from %s", config_path)
    raw = config_path.read_text(encoding="utf8")
    data = _parse_config(raw)
    providers_raw = data.get("providers", []) if isinstance(data, dict) else []
    favorites_raw = data.get("favorites", []) if isinstance(data, dict) else []
    providers: list[ProviderConfig] = []
    favorites: list[FavoriteChannel] = []
    for entry in providers_raw:
        if not isinstance(entry, dict):  # pragma: no cover - invalid config guard
            continue
        name = entry.get("name")
        playlist_raw = entry.get("playlist_url")
        api_raw = entry.get("api_url")
        base_raw = entry.get("xtream_base_url")
        username_raw = entry.get("xtream_username")
        password_raw = entry.get("xtream_password")

        playlist: Optional[str] = None
        if isinstance(playlist_raw, str) and playlist_raw.strip():
            playlist = playlist_raw.strip()

        api_url: Optional[str] = None
        if isinstance(api_raw, str):
            candidate = api_raw.strip()
            if candidate:
                api_url = candidate

        xtream_base = base_raw.strip() if isinstance(base_raw, str) else None
        xtream_username = username_raw.strip() if isinstance(username_raw, str) else None
        xtream_password = password_raw.strip() if isinstance(password_raw, str) else None
        has_xtream_credentials = all(
            value for value in (xtream_base, xtream_username, xtream_password)
        )
        if has_xtream_credentials:
            try:
                generated_playlist, generated_api = build_xtream_urls(
                    xtream_base or "",
                    xtream_username or "",
                    xtream_password or "",
                )
            except ValueError:
                log.warning("Invalid Xtream configuration for provider %s", name or "<unknown>")
            else:
                if not playlist:
                    playlist = generated_playlist
                if not api_url:
                    api_url = generated_api
        else:
            xtream_base = None
            xtream_username = None
            xtream_password = None

        if not name or not playlist:
            log.warning(
                "Skipping provider with missing fields (name present: %s, playlist present: %s)",
                bool(name),
                bool(playlist),
            )
            continue
        last_loaded_raw = entry.get("last_loaded_at")
        last_loaded_at: Optional[str] = None
        if isinstance(last_loaded_raw, str):
            candidate = last_loaded_raw.strip()
            if candidate:
                try:
                    datetime.fromisoformat(candidate)
                except ValueError:
                    log.warning(
                        "Ignoring invalid last_loaded_at timestamp for provider %s", name
                    )
                else:
                    last_loaded_at = candidate
        enable_vod = _parse_bool(entry.get("enable_vod"), default=True)
        providers.append(
            ProviderConfig(
                name=str(name),
                playlist_url=str(playlist),
                api_url=api_url,
                last_loaded_at=last_loaded_at,
                xtream_base_url=xtream_base,
                xtream_username=xtream_username,
                xtream_password=xtream_password,
                enable_vod=enable_vod,
            )
        )
    for entry in favorites_raw:
        if not isinstance(entry, dict):  # pragma: no cover - invalid config guard
            continue
        provider = entry.get("provider")
        channel_url = entry.get("channel_url") or entry.get("url")
        channel_name = entry.get("channel_name") or entry.get("name")
        if not provider or not channel_url:
            log.warning("Skipping favorite with missing fields: %s", entry)
            continue
        favorites.append(
            FavoriteChannel(
                provider=str(provider),
                channel_name=str(channel_name) if channel_name is not None else str(channel_url),
                channel_url=str(channel_url),
                group=str(entry.get("group")) if entry.get("group") is not None else None,
                logo=str(entry.get("logo")) if entry.get("logo") is not None else None,
            )
        )
    log.info(
        "Loaded %d providers and %d favorite(s) from %s",
        len(providers),
        len(favorites),
        config_path,
    )
    return AppConfig(providers=providers, favorites=favorites)


def save_config(config: AppConfig, path: Optional[Path] = None) -> None:
    """Persist *config* to disk at *path*."""

    config_path = path or CONFIG_PATH
    log.debug("Writing configuration with %d providers to %s", len(config.providers), config_path)
    _ensure_parent(config_path)
    config_path.write_text(_dump_config(config), encoding="utf8")
    log.info("Configuration saved to %s", config_path)


__all__ = [
    "AppConfig",
    "ProviderConfig",
    "FavoriteChannel",
    "CONFIG_PATH",
    "build_xtream_urls",
    "build_xtream_playlist_variants",
    "extract_xtream_credentials",
    "load_config",
    "save_config",
]
