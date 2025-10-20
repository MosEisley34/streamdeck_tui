"""Configuration management for Streamdeck TUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        value = value[1:-1]
    return value


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
            if provider.last_loaded_at:
                lines.append("    last_loaded_at: " + provider.last_loaded_at)
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
        playlist = entry.get("playlist_url")
        if not name or not playlist:
            log.warning("Skipping provider with missing fields: %s", entry)
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
        providers.append(
            ProviderConfig(
                name=str(name),
                playlist_url=str(playlist),
                api_url=str(entry.get("api_url")) if entry.get("api_url") is not None else None,
                last_loaded_at=last_loaded_at,
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
    "load_config",
    "save_config",
]
