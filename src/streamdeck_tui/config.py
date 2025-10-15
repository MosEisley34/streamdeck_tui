"""Configuration management for Streamdeck TUI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

CONFIG_PATH = Path.home() / ".config" / "streamdeck_tui" / "config.yaml"


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a single IPTV provider."""

    name: str
    playlist_url: str
    api_url: Optional[str] = None


@dataclass(slots=True)
class AppConfig:
    """Top level application configuration."""

    providers: list[ProviderConfig] = field(default_factory=list)

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

    def names(self) -> Iterable[str]:
        """Return provider names."""

        return [provider.name for provider in self.providers]


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
    if not data.providers:
        return "providers: []\n"
    lines = ["providers:"]
    for provider in data.providers:
        lines.append("  - name: " + provider.name)
        lines.append("    playlist_url: " + provider.playlist_url)
        if provider.api_url:
            lines.append("    api_url: " + provider.api_url)
    lines.append("")
    return "\n".join(lines)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from *path* or return an empty configuration."""

    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return AppConfig()
    raw = config_path.read_text(encoding="utf8")
    data = _parse_config(raw)
    providers_raw = data.get("providers", []) if isinstance(data, dict) else []
    providers: list[ProviderConfig] = []
    for entry in providers_raw:
        if not isinstance(entry, dict):  # pragma: no cover - invalid config guard
            continue
        name = entry.get("name")
        playlist = entry.get("playlist_url")
        if not name or not playlist:
            continue
        providers.append(
            ProviderConfig(
                name=str(name),
                playlist_url=str(playlist),
                api_url=str(entry.get("api_url")) if entry.get("api_url") is not None else None,
            )
        )
    return AppConfig(providers=providers)


def save_config(config: AppConfig, path: Optional[Path] = None) -> None:
    """Persist *config* to disk at *path*."""

    config_path = path or CONFIG_PATH
    _ensure_parent(config_path)
    config_path.write_text(_dump_config(config), encoding="utf8")


__all__ = [
    "AppConfig",
    "ProviderConfig",
    "CONFIG_PATH",
    "load_config",
    "save_config",
]
