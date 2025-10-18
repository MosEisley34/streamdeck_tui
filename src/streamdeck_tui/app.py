"""Textual application implementing the IPTV management TUI."""
from __future__ import annotations

import sys
from collections import deque
from contextlib import nullcontext
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow running this module directly via ``python app.py`` during local development.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "streamdeck_tui"

import asyncio
import colorsys
import hashlib
import json
import math
import os
import shutil
import threading
from asyncio.subprocess import Process
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, List, Optional, Sequence

try:
    from textual import events, on
    from textual.actions import SkipAction
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.timer import Timer
    from textual.worker import Worker
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Static,
        TabPane,
        TabbedContent,
    )
    from textual.widgets._tabbed_content import ContentTabs
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "The 'textual' package is required to run streamdeck_tui. "
        "Install dependencies with 'pip install -e .[dev]' or 'pip install streamdeck-tui'."
    ) from exc

from rich.markup import escape

from .config import AppConfig, FavoriteChannel, ProviderConfig, CONFIG_PATH, save_config
from .logging_utils import configure_logging, get_logger
from .playlist import Channel, build_search_index, filter_channels, load_playlist
from .player import (
    PREFERRED_PLAYER_DEFAULT,
    PlayerHandle,
    launch_player,
    probe_player,
)
from .providers import ConnectionStatus, fetch_connection_status
from .log_viewer import LogViewer
from .stats import StreamStats, StreamStatsAccumulator
from .themes import CUSTOM_THEMES, DEFAULT_THEME_NAME


log = get_logger(__name__)


RECENT_RELOAD_THRESHOLD = timedelta(hours=6)

# Provider colours are generated dynamically to avoid collisions with other
# semantic colours used throughout the UI (for example, resolution tags). The
# hue is derived from a hash of the provider name so that each provider is
# always rendered with a unique, stable colour.
PROVIDER_COLOR_SATURATION = 0.55
PROVIDER_COLOR_LIGHTNESS = 0.5

RESOLUTION_BUCKETS: tuple[tuple[int, str, str], ...] = (
    (2160, "2160p", "bright_magenta"),
    (1440, "1440p", "magenta"),
    (1080, "1080p", "green"),
    (720, "720p", "yellow"),
    (576, "576p", "bright_yellow"),
    (480, "480p", "orange1"),
    (360, "360p", "red"),
)


def _format_timedelta(delta: timedelta) -> str:
    """Return a human-friendly description of ``delta``."""

    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "less than a minute"
    minutes, _ = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        parts.append("less than a minute")
    return " ".join(parts)


def _format_bytes(size: int) -> str:
    """Return ``size`` formatted as a human-readable string."""

    if size <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"  # pragma: no cover - extremely large files


def _format_bitrate(bitrate: Optional[float]) -> str:
    """Render a bitrate value in Mbps or Kbps."""

    if bitrate is None or bitrate <= 0:
        return "–"
    if bitrate >= 1_000_000:
        return f"{bitrate / 1_000_000:.2f} Mbps"
    if bitrate >= 1_000:
        return f"{bitrate / 1_000:.1f} Kbps"
    return f"{bitrate:.0f} bps"


def resolution_tag_for_height(height: Optional[int]) -> Optional[tuple[str, str]]:
    """Return a resolution label and colour for the given height."""

    if height is None or height <= 0:
        return None
    for threshold, label, color in RESOLUTION_BUCKETS:
        if height >= threshold:
            return label, color
    return f"{height}p", "white"


@dataclass
class ProviderState:
    """Runtime state tracked for each provider."""

    config: ProviderConfig
    channels: Optional[List[Channel]] = None
    connection_status: Optional[ConnectionStatus] = None
    connection_usage_percent: Optional[float] = None
    last_error: Optional[str] = None
    loading: bool = False
    last_loaded_at: Optional[datetime] = None
    search_index: Optional[dict[str, set[int]]] = None
    loading_progress: float = 0.0
    loading_bytes_read: int = 0
    loading_bytes_total: Optional[int] = None
    last_channel_count: Optional[int] = None


class ChannelListItem(ListItem):
    """Render an IPTV channel in the list."""

    def __init__(
        self,
        provider_name: str,
        channel: Channel,
        *,
        queued: bool = False,
        provider_markup: Optional[str] = None,
    ) -> None:
        self.channel = channel
        self.provider_name = provider_name
        self._provider_markup = provider_markup
        self._queued = queued
        # Enable markup so provider colours can be applied while ensuring
        # channel names are escaped to avoid injection of markup sequences.
        self._label = Label("", id="channel-name", markup=True)
        super().__init__(self._label)
        self._refresh_label()

    def set_queued(self, queued: bool) -> None:
        """Update the queued indicator for this item."""

        if self._queued == queued:
            return
        self._queued = queued
        self._refresh_label()

    def _refresh_label(self) -> None:
        prefix = "⏳ " if self._queued else ""
        provider = (
            self._provider_markup
            if self._provider_markup is not None
            else escape(self.provider_name)
        )
        name = escape(self.channel.name)
        self._label.update(f"{prefix}{provider} • {name}")


class SearchInput(Input):
    """Channel search field that hands arrow navigation to the results list."""

    def on_key(self, event: events.Key) -> None:  # pragma: no cover - UI callback
        if event.key in ("down", "up"):
            app = getattr(self, "app", None)
            if isinstance(app, StreamdeckApp):
                event.stop()
                app.call_after_refresh(
                    app._focus_channel_list, select_last=event.key == "up"
                )
                return
        handler = getattr(super(), "on_key", None)
        if callable(handler):
            handler(event)
        else:
            self._forward_event(event)


class ChannelListView(ListView):
    """List view that notifies the app when it is mounted."""

    def on_mount(self) -> None:  # pragma: no cover - relies on Textual runtime
        app = self.app
        if isinstance(app, StreamdeckApp):
            app._mark_channel_list_ready()


class FavoriteListItem(ListItem):
    """Render a favorited channel in the list."""

    def __init__(self, favorite: FavoriteChannel) -> None:
        label = f"{favorite.channel_name} ({favorite.provider})"
        super().__init__(Label(label, id="channel-name", markup=False))
        self.favorite = favorite


class _ChannelListSummary(Static):
    """Render a titled list of channel summaries."""

    entries: reactive[tuple[str, ...]] = reactive(tuple())

    def __init__(self, *, title: str, empty_message: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._empty_message = empty_message

    def on_mount(self) -> None:
        self._refresh()

    def watch_entries(self, _: tuple[str, ...]) -> None:
        self._refresh()

    def update_entries(self, entries: Sequence[str]) -> None:
        self.entries = tuple(entries)

    def _refresh(self) -> None:
        body = "\n\n".join(self.entries) if self.entries else self._empty_message
        self.update(f"[b]{self._title}[/]\n{body}")


class SelectedChannelsPanel(_ChannelListSummary):
    """Display all channels that have been queued for playback."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            title="Selected channels",
            empty_message="No channels have been queued for playback.",
            **kwargs,
        )


class PlayingChannelsPanel(_ChannelListSummary):
    """Display information about all actively playing channels."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            title="Now playing",
            empty_message="No streams are currently playing.",
            **kwargs,
        )


class _ChannelSummary(Static):
    """Render a compact summary for a channel selection."""

    channel: reactive[Optional[Channel]] = reactive(None)
    provider: reactive[Optional[str]] = reactive(None)

    def __init__(
        self,
        *,
        title: str,
        empty_message: str,
        show_provider: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._empty_message = empty_message
        self._show_provider = show_provider

    def on_mount(self) -> None:
        self._refresh_summary()

    def watch_channel(self, _: Optional[Channel]) -> None:
        self._refresh_summary()

    def watch_provider(self, _: Optional[str]) -> None:
        if self._show_provider:
            self._refresh_summary()

    def _refresh_summary(self) -> None:
        lines = [f"[b]{self._title}[/b]"]
        channel = self.channel
        if channel is None:
            lines.append(self._empty_message)
        else:
            lines.append(escape(channel.name))
            if self._show_provider and self.provider:
                lines.append(f"Provider: {escape(self.provider)}")
            if channel.group:
                lines.append(f"Group: {escape(channel.group)}")
        self.update("\n".join(lines))


class ChannelInfo(_ChannelSummary):
    """Display information about the currently selected channel."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            title="Selected channel",
            empty_message="Select a channel to view details.",
            show_provider=False,
            **kwargs,
        )


class PlayingChannelInfo(_ChannelSummary):
    """Legacy single-channel playback summary widget."""

    stats: reactive[Optional[StreamStats]] = reactive(None)
    provider_color: reactive[Optional[str]] = reactive(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(
            title="Now playing",
            empty_message="No stream is currently playing.",
            show_provider=True,
            **kwargs,
        )

    def watch_stats(self, _: Optional[StreamStats]) -> None:
        self._refresh_summary()

    def watch_provider_color(self, _: Optional[str]) -> None:
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        lines = [f"[b]{self._title}[/b]"]
        channel = self.channel
        provider = self.provider
        if channel is None:
            lines.append(self._empty_message)
            self.update("\n".join(lines))
            return
        if provider:
            color = self.provider_color or "white"
            lines.append(f"[{color}]{escape(provider)}[/]")
        lines.append(escape(channel.name))
        if channel.group:
            lines.append(f"Group: {escape(channel.group)}")
        stats = self.stats
        if stats is not None:
            lines.extend(self._format_stats(stats))
        self.update("\n".join(lines))

    def _format_stats(self, stats: StreamStats) -> list[str]:
        lines: list[str] = []
        resolution_line = self._format_resolution(stats)
        if resolution_line:
            lines.append(resolution_line)
        bitrate_line = self._format_bitrate(stats)
        if bitrate_line:
            lines.append(bitrate_line)
        return lines

    def _format_resolution(self, stats: StreamStats) -> Optional[str]:
        if stats.height is None and stats.width is None:
            return None
        tag = resolution_tag_for_height(stats.height)
        if tag is None:
            return None
        label, color = tag
        if stats.width and stats.height:
            dims = f"{stats.width}×{stats.height}"
        elif stats.height:
            dims = f"{stats.height}p"
        else:
            dims = "Unknown"
        return f"Resolution: [{color}]{label}[/] ({dims})"

    def _format_bitrate(self, stats: StreamStats) -> Optional[str]:
        if stats.live_bitrate is None and stats.average_bitrate is None:
            return None
        live = _format_bitrate(stats.live_bitrate)
        average = _format_bitrate(stats.average_bitrate)
        return f"Bitrate: Live {live} • Avg {average}"


@dataclass
class NowPlayingEntry:
    """Snapshot of an active playback session for modal display."""

    key: tuple[str, str]
    provider: str
    channel: Channel
    stats: Optional[StreamStats] = None
    started_at: Optional[datetime] = None

    @property
    def bitrate_score(self) -> float:
        if self.stats and self.stats.average_bitrate:
            return self.stats.average_bitrate
        return 0.0


class NowPlayingListItem(ListItem):
    """Render a playing channel inside the management modal."""

    def __init__(
        self, app: "StreamdeckApp", entry: NowPlayingEntry, *, selected: bool = False
    ) -> None:
        self._app = app
        self.entry = entry
        self._selected = selected
        self._label = Label("", markup=True)
        super().__init__(self._label)
        self._refresh()

    def set_selected(self, selected: bool) -> None:
        if self._selected == selected:
            return
        self._selected = selected
        self._refresh()

    def _refresh(self) -> None:
        provider_markup = self._app._provider_markup(self.entry.provider)
        channel_name = escape(self.entry.channel.name)
        indicator = "☑" if self._selected else "☐"
        lines = [f"{indicator} {provider_markup} • {channel_name}"]
        if self.entry.channel.group:
            lines.append(f"Group: {escape(self.entry.channel.group)}")
        elapsed = self._app._format_playback_elapsed(self.entry.started_at)
        if elapsed:
            lines.append(f"Elapsed: {elapsed}")
        stats = self.entry.stats
        if stats is not None:
            lines.extend(self._app._format_stream_stats(stats))
        self._label.update("\n".join(lines))


class NowPlayingModal(ModalScreen[None]):
    """Modal screen used to manage currently playing channels."""

    def __init__(self, app: "StreamdeckApp") -> None:
        super().__init__()
        self._app = app
        self._entries: list[NowPlayingEntry] = []
        self._sort_by_bitrate: bool = True
        self._selected_keys: set[tuple[str, str]] = set()

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Now playing", id="now-playing-title"),
            ListView(id="now-playing-list"),
            Static(
                "Sort by bitrate to close the lowest quality streams manually.",
                id="now-playing-help",
            ),
            Horizontal(
                Button(
                    "Refresh bitrate order",
                    id="now-playing-sort",
                ),
                Button(
                    "Stop selected",
                    id="now-playing-stop",
                    variant="warning",
                ),
                Button(
                    "Stop all",
                    id="now-playing-stop-all",
                    variant="warning",
                ),
                Button("Close", id="now-playing-close", variant="primary"),
                id="now-playing-actions",
            ),
        )

    def on_mount(self) -> None:  # pragma: no cover - UI callback
        self._refresh_list()
        self._refresh_help()

    def on_unmount(self) -> None:  # pragma: no cover - UI callback
        self._app._on_now_playing_modal_closed(self)

    def set_entries(self, entries: Sequence[NowPlayingEntry]) -> None:
        self._entries = list(entries)
        active_keys = {entry.key for entry in self._entries}
        self._selected_keys &= active_keys
        self._apply_sort()
        self._refresh_list()
        self._refresh_help()

    def _apply_sort(self) -> None:
        if not self._sort_by_bitrate:
            return
        self._entries.sort(key=lambda entry: entry.bitrate_score, reverse=True)

    def _get_list_view(self) -> Optional[ListView]:
        try:
            return self.query_one("#now-playing-list", ListView)
        except Exception:
            return None

    def _refresh_list(self) -> None:
        list_view = self._get_list_view()
        if list_view is None:
            return
        current_key: Optional[tuple[str, str]] = None
        index = getattr(list_view, "index", None)
        if index is not None and 0 <= index < len(self._entries):
            current_key = self._entries[index].key
        list_view.clear()
        if not self._entries:
            list_view.index = None
            self._update_controls()
            return
        for entry in self._entries:
            list_view.append(
                NowPlayingListItem(
                    self._app,
                    entry,
                    selected=entry.key in self._selected_keys,
                )
            )
        if current_key is not None:
            for idx, entry in enumerate(self._entries):
                if entry.key == current_key:
                    list_view.index = idx
                    break
        if getattr(list_view, "index", None) is None:
            list_view.index = 0
        self._update_controls()

    def _refresh_help(self) -> None:
        try:
            help_text = self.query_one("#now-playing-help", Static)
        except Exception:
            return
        if self._entries:
            help_text.update(
                "Sort by bitrate to close the lowest quality streams manually. "
                "Press space to toggle channel selection."
            )
        else:
            help_text.update("No streams are currently playing.")

    def _update_controls(self) -> None:
        list_view = self._get_list_view()
        selected_index: Optional[int] = None
        if list_view is not None:
            selected_index = getattr(list_view, "index", None)
        has_entries = bool(self._entries)
        has_selection = bool(self._selected_keys)
        selected_valid = (
            selected_index is not None
            and 0 <= selected_index < len(self._entries)
        )
        for selector, enabled in (
            ("#now-playing-sort", has_entries),
            ("#now-playing-stop", has_entries and (selected_valid or has_selection)),
            ("#now-playing-stop-all", has_entries),
        ):
            try:
                button = self.query_one(selector, Button)
            except Exception:
                continue
            button.disabled = not enabled

    def _selected_entry(self) -> Optional[NowPlayingEntry]:
        list_view = self._get_list_view()
        if list_view is None:
            return None
        index = getattr(list_view, "index", None)
        if index is None or not (0 <= index < len(self._entries)):
            return None
        return self._entries[index]

    @on(ListView.Highlighted, "#now-playing-list")
    def _on_highlighted(self, _: ListView.Highlighted) -> None:  # pragma: no cover
        self._update_controls()

    @on(Button.Pressed, "#now-playing-sort")
    def _on_sort_pressed(self, _: Button.Pressed) -> None:  # pragma: no cover
        self._sort_by_bitrate = True
        self._apply_sort()
        self._refresh_list()

    @on(Button.Pressed, "#now-playing-stop")
    def _on_stop_selected(self, _: Button.Pressed) -> None:  # pragma: no cover
        keys: list[tuple[str, str]]
        if self._selected_keys:
            keys = list(self._selected_keys)
        else:
            entry = self._selected_entry()
            if entry is None:
                return
            keys = [entry.key]
        self._app._stop_channels(keys)
        for key in keys:
            self._selected_keys.discard(key)
        self.set_entries(self._app._collect_now_playing_entries())

    @on(Button.Pressed, "#now-playing-stop-all")
    def _on_stop_all(self, _: Button.Pressed) -> None:  # pragma: no cover
        self._app.action_stop_all_playback()
        self._selected_keys.clear()
        self.set_entries(self._app._collect_now_playing_entries())

    @on(Button.Pressed, "#now-playing-close")
    def _on_close(self, _: Button.Pressed) -> None:  # pragma: no cover
        self.dismiss(None)

    def on_key(self, event: events.Key) -> None:  # pragma: no cover - UI callback
        if event.key == "space":
            entry = self._selected_entry()
            if entry is not None:
                if entry.key in self._selected_keys:
                    self._selected_keys.remove(entry.key)
                else:
                    self._selected_keys.add(entry.key)
                self._refresh_list()
                self._update_controls()
            event.stop()
            return
        super().on_key(event)


class StatusBar(Static):
    """A simple status bar widget."""

    status: reactive[str] = reactive("Ready")

    def watch_status(self, status: str) -> None:
        self.update(status)


class ProviderProgress(Static):
    """Visual progress indicator for provider loading."""

    def on_mount(self) -> None:  # pragma: no cover - relies on runtime styles
        self.hide()

    def hide(self) -> None:
        self.styles.display = "none"

    def show_progress(
        self,
        *,
        provider: str,
        progress: float,
        loaded: int,
        total: Optional[int],
        markup: bool = False,
    ) -> None:
        self.styles.display = "block"
        clamped = max(0.0, min(progress, 1.0))
        percent = int(round(clamped * 100))
        bar_width = 28
        filled = int(round(clamped * bar_width))
        filled = max(0, min(bar_width, filled))
        empty = bar_width - filled
        bar = f"{'█' * filled}{'░' * empty}"
        if total and total > 0:
            totals = f"{_format_bytes(loaded)} / {_format_bytes(total)}"
        else:
            totals = f"{_format_bytes(loaded)} received"
        provider_label = provider if markup else escape(provider)
        text = (
            f"[b]{provider_label}[/b] — {percent}% ({totals})\n"
            f"[{bar}]"
        )
        self.update(text)


class ConnectionUsageBar(Static):
    """Visualise the connection usage reported by a provider API."""

    BAR_WIDTH = 28

    def __init__(self, *children: Any, **kwargs: Any) -> None:
        super().__init__(*children, **kwargs)
        self._last_markup: str = ""

    def on_mount(self) -> None:  # pragma: no cover - relies on runtime styles
        self.show_unavailable()

    @property
    def last_markup(self) -> str:
        return self._last_markup

    @staticmethod
    def _clamp_percent(percent: float) -> float:
        return max(0.0, min(percent, 1.0))

    @staticmethod
    def _color_for_percent(percent: float) -> str:
        if percent >= 0.9:
            return "red"
        if percent >= 0.75:
            return "yellow"
        return "green"

    def show_unavailable(self, message: str = "Connection status unavailable") -> None:
        markup = f"[dim]{escape(message)}[/dim]"
        self._last_markup = markup
        self.update(markup)

    def update_status(
        self,
        *,
        active_connections: Optional[int],
        max_connections: Optional[int],
        percent: Optional[float],
        message: Optional[str] = None,
    ) -> None:
        if (
            active_connections is None
            or max_connections is None
            or max_connections <= 0
        ):
            fallback = message or "Connection status unavailable"
            self.show_unavailable(fallback)
            return

        usage = (
            self._clamp_percent(percent)
            if percent is not None
            else self._clamp_percent(active_connections / max_connections)
        )
        filled = int(round(usage * self.BAR_WIDTH))
        filled = max(0, min(self.BAR_WIDTH, filled))
        empty = self.BAR_WIDTH - filled
        color = self._color_for_percent(usage)
        bar = f"[{color}]{'█' * filled}{'░' * empty}[/]"
        percent_label = int(round(usage * 100))
        header = (
            f"[b]{active_connections}/{max_connections} connections"
            f" ({percent_label}%)[/b]"
        )
        if message:
            markup = f"{header}\n{bar}\n[dim]{escape(message)}[/dim]"
        else:
            markup = f"{header}\n{bar}"
        self._last_markup = markup
        self.update(markup)


class ProviderForm(Static):
    """Form used to edit provider configuration."""

    def compose(self) -> ComposeResult:
        yield Label("Provider name")
        yield Input(placeholder="Name", id="provider-name")
        yield Label("Playlist URL")
        yield Input(placeholder="Playlist URL", id="provider-playlist")
        yield Label("API URL (optional)")
        yield Input(placeholder="Status API URL (optional)", id="provider-api")
        with Horizontal(id="form-buttons"):
            yield Button("Reset", id="provider-reset")
            yield Button("Save", id="provider-save", variant="success")

    def populate(self, provider: ProviderConfig) -> None:
        self.query_one("#provider-name", Input).value = provider.name
        self.query_one("#provider-playlist", Input).value = provider.playlist_url
        self.query_one("#provider-api", Input).value = provider.api_url or ""

    def read(self) -> ProviderConfig:
        name = self.query_one("#provider-name", Input).value.strip()
        playlist = self.query_one("#provider-playlist", Input).value.strip()
        api = self.query_one("#provider-api", Input).value.strip() or None
        return ProviderConfig(name=name, playlist_url=playlist, api_url=api)

    def clear(self) -> None:
        self.query_one("#provider-name", Input).value = ""
        self.query_one("#provider-playlist", Input).value = ""
        self.query_one("#provider-api", Input).value = ""

    def focus_name(self) -> None:
        self.query_one("#provider-name", Input).focus()


class ReloadConfirmation(ModalScreen[bool]):
    """Modal dialog asking the user to confirm a reload."""

    def __init__(self, provider_name: str, elapsed: timedelta) -> None:
        super().__init__()
        self._provider_name = provider_name
        self._elapsed = elapsed

    def compose(self) -> ComposeResult:
        message = (
            f"{self._provider_name} was last loaded {_format_timedelta(self._elapsed)} ago."
            "\nReloading again may not be necessary. Proceed?"
        )
        with Vertical(id="reload-confirmation"):
            yield Label(message, id="reload-confirmation-message")
            with Horizontal(id="reload-confirmation-buttons"):
                yield Button("Cancel", id="reload-cancel", variant="warning")
                yield Button("Reload", id="reload-confirm", variant="success")

    @on(Button.Pressed, "#reload-confirm")
    def _on_confirm(self, _: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reload-cancel")
    def _on_cancel(self, _: Button.Pressed) -> None:
        self.dismiss(False)


# We keep a very small inline stylesheet so the application can always boot
# without depending on external CSS files.
_INLINE_DEFAULT_CSS = """
#main-tabs {
    height: 1fr;
}

TabPane {
    padding: 0;
}

#providers-pane,
#channels-pane,
#favorites-pane,
#logs-pane {
    layout: vertical;
    height: 1fr;
    padding: 1;
}

#providers-pane {
    min-width: 30;
}

#provider-actions {
    layout: horizontal;
    padding-top: 1;
}

#provider-progress {
    border: heavy $surface;
    padding: 0 1;
    min-height: 3;
}

#connection-usage {
    border: heavy $surface;
    padding: 0 1;
    min-height: 3;
}

#channel-browser {
    layout: horizontal;
    height: 1fr;
}

#channel-list,
#favorites-list,
#log-viewer {
    height: 1fr;
}

#channel-list {
    width: 2fr;
    min-width: 42;
}

#channel-sidebar {
    layout: vertical;
    width: 3fr;
    min-width: 36;
}

#channel-info,
#selected-channels,
#playing-channels,
#favorites-help {
    padding: 1;
}

#channel-info,
#selected-channels,
#playing-channels {
    border: heavy $surface;
    width: 1fr;
    min-height: 8;
    height: 1fr;
    overflow-y: auto;
}

#channel-info,
#selected-channels {
    margin-bottom: 1;
}

#channel-actions {
    layout: horizontal;
}

#playing-actions {
    layout: horizontal;
}

#playing-actions Button {
    width: 1fr;
}

#now-playing-actions {
    layout: horizontal;
    margin-top: 1;
}

#now-playing-actions Button {
    width: 1fr;
}

#log-viewer {
    border: heavy $surface;
    padding: 0 1;
    overflow-y: auto;
}

StatusBar {
    padding: 0 1;
}
"""


DEFAULT_CSS = _INLINE_DEFAULT_CSS


class StreamdeckApp(App[None]):
    """Main Textual application."""

    CSS = DEFAULT_CSS
    CSS_PATH: list[str] = []
    CHANNEL_RENDER_BATCH_SIZE = 200
    CHANNEL_RENDER_BATCH_DELAY = 0.001
    CHANNEL_WINDOW_SIZE = 200
    CHANNEL_WINDOW_MARGIN = 40
    provider_form_visible: reactive[bool] = reactive(False)
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
        Binding("f1", "switch_tab('providers')", "Providers"),
        Binding("f2", "switch_tab('channels')", "Channels"),
        Binding("f3", "switch_tab('favorites')", "Favorites"),
        Binding("f4", "switch_tab('logs')", "Logs"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear search"),
        Binding("n", "new_provider", "New provider"),
        Binding("ctrl+s", "save_provider", "Save provider"),
        Binding("delete", "delete_provider", "Delete provider"),
        Binding("r", "reload_provider", "Reload provider"),
        Binding("p", "play_channel", "Play"),
        Binding("space", "toggle_channel_queue", "Queue channel"),
        Binding("s", "stop_channel", "Stop selected"),
        Binding("ctrl+shift+s", "stop_all_playback", "Stop all"),
        Binding("f", "toggle_favorite", "Favorite"),
        Binding("ctrl+shift+p", "probe_player", "Probe player"),
        Binding("m", "show_now_playing", "Manage playback"),
        Binding("down", "focus_active_tab_content", show=False, priority=True),
    ]
    _PROVIDERS_TAB_REQUIRED_STATUS = "Switch to the Providers tab to manage providers"

    def _register_custom_themes(self) -> None:
        for theme in CUSTOM_THEMES.values():
            self.register_theme(theme)

    def _apply_requested_theme(self, requested: Optional[str]) -> None:
        preferred = requested or DEFAULT_THEME_NAME
        theme = self.get_theme(preferred)
        if theme is None:
            if requested:
                log.warning(
                    "Requested theme '%s' is unavailable; falling back to %s",
                    requested,
                    DEFAULT_THEME_NAME,
                )
            fallback = self.get_theme(DEFAULT_THEME_NAME)
            if fallback is not None:
                self.theme = fallback.name
            return
        log.debug("Applying theme %s", theme.name)
        self.theme = theme.name

    def __init__(
        self,
        config: AppConfig,
        *,
        config_path: Optional[Path] = None,
        preferred_player: Optional[str] = None,
        theme: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._register_custom_themes()
        self._apply_requested_theme(theme)
        self._config = config
        self._config_path = config_path or CONFIG_PATH
        self._states: list[ProviderState] = [ProviderState(provider) for provider in config.providers]
        self._active_index: Optional[int] = 0 if self._states else None
        self._editing_name: Optional[str] = (
            self._states[0].config.name if self._states else None
        )
        self.filtered_channels: list[tuple[str, Channel]] = []
        self._all_channels: list[tuple[str, Channel]] = []
        self._all_channel_objects: list[Channel] = []
        self._all_channels_search_index: Optional[dict[str, set[int]]] = None
        self._favorite_entries: list[FavoriteChannel] = list(config.favorites)
        self._active_tab: str = "channels"
        self._queued_channels: set[tuple[str, str]] = set()
        self._queued_order: list[tuple[str, str]] = []
        self._player_handles: dict[tuple[str, str], PlayerHandle] = {}
        self._player_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._player_monitor_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._player_stop_requested: set[tuple[str, str]] = set()
        self._playing_channels: dict[tuple[str, str], tuple[str, Channel]] = {}
        self._stream_stats: dict[
            tuple[str, str], StreamStatsAccumulator
        ] = {}
        self._latest_stats: dict[tuple[str, str], StreamStats] = {}
        self._playback_started_at: dict[tuple[str, str], datetime] = {}
        self._now_playing_modal: Optional[NowPlayingModal] = None
        self._player_process: Optional[Process] = None
        self._worker: Optional[Worker] = None
        self._provider_load_queue: deque[tuple[int, bool]] = deque()
        self._playback_timer: Optional[Timer] = None
        self._playback_timer_active: bool = False
        self._current_provider_index: Optional[int] = None
        self._app_thread_id: int = threading.get_ident()
        self._log_viewer: Optional[LogViewer] = None
        self._status_bar: Optional[StatusBar] = None
        self._channel_render_task: Optional[asyncio.Task[None]] = None
        self._channel_render_generation: int = 0
        self._channel_rendered_count: int = 0
        self._channel_first_batch_size: int = 0
        self._channel_list_ready: bool = False
        self._pending_channel_operations: list[Callable[[], None]] = []
        self._channel_window_start: int = 0
        self._probing_player: bool = False
        self._provider_colors: dict[str, str] = {}
        self._preferred_player: Optional[str] = (
            preferred_player if preferred_player is not None else PREFERRED_PLAYER_DEFAULT
        )
        log.debug("Inline stylesheet active (%d characters)", len(self.CSS))
        log.info(
            "StreamdeckApp initialized with %d provider(s); config path=%s",
            len(self._states),
            self._config_path,
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="main-tabs"):
            with TabPane("Providers", id="providers-tab"):
                with Vertical(id="providers-pane"):
                    yield Label("Providers", id="providers-title")
                    yield ListView(id="provider-list")
                    yield ConnectionUsageBar(id="connection-usage")
                    yield ProviderProgress(id="provider-progress")
                    with Horizontal(id="provider-actions"):
                        yield Button("New", id="provider-new", variant="primary")
                        yield Button("Delete", id="provider-delete", variant="warning")
                    yield Static(
                        f"Add or remove providers below. Changes are saved to config.yaml ({self._config_destination_label()}).",
                        id="providers-help",
                    )
                    with Vertical(id="provider-form-container"):
                        yield ProviderForm(id="provider-form")
            with TabPane("Channel browser", id="channels-tab"):
                with Vertical(id="channels-pane"):
                    yield SearchInput(placeholder="Search channels…", id="search")
                    with Horizontal(id="channel-browser"):
                        yield ChannelListView(id="channel-list")
                        with Vertical(id="channel-sidebar"):
                            yield ChannelInfo(id="channel-info")
                            yield SelectedChannelsPanel(id="selected-channels")
                            with Horizontal(id="channel-actions"):
                                yield Button(
                                    "Play",
                                    id="channel-play",
                                    variant="success",
                                )
                                yield Button(
                                    "Stop selected",
                                    id="channel-stop",
                                    variant="warning",
                                    disabled=True,
                                )
                                yield Button(
                                    "Favorite",
                                    id="channel-favorite",
                                )
                                yield Button(
                                    "Probe player",
                                    id="channel-probe",
                                    variant="default",
                                )
                            yield PlayingChannelsPanel(id="playing-channels")
                            with Horizontal(id="playing-actions"):
                                yield Button(
                                    "Manage playback",
                                    id="playing-manage",
                                    disabled=True,
                                )
                                yield Button(
                                    "Stop all",
                                    id="playing-stop",
                                    variant="warning",
                                    disabled=True,
                                )
            with TabPane("Favorites", id="favorites-tab"):
                with Vertical(id="favorites-pane"):
                    yield ListView(id="favorites-list")
                    yield Static(
                        "Mark channels as favorites to see them listed here.",
                        id="favorites-help",
                    )
            with TabPane("Logs", id="logs-tab"):
                with Vertical(id="logs-pane"):
                    yield LogViewer(id="log-viewer")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        log.debug("Application mounted")
        self._app_thread_id = threading.get_ident()
        try:
            self._log_viewer = self.query_one(LogViewer)
        except Exception:  # pragma: no cover - log viewer may be missing in tests
            self._log_viewer = None
        if self._log_viewer is not None:
            configure_logging(log_viewer=self._log_viewer, app=self)
        try:
            self._status_bar = self.query_one(StatusBar)
        except Exception:
            self._status_bar = None
        self._refresh_provider_list()
        if self._states:
            self._select_provider(self._active_index or 0)
            self.query_one("#provider-list", ListView).focus()
        else:
            self.query_one(StatusBar).status = "Add a provider to get started"
            self._show_provider_form()
            self.query_one(ProviderForm).focus_name()
            log.info("No providers configured; prompting user to add one")
        self._update_tab_buttons()
        self._refresh_favorites_view()
        self._update_provider_form_visibility(self.provider_form_visible)
        self._refresh_selected_channels_panel()
        self._refresh_playing_channels_panel()
        self._ensure_playback_timer()
        self._update_playback_timer_state()
        self._refresh_stale_providers()
        self._update_provider_form_visibility(self.provider_form_visible)

    def _focus_provider_list(self) -> None:
        provider_list = self._query_optional_widget("#provider-list", ListView)
        if provider_list is not None:
            provider_list.focus()

    def _focus_channel_list(self, *, select_last: bool = False) -> None:
        list_view = self._query_optional_widget("#channel-list", ListView)
        if list_view is None:
            return
        if list_view.children and getattr(list_view, "index", None) is None:
            target = len(list_view.children) - 1 if select_last else 0
            list_view.index = max(0, target)
        list_view.focus()

    def _update_provider_form_visibility(self, visible: bool) -> None:
        container = self._query_optional_widget("#provider-form-container", Vertical)
        if container is None:
            return
        container.display = visible
        if not visible:
            self._focus_provider_list()

    def watch_provider_form_visible(self, visible: bool) -> None:
        self._update_provider_form_visibility(visible)

    def _show_provider_form(self) -> None:
        if self.provider_form_visible:
            self._update_provider_form_visibility(True)
        else:
            self.provider_form_visible = True

    def _hide_provider_form(self) -> None:
        if not self.provider_form_visible:
            self._update_provider_form_visibility(False)
            return
        self.provider_form_visible = False

    def _mark_channel_list_ready(self) -> None:
        if self._channel_list_ready:
            return
        self._channel_list_ready = True
        pending = self._pending_channel_operations
        self._pending_channel_operations = []
        for callback in pending:
            try:
                callback()
            except Exception:  # pragma: no cover - diagnostic logging only
                log.exception("Deferred channel operation failed")

    def _query_optional_widget(
        self, query: object, widget_type: Optional[type[Any]] = None
    ) -> Optional[Any]:
        """Return the first matching widget if it exists."""

        try:
            if widget_type is None:
                return self.query_one(query)  # type: ignore[arg-type]
            return self.query_one(query, widget_type)  # type: ignore[arg-type]
        except Exception:
            return None

    def _current_state(self) -> Optional[ProviderState]:
        if self._active_index is None:
            return None
        if 0 <= self._active_index < len(self._states):
            return self._states[self._active_index]
        return None

    def _provider_state_by_name(self, provider_name: str) -> Optional[ProviderState]:
        for state in self._states:
            if state.config.name == provider_name:
                return state
        return None

    def _channel_queue_key(self, provider_name: str, channel: Channel) -> tuple[str, str]:
        return (provider_name, channel.url)

    def _is_channel_queued(
        self, provider_name: Optional[str], channel: Channel
    ) -> bool:
        if provider_name is None:
            return False
        return (provider_name, channel.url) in self._queued_channels

    def _queue_add(self, key: tuple[str, str]) -> None:
        if key in self._queued_channels:
            return
        self._queued_channels.add(key)
        self._queued_order.append(key)

    def _queue_remove(self, key: tuple[str, str]) -> None:
        if key not in self._queued_channels:
            return
        self._queued_channels.remove(key)
        try:
            self._queued_order.remove(key)
        except ValueError:
            pass

    def _queue_remove_many(self, keys: Sequence[tuple[str, str]]) -> None:
        removal_list = [key for key in keys if key in self._queued_channels]
        if not removal_list:
            return
        removal_set = set(removal_list)
        self._queued_channels.difference_update(removal_set)
        self._queued_order = [
            key for key in self._queued_order if key not in removal_set
        ]

    def _update_channel_item_queue_state(
        self, provider_name: str, channel: Channel, queued: bool
    ) -> None:
        list_view = self._query_optional_widget("#channel-list", ListView)
        if list_view is None:
            return
        for item in list_view.children:
            if isinstance(item, ChannelListItem) and item.channel.url == channel.url:
                item.set_queued(queued)

    def _refresh_channel_queue_indicators(self) -> None:
        list_view = self._query_optional_widget("#channel-list", ListView)
        if list_view is None:
            return
        state = self._current_state()
        provider_name = state.config.name if state else None
        for item in list_view.children:
            if isinstance(item, ChannelListItem):
                item.set_queued(self._is_channel_queued(provider_name, item.channel))

    def _refresh_selected_channels_panel(self) -> None:
        panel = self._query_optional_widget(SelectedChannelsPanel)
        if panel is None:
            return
        entries: list[str] = []
        missing: list[tuple[str, str]] = []
        for key in self._queued_order:
            if key not in self._queued_channels:
                continue
            provider_name, channel_identifier = key
            channel = self._find_channel_by_identifier(provider_name, channel_identifier)
            if channel is None:
                missing.append(key)
                continue
            provider_markup = self._provider_markup(provider_name)
            lines = [f"{provider_markup} • {escape(channel.name)}"]
            if channel.group:
                lines.append(f"Group: {escape(channel.group)}")
            if key in self._playing_channels:
                elapsed = self._format_playback_elapsed(self._playback_started_at.get(key))
                status = "[green]Playing[/]"
                if elapsed:
                    status = f"{status} ({elapsed})"
                lines.append(status)
            entries.append("\n".join(lines))
        if missing:
            self._queue_remove_many(missing)
            self._refresh_channel_queue_indicators()
            self._refresh_selected_channels_panel()
            return
        panel.update_entries(entries)

    def _format_stream_resolution(self, stats: StreamStats) -> Optional[str]:
        if stats.height is None and stats.width is None:
            return None
        tag = resolution_tag_for_height(stats.height)
        if tag is None:
            return None
        label, color = tag
        if stats.width and stats.height:
            dims = f"{stats.width}×{stats.height}"
        elif stats.height:
            dims = f"{stats.height}p"
        else:
            dims = "Unknown"
        return f"Resolution: [{color}]{label}[/] ({dims})"

    def _format_stream_bitrate(self, stats: StreamStats) -> Optional[str]:
        if stats.live_bitrate is None and stats.average_bitrate is None:
            return None
        live = _format_bitrate(stats.live_bitrate)
        average = _format_bitrate(stats.average_bitrate)
        return f"Bitrate: Live {live} • Avg {average}"

    def _format_stream_stats(self, stats: StreamStats) -> list[str]:
        lines: list[str] = []
        resolution_line = self._format_stream_resolution(stats)
        if resolution_line:
            lines.append(resolution_line)
        bitrate_line = self._format_stream_bitrate(stats)
        if bitrate_line:
            lines.append(bitrate_line)
        return lines

    def _format_playback_elapsed(
        self, started_at: Optional[datetime]
    ) -> Optional[str]:
        if started_at is None:
            return None
        now = datetime.now(tz=timezone.utc)
        if started_at > now:
            return None
        elapsed = now - started_at
        total_seconds = int(elapsed.total_seconds())
        if total_seconds < 0:
            total_seconds = 0
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _refresh_playing_channels_panel(self) -> None:
        panel = self._query_optional_widget(PlayingChannelsPanel)
        if panel is None:
            return
        entries: list[str] = []
        for key, (provider_name, channel) in self._playing_channels.items():
            provider_markup = self._provider_markup(provider_name)
            lines = [f"{provider_markup} • {escape(channel.name)}"]
            if channel.group:
                lines.append(f"Group: {escape(channel.group)}")
            elapsed = self._format_playback_elapsed(self._playback_started_at.get(key))
            if elapsed:
                lines.append(f"Elapsed: {elapsed}")
            stats = self._latest_stats.get(key)
            if stats is not None:
                lines.extend(self._format_stream_stats(stats))
            entries.append("\n".join(lines))
        panel.update_entries(entries)
        self._set_stop_buttons_enabled(bool(self._playing_channels))
        self._update_now_playing_modal_entries()
        self._refresh_selected_channels_panel()

    def _ensure_playback_timer(self) -> None:
        if self._playback_timer is None:
            self._playback_timer = self.set_interval(
                1.0, self._on_playback_timer_tick, pause=True
            )
            self._playback_timer_active = False

    def _on_playback_timer_tick(self) -> None:
        if self._playing_channels:
            self._refresh_playing_channels_panel()

    def _update_playback_timer_state(self) -> None:
        self._ensure_playback_timer()
        timer = self._playback_timer
        if timer is None:
            return
        if self._playing_channels and not self._playback_timer_active:
            timer.resume()
            self._playback_timer_active = True
        elif not self._playing_channels and self._playback_timer_active:
            timer.pause()
            self._playback_timer_active = False

    def _stats_accumulator_for_key(
        self, key: tuple[str, str]
    ) -> StreamStatsAccumulator:
        accumulator = self._stream_stats.get(key)
        if accumulator is None:
            accumulator = StreamStatsAccumulator()
            self._stream_stats[key] = accumulator
        return accumulator

    def _clear_playing_stats(self, key: Optional[tuple[str, str]] = None) -> None:
        if key is None:
            self._stream_stats.clear()
            self._latest_stats.clear()
            return
        self._stream_stats.pop(key, None)
        self._latest_stats.pop(key, None)

    def _remove_playing_entry(self, key: tuple[str, str]) -> None:
        self._playing_channels.pop(key, None)
        self._clear_playing_stats(key)
        self._playback_started_at.pop(key, None)

    def _handle_bitrate_update(self, key: tuple[str, str], value: float) -> None:
        accumulator = self._stats_accumulator_for_key(key)
        stats = accumulator.push_bitrate(value)
        self._latest_stats[key] = stats
        self._refresh_playing_channels_panel()

    def _handle_resolution_update(
        self,
        key: tuple[str, str],
        width: Optional[int],
        height: Optional[int],
    ) -> None:
        accumulator = self._stats_accumulator_for_key(key)
        stats = accumulator.set_resolution(width, height)
        self._latest_stats[key] = stats
        self._refresh_playing_channels_panel()

    def _find_channel_by_identifier(
        self, provider_name: str, channel_identifier: str
    ) -> Optional[Channel]:
        state = self._provider_state_by_name(provider_name)
        if state is None or not state.channels:
            return None
        for channel in state.channels:
            if channel.url == channel_identifier:
                return channel
        return None

    def _set_status(self, message: str) -> None:
        log.debug("Status update: %s", message)
        status_bar = self._status_bar or self._query_optional_widget(StatusBar)
        if status_bar is None:
            log.debug("Dropping status update; status bar unavailable")
            return
        self._status_bar = status_bar
        status_bar.status = message

    def _config_destination_label(self) -> str:
        return str(self._config_path)

    def _update_active_channel_info(
        self, channel: Optional[Channel], provider: Optional[str]
    ) -> None:
        info = self._query_optional_widget(ChannelInfo)
        if info is None:
            return
        info.provider = provider
        info.channel = channel

    def _clear_active_channel_info(self) -> None:
        self._update_active_channel_info(None, None)

    def _update_playing_info(
        self, channel: Optional[Channel], provider: Optional[str]
    ) -> None:
        """Compatibility shim for legacy single-channel playback updates."""

        self._refresh_playing_channels_panel()

    def _clear_playing_info(self) -> None:
        """Compatibility shim to clear the legacy playing panel."""

        self._playing_channels.clear()
        self._clear_playing_stats()
        self._playback_started_at.clear()
        self._refresh_playing_channels_panel()
        self._player_process = None

    def _set_stop_buttons_enabled(self, enabled: bool) -> None:
        for selector in ("#channel-stop", "#playing-stop", "#playing-manage"):
            button = self._query_optional_widget(selector, Button)
            if button is not None:
                button.disabled = not enabled

    def _active_playback_keys(self) -> set[tuple[str, str]]:
        return set(self._player_handles.keys()) | set(self._player_tasks.keys())

    def _stop_channels(self, keys: Iterable[tuple[str, str]]) -> None:
        active_keys = self._active_playback_keys()
        keys_to_stop = [key for key in keys if key in active_keys]
        if not keys_to_stop:
            return
        stopped_names: list[str] = []
        for key in keys_to_stop:
            provider, channel_identifier = key
            channel_obj = self._find_channel_by_identifier(provider, channel_identifier)
            if channel_obj is not None:
                stopped_names.append(channel_obj.name)
            self._stop_player_process(key, user_requested=True)
        if not stopped_names:
            self._set_status("Stopping playback")
            log.info("Stop requested for %d channel(s)", len(keys_to_stop))
        elif len(stopped_names) == 1:
            self._set_status(f"Stopping playback for {stopped_names[0]}")
            log.info("Stop requested for %s", stopped_names[0])
        else:
            self._set_status(
                f"Stopping playback for {len(stopped_names)} channels"
            )
            log.info(
                "Stop requested for channels: %s", ", ".join(stopped_names)
            )

    def _collect_now_playing_entries(self) -> list[NowPlayingEntry]:
        entries: list[NowPlayingEntry] = []
        for key, (provider_name, channel) in self._playing_channels.items():
            entries.append(
                NowPlayingEntry(
                    key=key,
                    provider=provider_name,
                    channel=channel,
                    stats=self._latest_stats.get(key),
                    started_at=self._playback_started_at.get(key),
                )
            )
        return entries

    def _update_now_playing_modal_entries(self) -> None:
        modal = self._now_playing_modal
        if modal is None:
            return
        modal.set_entries(self._collect_now_playing_entries())

    def _on_now_playing_modal_closed(self, modal: NowPlayingModal) -> None:
        if self._now_playing_modal is modal:
            self._now_playing_modal = None

    @staticmethod
    def _generate_provider_color(provider: str) -> str:
        digest = hashlib.sha1(provider.encode("utf8")).digest()
        hue = int.from_bytes(digest[:2], "big") / 65535.0
        red, green, blue = colorsys.hls_to_rgb(
            hue, PROVIDER_COLOR_LIGHTNESS, PROVIDER_COLOR_SATURATION
        )
        return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"

    def _provider_color(self, provider: Optional[str]) -> Optional[str]:
        if provider is None:
            return None
        if provider not in self._provider_colors:
            self._provider_colors[provider] = self._generate_provider_color(provider)
        return self._provider_colors[provider]

    def _provider_markup(self, provider: str) -> str:
        color = self._provider_color(provider) or "white"
        return f"[{color}]{escape(provider)}[/]"

    def _start_player_monitor(
        self, key: tuple[str, str], handle: PlayerHandle
    ) -> None:
        self._cancel_player_monitor(key)
        ipc_path = handle.command.ipc_path
        if not ipc_path:
            return
        if sys.platform == "win32":  # pragma: no cover - platform dependent
            log.info("MPV IPC monitoring is not supported on Windows; skipping stats")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - defensive guard for tests
            log.debug("No running event loop available for player monitor")
            return
        task = loop.create_task(self._monitor_mpv_stats(key, ipc_path))
        self._player_monitor_tasks[key] = task

    def _cancel_player_monitor(self, key: Optional[tuple[str, str]] = None) -> None:
        if key is None:
            for pending_key in list(self._player_monitor_tasks.keys()):
                self._cancel_player_monitor(pending_key)
            return
        task = self._player_monitor_tasks.pop(key, None)
        if task is not None:
            task.cancel()

    def _cleanup_player_resources(
        self, key: Optional[tuple[str, str]] = None
    ) -> None:
        if key is None:
            for pending_key in list(self._player_handles.keys()):
                self._cleanup_player_resources(pending_key)
            return
        self._remove_playing_entry(key)
        handle = self._player_handles.pop(key, None)
        self._update_playback_timer_state()
        if handle is None:
            return
        for path in handle.command.cleanup_paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
            except Exception:  # pragma: no cover - cleanup best-effort
                log.debug("Failed to remove %s", path, exc_info=True)

    async def _connect_to_mpv_ipc(
        self, ipc_path: str, retries: int = 50, delay: float = 0.1
    ) -> Optional[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        for attempt in range(retries):
            try:
                return await asyncio.open_unix_connection(ipc_path)
            except (FileNotFoundError, ConnectionRefusedError):
                await asyncio.sleep(delay)
            except Exception as exc:  # pragma: no cover - diagnostic logging only
                log.debug(
                    "Attempt %s to connect to mpv IPC at %s failed: %s",
                    attempt + 1,
                    ipc_path,
                    exc,
                )
                await asyncio.sleep(delay)
        log.warning("Unable to connect to mpv IPC server at %s", ipc_path)
        return None

    async def _monitor_mpv_stats(
        self, key: tuple[str, str], ipc_path: str
    ) -> None:
        if sys.platform == "win32":  # pragma: no cover - platform dependent
            return
        connection = await self._connect_to_mpv_ipc(ipc_path)
        if connection is None:
            return
        reader, writer = connection
        try:
            commands = [
                {"command": ["observe_property", 1, "video-bitrate"]},
                {"command": ["observe_property", 2, "video-params"]},
            ]
            for command in commands:
                writer.write((json.dumps(command) + "\n").encode("utf-8"))
            await writer.drain()
            while True:
                line = await reader.readline()
                if not line:
                    if reader.at_eof():
                        break
                    await asyncio.sleep(0.1)
                    continue
                try:
                    payload = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                if payload.get("event") != "property-change":
                    continue
                name = payload.get("name")
                data = payload.get("data")
                if name == "video-bitrate" and isinstance(data, (int, float)):
                    self._call_on_app_thread(
                        self._handle_bitrate_update, key, float(data)
                    )
                elif name == "video-params" and isinstance(data, dict):
                    width = data.get("w")
                    height = data.get("h")
                    self._call_on_app_thread(
                        self._handle_resolution_update,
                        key,
                        width,
                        height,
                    )
        except asyncio.CancelledError:  # pragma: no cover - cancellation path
            raise
        except Exception:  # pragma: no cover - monitoring errors are logged
            log.exception("Error while monitoring mpv IPC at %s", ipc_path)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # pragma: no cover - cleanup best-effort
                pass

    def _update_tab_buttons(self) -> None:
        """Refresh tab-adjacent controls (search field enablement)."""

        search_input = self._query_optional_widget("#search", Input)
        if search_input is not None:
            search_input.disabled = self._active_tab != "channels"

    def _favorite_to_channel(self, favorite: FavoriteChannel) -> Channel:
        """Convert a stored favorite entry back into a channel."""

        raw_attributes = {"provider": favorite.provider}
        return Channel(
            name=favorite.channel_name,
            url=favorite.channel_url,
            group=favorite.group,
            logo=favorite.logo,
            raw_attributes=raw_attributes,
        )

    def _refresh_favorites_view(self) -> None:
        """Update the favorites list when the tab is active."""

        self._favorite_entries = list(self._config.favorites)
        if self._active_tab != "favorites":
            return
        try:
            list_view = self.query_one("#favorites-list", ListView)
        except Exception:  # pragma: no cover - defensive for tests without UI
            return
        list_view.clear()
        if not self._favorite_entries:
            list_view.append(ListItem(Label("No favorite channels saved")))
            self._clear_active_channel_info()
            return
        for favorite in self._favorite_entries:
            list_view.append(FavoriteListItem(favorite))
        list_view.index = 0
        first = self._favorite_entries[0]
        self._update_active_channel_info(
            self._favorite_to_channel(first), first.provider
        )
        self._set_status(f"Showing {len(self._favorite_entries)} favorite channel(s)")

    def _focus_tab_content(self, tab: str) -> None:
        if tab == "providers":
            if self._states:
                list_view = self._query_optional_widget("#provider-list", ListView)
                if list_view is not None:
                    list_view.focus()
                    return
            form = self._query_optional_widget(ProviderForm)
            if form is not None:
                form.focus_name()
        elif tab == "channels":
            if not self._channel_list_ready:
                self._pending_channel_operations.append(
                    lambda: self._focus_tab_content("channels")
                )
                return
            list_view = self._query_optional_widget("#channel-list", ListView)
            if list_view is not None:
                list_view.focus()
        elif tab == "favorites":
            list_view = self._query_optional_widget("#favorites-list", ListView)
            if list_view is not None:
                list_view.focus()
        elif tab == "logs":
            viewer = self._query_optional_widget(LogViewer)
            if viewer is not None:
                viewer.focus()

    def _require_active_tab(self, tab: str, message: str) -> bool:
        if self.is_running and self._active_tab != tab:
            self._set_status(message)
            return False
        return True

    def _set_active_tab(
        self, tab: str, *, update_widget: bool = True, focus: bool = True
    ) -> None:
        valid_tabs = {"providers", "channels", "favorites", "logs"}
        if tab not in valid_tabs:
            return
        pane_id = f"{tab}-tab"
        if update_widget:
            tabbed = self._query_optional_widget("#main-tabs", TabbedContent)
            if tabbed is not None and tabbed.active != pane_id:
                tabbed.active = pane_id
        if self._active_tab == tab:
            if focus:
                self._focus_tab_content(tab)
            return
        self._active_tab = tab
        if tab != "channels":
            self._cancel_channel_render()
        self._update_tab_buttons()
        if tab == "channels":
            try:
                query = self.query_one("#search", Input).value
            except Exception:  # pragma: no cover - tests without UI
                query = ""
            self._apply_filter(query)
        elif tab == "favorites":
            self._refresh_favorites_view()
        if focus:
            self._focus_tab_content(tab)

    def _get_selected_entry(
        self,
    ) -> tuple[Optional[str], Optional[Channel], Optional[FavoriteChannel]]:
        """Return information about the currently selected entry."""

        if self._active_tab == "channels":
            try:
                list_view = self.query_one("#channel-list", ListView)
            except Exception:
                return None, None, None
        else:
            try:
                list_view = self.query_one("#favorites-list", ListView)
            except Exception:
                return None, None, None
        index = getattr(list_view, "index", None)
        if index is None:
            return None, None, None
        if self._active_tab == "channels":
            global_index = self._channel_window_start + index
            if not self.filtered_channels or not (0 <= global_index < len(self.filtered_channels)):
                return None, None, None
            entry = self.filtered_channels[global_index]
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[1], Channel)
            ):
                provider, channel = entry
            else:
                state = self._current_state()
                provider = state.config.name if state else None
                channel = entry  # type: ignore[assignment]
            return provider, channel, None
        if not self._favorite_entries or not (0 <= index < len(self._favorite_entries)):
            return None, None, None
        favorite = self._favorite_entries[index]
        return favorite.provider, self._favorite_to_channel(favorite), favorite

    def _start_player_for_channel(self, provider_name: str, channel: Channel) -> None:
        """Launch a media player for the supplied channel."""

        key = self._channel_queue_key(provider_name, channel)

        async def runner() -> None:
            handle: Optional[PlayerHandle] = None
            try:
                handle = await launch_player(channel, preferred=self._preferred_player)
            except asyncio.CancelledError:
                log.info(
                    "Player launch cancelled for %s (%s)",
                    channel.name,
                    provider_name,
                )
                raise
            except Exception as exc:  # pragma: no cover - exercised via tests
                self._call_on_app_thread(
                    self._handle_player_failure,
                    key,
                    provider_name,
                    channel.name,
                    str(exc),
                )
                return
            self._call_on_app_thread(
                self._handle_player_started, key, provider_name, channel, handle
            )
            returncode: Optional[int] = None
            try:
                returncode = await handle.process.wait()
            except asyncio.CancelledError:
                log.info(
                    "Player task cancelled for %s (%s)", channel.name, provider_name
                )
                try:
                    handle.process.terminate()
                except ProcessLookupError:  # pragma: no cover - defensive guard
                    pass
                raise
            except Exception as exc:  # pragma: no cover - process wait failures are rare
                log.warning("Player wait failed for %s: %s", channel.name, exc)
                returncode = getattr(handle.process, "returncode", None)
            finally:
                self._call_on_app_thread(
                    self._handle_player_exit,
                    key,
                    provider_name,
                    channel.name,
                    returncode,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(runner())
        else:
            task = loop.create_task(runner())
            self._player_tasks[key] = task

    def _handle_player_started(
        self,
        key: tuple[str, str],
        provider_name: str,
        channel: Channel,
        handle: PlayerHandle,
    ) -> None:
        self._player_handles[key] = handle
        self._playing_channels[key] = (provider_name, channel)
        self._playback_started_at[key] = datetime.now(tz=timezone.utc)
        self._clear_playing_stats(key)
        self._refresh_playing_channels_panel()
        self._player_process = handle.process
        self._set_status(f"Playing {channel.name} from {provider_name}")
        log.info("Player launched for %s (%s)", channel.name, provider_name)
        self._start_player_monitor(key, handle)
        self._update_playback_timer_state()

    def _handle_player_exit(
        self,
        key: tuple[str, str],
        provider_name: str,
        channel_name: str,
        returncode: Optional[int],
    ) -> None:
        self._cancel_player_monitor(key)
        self._cleanup_player_resources(key)
        self._player_tasks.pop(key, None)
        user_stopped = key in self._player_stop_requested
        self._player_stop_requested.discard(key)
        self._refresh_playing_channels_panel()
        if not self._player_handles:
            self._player_process = None
        if user_stopped:
            log.info("Player exited for %s after stop request", channel_name)
            return
        if returncode not in (None, 0):
            message = (
                f"Playback failed for {channel_name} (exit code {returncode})"
            )
            self._set_status(message)
            log.error(
                "Player exited with code %s for %s (%s)",
                returncode,
                channel_name,
                provider_name,
            )
        else:
            self._set_status(f"Playback finished for {channel_name}")
            log.info("Player exited for %s (%s)", channel_name, provider_name)

    def _handle_player_failure(
        self,
        key: tuple[str, str],
        provider_name: str,
        channel_name: str,
        message: str,
    ) -> None:
        self._cancel_player_monitor(key)
        self._cleanup_player_resources(key)
        self._player_tasks.pop(key, None)
        self._player_stop_requested.discard(key)
        self._refresh_playing_channels_panel()
        if not self._player_handles:
            self._player_process = None
        error = f"Failed to launch player for {channel_name} ({provider_name}): {message}"
        self._set_status(error)
        log.error(error)

    def _stop_player_process(
        self,
        key: Optional[tuple[str, str]] = None,
        *,
        user_requested: bool = False,
        suppress_status: bool = False,
    ) -> None:
        if key is None:
            targets = list(
                set(self._player_handles.keys()) | set(self._player_tasks.keys())
            )
        else:
            targets = [key]
        if not targets:
            return
        if user_requested or suppress_status:
            self._player_stop_requested.update(targets)
        else:
            for target in targets:
                self._player_stop_requested.discard(target)
        for target in targets:
            self._cancel_player_monitor(target)
            task = self._player_tasks.get(target)
            if task is not None:
                task.cancel()
            handle = self._player_handles.get(target)
            if handle is not None:
                try:
                    handle.process.terminate()
                except ProcessLookupError:  # pragma: no cover - defensive guard
                    pass

    def _provider_label(self, state: ProviderState) -> str:
        details: list[str] = []

        def add_detail(fragment: Optional[str]) -> None:
            if fragment:
                details.append(escape(fragment))

        if state.loading:
            percent = int(round(max(0.0, min(state.loading_progress, 1.0)) * 100))
            add_detail(f"loading {percent}%")
        elif state.last_error:
            add_detail(f"error: {state.last_error}")

        if state.last_channel_count is not None:
            add_detail(f"{state.last_channel_count} channels")
        elif state.channels is not None:
            add_detail(f"{len(state.channels)} channels")

        if state.connection_status:
            status = state.connection_status
            if status.active_connections is not None:
                add_detail(f"active {status.active_connections}")
            if status.max_connections is not None:
                add_detail(f"max {status.max_connections}")
            if status.message:
                add_detail(status.message)

        if not details:
            add_detail("not loaded")

        provider_markup = self._provider_markup(state.config.name)
        joined = " • ".join(details)
        return f"{provider_markup} ([dim]{joined}[/dim])"

    def _refresh_provider_list(self) -> None:
        log.debug("Refreshing provider list UI")
        list_view = self.query_one("#provider-list", ListView)
        previous_index = list_view.index
        list_view.clear()
        for state in self._states:
            list_view.append(ListItem(Label(self._provider_label(state), markup=True)))
        if self._states:
            if self._active_index is not None:
                list_view.index = max(0, min(self._active_index, len(self._states) - 1))
            elif previous_index is not None:
                list_view.index = max(0, min(previous_index, len(self._states) - 1))
            else:
                list_view.index = 0
        else:
            list_view.index = None
        self._update_provider_progress_widget()
        self._update_connection_usage_widget()

    def _update_provider_progress_widget(
        self, state: Optional[ProviderState] = None
    ) -> None:
        widget = self._query_optional_widget("#provider-progress", ProviderProgress)
        if widget is None:
            return
        if state is None:
            state = self._current_state()
        if state is None or not state.loading:
            widget.hide()
            return
        widget.show_progress(
            provider=self._provider_markup(state.config.name),
            progress=state.loading_progress,
            loaded=state.loading_bytes_read,
            total=state.loading_bytes_total,
            markup=True,
        )

    def _update_connection_usage_widget(
        self, state: Optional[ProviderState] = None
    ) -> None:
        widget = self._query_optional_widget("#connection-usage", ConnectionUsageBar)
        if widget is None:
            return
        if state is None:
            state = self._current_state()
        if state is None:
            widget.show_unavailable("No provider selected")
            return
        status = state.connection_status
        if status is None:
            widget.show_unavailable("Connection status unavailable")
            return
        widget.update_status(
            active_connections=status.active_connections,
            max_connections=status.max_connections,
            percent=state.connection_usage_percent,
            message=status.message,
        )

    def _clear_channels(self, message: str = "No provider selected") -> None:
        self.filtered_channels = []
        self._clear_active_channel_info()
        self._cancel_channel_render()
        if self._active_tab != "channels":
            return
        if not self._channel_list_ready:
            self._pending_channel_operations.append(
                lambda message=message: self._clear_channels(message)
            )
            return
        list_view = self.query_one("#channel-list", ListView)
        list_view.clear()
        list_view.append(ListItem(Label(message)))
        list_view.index = 0

    def _apply_filter(self, query: str) -> None:
        log.debug("Applying channel filter: %s", query)
        self._cancel_channel_render()
        state = self._current_state()
        aggregated = self._all_channels
        search_index = self._all_channels_search_index
        if not aggregated and any(state.channels for state in self._states):
            self._rebuild_all_channels()
            aggregated = self._all_channels
            search_index = self._all_channels_search_index
        if not aggregated:
            self.filtered_channels = []
            if self._active_tab == "channels":
                if not self._channel_list_ready:
                    self._pending_channel_operations.append(
                        lambda query=query: self._apply_filter(query)
                    )
                    return
                if not self._states:
                    message = "Add a provider to load channels"
                elif state is None:
                    message = "No provider selected"
                elif state.loading:
                    message = "Loading channels…"
                else:
                    message = "Channels not loaded"
                list_view = self.query_one("#channel-list", ListView)
                list_view.clear()
                list_view.append(ListItem(Label(message)))
                list_view.index = 0
                self._clear_active_channel_info()
                self._set_status(message)
            else:
                self._clear_active_channel_info()
            return
        channels_only = self._all_channel_objects
        providers_in_view = {provider for provider, _ in aggregated}
        search_index_to_use = search_index
        if (
            state
            and state.search_index is not None
            and providers_in_view == {state.config.name}
        ):
            search_index_to_use = state.search_index
        match_indexes = filter_channels(
            channels_only,
            query,
            search_index_to_use,
            return_indices=True,
        )
        filtered: list[tuple[str, Channel]] = []
        providers_in_result: set[str] = set()
        total_available = len(aggregated)
        for index in match_indexes:
            if not (0 <= index < total_available):
                continue
            provider_name, channel = aggregated[index]
            filtered.append((provider_name, channel))
            providers_in_result.add(provider_name)
        self.filtered_channels = filtered
        if self._active_tab == "channels":
            if not self._channel_list_ready:
                self._pending_channel_operations.append(
                    lambda query=query: self._apply_filter(query)
                )
                return
            if not filtered:
                list_view = self.query_one("#channel-list", ListView)
                list_view.clear()
                list_view.append(ListItem(Label("No channels found")))
                self._clear_active_channel_info()
                list_view.index = 0
            else:
                self._start_channel_render(filtered)
        provider_count = len(providers_in_result)
        if filtered:
            status = f"Showing {len(filtered)} channel(s)"
            if provider_count:
                status = f"{status} from {provider_count} provider(s)"
        else:
            status = "No channels found"
        self._set_status(status)

    def _rebuild_all_channels(self) -> None:
        """Regenerate the aggregated channel cache across all providers."""

        aggregated: list[tuple[str, Channel]] = []
        channel_objects: list[Channel] = []
        search_index: dict[str, set[int]] = {}

        for state in self._states:
            channels = state.channels
            if not channels:
                continue
            provider_name = state.config.name
            for channel in channels:
                position = len(channel_objects)
                aggregated.append((provider_name, channel))
                channel_objects.append(channel)
                for token in channel._tokens:
                    search_index.setdefault(token, set()).add(position)

        self._all_channels = aggregated
        self._all_channel_objects = channel_objects
        self._all_channels_search_index = search_index if aggregated else None

    def _start_channel_render(
        self, channels: Sequence[tuple[str, Channel]]
    ) -> None:
        if not channels:
            return
        self._channel_render_generation += 1
        self._channel_rendered_count = 0
        self._channel_first_batch_size = 0
        self._channel_window_start = 0
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug("Rendering channels synchronously outside the event loop")
            self._render_channel_window_for_start(0)
            self._channel_rendered_count = len(channels)
            self._channel_first_batch_size = min(
                len(channels), self.CHANNEL_RENDER_BATCH_SIZE
            )
            return
        generation = self._channel_render_generation
        task = loop.create_task(self._render_channel_list(channels, generation))
        self._channel_render_task = task

        def _finalize_render(completed: asyncio.Task[None]) -> None:
            if self._channel_render_task is completed:
                self._channel_render_task = None
            try:
                completed.result()
            except asyncio.CancelledError:  # pragma: no cover - expected cancellation
                pass
            except Exception:  # pragma: no cover - logged for diagnostics
                log.exception("Channel rendering failed")

        task.add_done_callback(_finalize_render)

    def _cancel_channel_render(self) -> None:
        task = self._channel_render_task
        if task is not None and not task.done():
            task.cancel()
        self._channel_render_task = None
        self._channel_render_generation += 1
        self._channel_rendered_count = 0
        self._channel_first_batch_size = 0
        self._channel_window_start = 0

    async def _render_channel_list(
        self,
        channels: Sequence[tuple[str, Channel]] | Sequence[Channel],
        generation: int,
        provider_name: Optional[str] = None,
    ) -> None:
        if self._active_tab != "channels":
            return
        state = self._current_state()
        fallback_provider = (
            provider_name
            or (state.config.name if state else "Unknown provider")
        )
        channel_entries: list[tuple[str, Channel]] = []
        for entry in channels:
            if (
                isinstance(entry, tuple)
                and len(entry) == 2
                and isinstance(entry[1], Channel)
            ):
                provider, channel = entry
            else:
                channel = entry  # type: ignore[assignment]
                provider = fallback_provider
            channel_entries.append((provider, channel))
        if self.filtered_channels is not channel_entries:
            self.filtered_channels = list(channel_entries)
        self._render_channel_window_for_start(0)
        if generation != self._channel_render_generation:
            return
        total = len(channel_entries)
        batch_size = self.CHANNEL_RENDER_BATCH_SIZE or len(channel_entries)
        delay = self.CHANNEL_RENDER_BATCH_DELAY
        for start in range(0, total, batch_size):
            if generation != self._channel_render_generation:
                return
            end = min(start + batch_size, total)
            batch_length = end - start
            if start == 0:
                self._channel_first_batch_size = batch_length
            self._channel_rendered_count = end
            if delay:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)
        if generation == self._channel_render_generation:
            self._channel_rendered_count = total

    def _render_channel_window_for_start(
        self,
        start: int,
        *,
        selected_global_index: Optional[int] = None,
    ) -> None:
        if self._active_tab != "channels":
            return
        total = len(self.filtered_channels)
        if total == 0:
            self._channel_window_start = 0
            try:
                list_view = self.query_one("#channel-list", ListView)
            except Exception:
                if not self._channel_list_ready:
                    self._pending_channel_operations.append(
                        lambda: self._render_channel_window_for_start(0, selected_global_index=None)
                    )
                return
            list_view.clear()
            list_view.index = None
            self._clear_active_channel_info()
            return
        max_start = max(0, total - self.CHANNEL_WINDOW_SIZE)
        start = max(0, min(start, max_start))
        end = min(start + self.CHANNEL_WINDOW_SIZE, total)
        state = self._current_state()
        fallback_provider = state.config.name if state else "Unknown provider"
        try:
            list_view = self.query_one("#channel-list", ListView)
        except Exception:
            if not self._channel_list_ready:
                self._pending_channel_operations.append(
                    lambda start=start, selected_global_index=selected_global_index: self._render_channel_window_for_start(
                        start, selected_global_index=selected_global_index
                    )
                )
            return

        def render_window() -> None:
            channels_slice = self.filtered_channels[start:end]
            batch_context = (
                list_view.batch_update()
                if hasattr(list_view, "batch_update")
                else nullcontext()
            )
            with batch_context:
                list_view.clear()
                for entry in channels_slice:
                    if (
                        isinstance(entry, tuple)
                        and len(entry) == 2
                        and isinstance(entry[1], Channel)
                    ):
                        provider_name, channel = entry
                    else:
                        provider_name = fallback_provider
                        channel = entry  # type: ignore[assignment]
                    list_view.append(
                        ChannelListItem(
                            provider_name,
                            channel,
                            queued=self._is_channel_queued(provider_name, channel),
                            provider_markup=self._provider_markup(provider_name),
                        )
                    )
            target_index = selected_global_index
            if target_index is None:
                if getattr(list_view, "index", None) is None:
                    target_index = start
                else:
                    target_index = self._channel_window_start + list_view.index
            target_index = max(start, min(end - 1, target_index)) if channels_slice else None
            if target_index is not None:
                list_view.index = target_index - start
                entry = self.filtered_channels[target_index]
                if (
                    isinstance(entry, tuple)
                    and len(entry) == 2
                    and isinstance(entry[1], Channel)
                ):
                    provider_name, channel = entry
                else:
                    provider_name = fallback_provider
                    channel = entry  # type: ignore[assignment]
                self._update_active_channel_info(channel, provider_name)
            else:
                list_view.index = None
                self._clear_active_channel_info()

        self._channel_window_start = start
        render_window()

    def _maybe_shift_channel_window(self, global_index: int) -> None:
        total = len(self.filtered_channels)
        if total <= self.CHANNEL_WINDOW_SIZE:
            return
        margin = min(self.CHANNEL_WINDOW_MARGIN, self.CHANNEL_WINDOW_SIZE // 2)
        start = self._channel_window_start
        end = start + self.CHANNEL_WINDOW_SIZE
        max_start = max(0, total - self.CHANNEL_WINDOW_SIZE)
        if global_index < start + margin and start > 0:
            new_start = max(0, global_index - margin)
            if new_start != start:
                self._render_channel_window_for_start(new_start, selected_global_index=global_index)
        elif global_index >= end - margin and start < max_start:
            new_start = min(max_start, global_index - self.CHANNEL_WINDOW_SIZE + margin + 1)
            if new_start != start:
                self._render_channel_window_for_start(new_start, selected_global_index=global_index)

    def _stop_worker(self) -> None:
        if self._worker and not self._worker.is_finished:
            self._worker.cancel()
        self._worker = None
        self._current_provider_index = None

    def _queue_provider_load(self, index: int, *, force: bool = False) -> None:
        if not (0 <= index < len(self._states)):
            return
        state = self._states[index]
        if state.loading:
            return
        queue = deque(
            (existing_index, existing_force)
            for existing_index, existing_force in self._provider_load_queue
            if existing_index != index
        )
        queue.append((index, force))
        self._provider_load_queue = queue
        self._process_provider_queue()

    def _remove_provider_from_queue(self, removed_index: int) -> None:
        queue: deque[tuple[int, bool]] = deque()
        for index, force in self._provider_load_queue:
            if index == removed_index:
                continue
            if index > removed_index:
                queue.append((index - 1, force))
            else:
                queue.append((index, force))
        self._provider_load_queue = queue

    def _process_provider_queue(self) -> None:
        if self._worker and not self._worker.is_finished:
            return
        while self._provider_load_queue:
            index, force = self._provider_load_queue.popleft()
            if not (0 <= index < len(self._states)):
                continue
            started = self._load_provider(index, force=force)
            if started:
                break

    def _refresh_stale_providers(self) -> None:
        if not self._states:
            return
        now = datetime.now(tz=timezone.utc)
        ordered_indexes = list(range(len(self._states)))
        if self._active_index is not None and 0 <= self._active_index < len(self._states):
            ordered_indexes.remove(self._active_index)
            ordered_indexes.insert(0, self._active_index)
        for index in ordered_indexes:
            state = self._states[index]
            if state.loading:
                continue
            if state.channels is None:
                self._queue_provider_load(index)
                continue
            last_loaded = state.last_loaded_at
            if last_loaded is not None and now - last_loaded >= RECENT_RELOAD_THRESHOLD:
                self._queue_provider_load(index)

    def _call_on_app_thread(self, callback: Callable[..., None], *args) -> None:
        if threading.get_ident() == self._app_thread_id:
            callback(*args)
        else:
            self.call_from_thread(callback, *args)

    def _worker_finished(self) -> None:
        self._worker = None
        self._current_provider_index = None
        self._process_provider_queue()


    def _load_provider(self, index: int, *, force: bool = False) -> bool:
        state = self._states[index]
        if state.loading:
            log.info("Provider %s is already loading", state.config.name)
            return False
        if (
            not force
            and state.channels is not None
            and state.last_loaded_at is not None
        ):
            now = datetime.now(tz=timezone.utc)
            elapsed = now - state.last_loaded_at
            if elapsed < RECENT_RELOAD_THRESHOLD:
                elapsed_label = _format_timedelta(elapsed)
                message = (
                    f"Skipped automatic reload for {state.config.name}; "
                    f"last loaded {elapsed_label} ago"
                )
                self._set_status(message)
                log.info(
                    "Skipping automatic reload for %s; last loaded %s ago",
                    state.config.name,
                    elapsed_label,
                )
                return False
        if (
            self._current_provider_index is not None
            and self._current_provider_index != index
        ):
            self._queue_provider_load(self._current_provider_index)
        self._stop_worker()
        state.loading = True
        state.last_error = None
        state.channels = None
        state.connection_status = None
        state.connection_usage_percent = None
        state.loading_progress = 0.0
        state.loading_bytes_read = 0
        state.loading_bytes_total = None
        self._rebuild_all_channels()
        if self._active_tab == "channels":
            try:
                query = self.query_one("#search", Input).value
            except Exception:
                query = ""
            self._apply_filter(query)
        log.info("Beginning channel load for provider %s", state.config.name)
        self._clear_channels("Loading channels…")
        self._refresh_provider_list()
        self._set_status(f"Loading channels for {state.config.name}…")
        self._update_provider_progress_widget(state)
        self._update_connection_usage_widget(state)
        self._current_provider_index = index
        self._worker = self.run_worker(self._fetch_provider(state), name=f"provider:{state.config.name}")
        return True

    async def _fetch_provider(self, state: ProviderState) -> None:
        log.debug("Worker started for provider %s", state.config.name)
        try:
            def progress_callback(loaded: int, total: Optional[int]) -> None:
                self._call_on_app_thread(
                    self._handle_loading_progress,
                    state,
                    loaded,
                    total,
                )

            channels = await asyncio.to_thread(
                load_playlist,
                state.config.playlist_url,
                progress=progress_callback,
            )
        except Exception as exc:  # pragma: no cover - network failures depend on environment
            log.exception("Error loading playlist for %s", state.config.name)
            self._call_on_app_thread(self._handle_provider_error, state, str(exc))
        else:
            self._call_on_app_thread(self._handle_channels_loaded, state, channels)
            if state.config.api_url:
                try:
                    status = await fetch_connection_status(state.config.api_url)
                except Exception as exc:  # pragma: no cover - network failures depend on environment
                    log.exception("Error fetching status for %s", state.config.name)
                    self._call_on_app_thread(self._handle_status_error, state, str(exc))
                else:
                    self._call_on_app_thread(self._handle_status_success, state, status)
        finally:
            log.debug("Worker finished for provider %s", state.config.name)
            self._call_on_app_thread(self._worker_finished)

    def _handle_loading_progress(
        self, state: ProviderState, loaded: int, total: Optional[int]
    ) -> None:
        state.loading_bytes_read = max(0, loaded)
        if total is not None and total > 0:
            state.loading_bytes_total = total
            state.loading_progress = max(0.0, min(1.0, loaded / total))
        else:
            state.loading_bytes_total = None
            estimate = 1.0 - math.exp(-max(0, loaded) / 1_500_000.0)
            state.loading_progress = min(0.95, max(state.loading_progress, estimate))
        if state is self._current_state():
            self._update_provider_progress_widget(state)
        self._refresh_provider_list()

    def _handle_provider_error(self, state: ProviderState, message: str) -> None:
        state.loading = False
        state.last_error = message
        state.channels = None
        state.search_index = None
        state.last_loaded_at = None
        state.loading_progress = 0.0
        state.loading_bytes_read = 0
        state.loading_bytes_total = None
        state.connection_usage_percent = None
        if state is self._current_state():
            self._clear_channels(f"Failed to load channels: {message}")
            self._show_provider_form()
            form = self._query_optional_widget(ProviderForm)
            if form is not None:
                form.focus_name()
        self._refresh_provider_list()
        self._set_status(f"Failed to load {state.config.name}: {message}")
        log.error("Provider %s failed to load: %s", state.config.name, message)
        self._update_connection_usage_widget(state)

    def _handle_channels_loaded(self, state: ProviderState, channels: List[Channel]) -> None:
        state.loading = False
        state.last_error = None
        state.channels = channels
        state.search_index = build_search_index(channels)
        self._rebuild_all_channels()
        state.last_loaded_at = datetime.now(tz=timezone.utc)
        state.loading_progress = 1.0
        state.last_channel_count = len(channels)
        if state.loading_bytes_total is None:
            state.loading_bytes_total = state.loading_bytes_read
        log.info(
            "Loaded %d channels for provider %s",
            len(channels),
            state.config.name,
        )
        if self._active_tab == "channels":
            try:
                query = self.query_one("#search", Input).value
            except Exception:
                query = ""
            self._apply_filter(query)
        if state is self._current_state():
            self._set_status(
                f"Loaded {len(channels)} channels for {state.config.name}"
            )
            self._hide_provider_form()
        self._refresh_provider_list()
        self._update_provider_progress_widget(state)
        self._update_connection_usage_widget(state)
        self._update_connection_usage_widget(state)

    def _handle_status_success(self, state: ProviderState, status: ConnectionStatus) -> None:
        state.connection_status = status
        if (
            status.active_connections is not None
            and status.max_connections is not None
            and status.max_connections > 0
        ):
            state.connection_usage_percent = max(
                0.0,
                min(status.active_connections / status.max_connections, 1.0),
            )
        else:
            state.connection_usage_percent = None
        state.last_error = None
        log.info(
            "Status updated for %s: %s",
            state.config.name,
            status.as_label(),
        )
        if state is self._current_state():
            self._set_status(f"{state.config.name}: {status.as_label()}")
        self._refresh_provider_list()
        self._update_provider_progress_widget(state)
        self._update_connection_usage_widget(state)

    def _handle_status_error(self, state: ProviderState, message: str) -> None:
        state.connection_status = None
        state.connection_usage_percent = None
        state.last_error = message
        log.warning(
            "Failed to retrieve status for %s: %s",
            state.config.name,
            message,
        )
        if state is self._current_state():
            self._set_status(f"Failed to fetch status: {message}")
        self._refresh_provider_list()
        self._update_provider_progress_widget(state)
        self._update_connection_usage_widget(state)

    def _select_provider(self, index: int) -> None:
        if not (0 <= index < len(self._states)):
            self._active_index = None
            self._editing_name = None
            self._clear_channels()
            return
        if self._active_index == index and self._states[index].channels is not None:
            self._apply_filter(self.query_one("#search", Input).value)
            return
        self._active_index = index
        state = self._states[index]
        self._editing_name = state.config.name
        log.info("Selected provider %s", state.config.name)
        self.query_one(ProviderForm).populate(state.config)
        if self.provider_form_visible:
            self._show_provider_form()
        else:
            self._hide_provider_form()
        if state.channels is None:
            self._load_provider(index)
        else:
            self._apply_filter(self.query_one("#search", Input).value)
        self._refresh_provider_list()
        self._update_provider_progress_widget(state)

    def _name_exists(self, name: str) -> bool:
        return any(state.config.name == name for state in self._states)

    def _save_current_config(self) -> None:
        log.debug("Persisting configuration changes")
        save_config(self._config, self._config_path)
        self._refresh_favorites_view()

    def action_switch_tab(self, tab: str) -> None:
        self._set_active_tab(tab)

    def action_focus_active_tab_content(self) -> None:
        focused = self.focused
        if not isinstance(focused, ContentTabs):
            raise SkipAction()
        self._focus_tab_content(self._active_tab)

    def action_focus_search(self) -> None:
        log.debug("Focusing search input")
        if self._active_tab != "channels":
            self._set_status("Search is only available in the Channels tab")
            return
        self.query_one("#search", Input).focus()

    def action_clear_search(self) -> None:
        log.debug("Clearing search input")
        try:
            search = self.query_one("#search", Input)
        except Exception:
            self._apply_filter("")
            return
        if search.value:
            search.value = ""
        self._apply_filter("")
        if not search.has_focus:
            search.focus()

    def action_new_provider(self) -> None:
        if not self._require_active_tab(
            "providers", self._PROVIDERS_TAB_REQUIRED_STATUS
        ):
            return
        form = self.query_one(ProviderForm)
        self._show_provider_form()
        form.clear()
        form.focus_name()
        self._editing_name = None
        self._set_status("Creating new provider…")
        log.info("Creating a new provider entry")

    def action_save_provider(self) -> None:
        if self._save_provider():
            self._hide_provider_form()

    def action_probe_player(self) -> None:
        if self._probing_player:
            self._set_status("Player probe already in progress")
            return
        self._probing_player = True
        self._set_status("Probing media player…")

        def worker() -> None:
            try:
                summary = probe_player(preferred=self._preferred_player)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                self._call_on_app_thread(
                    self._handle_probe_result,
                    False,
                    str(exc),
                )
            else:
                self._call_on_app_thread(
                    self._handle_probe_result,
                    True,
                    summary,
                )

        threading.Thread(target=worker, daemon=True).start()

    def _handle_probe_result(self, success: bool, message: str) -> None:
        self._probing_player = False
        if success:
            status = f"Player available: {message}"
            self._set_status(status)
            log.info("Player probe succeeded: %s", message)
        else:
            status = f"Player probe failed: {message}"
            self._set_status(status)
            log.error("Player probe failed: %s", message)

    def _save_provider(self) -> bool:
        form = self.query_one(ProviderForm)
        provider = form.read()
        if not provider.name:
            self._set_status("Provider name is required")
            form.focus_name()
            return False
        if not provider.playlist_url:
            self._set_status("Playlist URL is required")
            self.query_one("#provider-playlist", Input).focus()
            return False
        if self._editing_name is None:
            if self._name_exists(provider.name):
                self._set_status("A provider with that name already exists")
                return False
            state = ProviderState(provider)
            self._states.append(state)
            self._config.add_or_update(provider)
            self._active_index = len(self._states) - 1
            self._editing_name = provider.name
            self._save_current_config()
            self._refresh_provider_list()
            self._select_provider(self._active_index)
            self._set_status(
                f"Added provider {provider.name} (saved to {self._config_destination_label()})"
            )
            log.info("Added provider %s", provider.name)
            return True
        if self._active_index is None:
            self._set_status("No provider selected")
            return False
        state = self._states[self._active_index]
        previous_name = state.config.name
        if provider.name != previous_name and self._name_exists(provider.name):
            self._set_status("A provider with that name already exists")
            return False
        state.config = provider
        state.channels = None
        state.connection_status = None
        state.last_error = None
        state.last_loaded_at = None
        if provider.name != previous_name:
            self._config.remove(previous_name)
        self._config.add_or_update(provider)
        self._editing_name = provider.name
        self._save_current_config()
        self._refresh_provider_list()
        self._select_provider(self._active_index)
        self._set_status(
            f"Updated provider {provider.name} (saved to {self._config_destination_label()})"
        )
        log.info("Updated provider %s (previously %s)", provider.name, previous_name)
        return True

    def action_toggle_channel_queue(self) -> None:
        if self._active_tab != "channels":
            self._set_status("Queue channels from the channel browser")
            return
        provider_name, channel, _ = self._get_selected_entry()
        if provider_name is None or channel is None:
            self._set_status("No channel selected")
            return
        key = self._channel_queue_key(provider_name, channel)
        if key in self._queued_channels:
            self._queue_remove(key)
            self._update_channel_item_queue_state(provider_name, channel, False)
            self._set_status(f"Removed {channel.name} from queue")
            log.info(
                "Removed %s (%s) from playback queue", channel.name, provider_name
            )
        else:
            self._queue_add(key)
            self._update_channel_item_queue_state(provider_name, channel, True)
            self._set_status(f"Queued {channel.name} for playback")
            log.info("Queued %s (%s) for playback", channel.name, provider_name)
        self._refresh_selected_channels_panel()

    def action_play_channel(self) -> None:
        if self._queued_channels:
            launched = 0
            missing_keys: list[tuple[str, str]] = []
            for key in list(self._queued_order):
                if key not in self._queued_channels:
                    continue
                provider_name, channel_identifier = key
                channel = self._find_channel_by_identifier(
                    provider_name, channel_identifier
                )
                if channel is None:
                    log.warning(
                        "Skipping queued channel %s (%s); channel not available",
                        channel_identifier,
                        provider_name,
                    )
                    missing_keys.append(key)
                    continue
                if key in self._player_handles or key in self._player_tasks:
                    log.debug(
                        "Skipping queued channel %s (%s); already launching or playing",
                        channel_identifier,
                        provider_name,
                    )
                    continue
                self._start_player_for_channel(provider_name, channel)
                launched += 1
            if missing_keys:
                self._queue_remove_many(missing_keys)
            self._refresh_channel_queue_indicators()
            self._refresh_selected_channels_panel()
            if launched:
                message = f"Launched {launched} queued channel(s)"
                self._set_status(message)
                log.info(message)
            else:
                self._set_status("No queued channels available for playback")
            return
        provider_name, channel, _ = self._get_selected_entry()
        if provider_name is None or channel is None:
            self._set_status("No channel selected")
            return
        key = self._channel_queue_key(provider_name, channel)
        if key in self._queued_channels:
            self._queue_remove(key)
            self._update_channel_item_queue_state(provider_name, channel, False)
            self._refresh_selected_channels_panel()
        self._set_status(f"Launching player for {channel.name}…")
        self._start_player_for_channel(provider_name, channel)

    def action_stop_channel(self) -> None:
        active_keys = self._active_playback_keys()
        if not active_keys:
            self._set_status("No active player to stop")
            return
        provider_name, channel, _ = self._get_selected_entry()
        if provider_name is None or channel is None:
            self._set_status("Select a channel to stop")
            return
        key = self._channel_queue_key(provider_name, channel)
        if key not in active_keys:
            self._set_status(f"{channel.name} is not currently playing")
            return
        self._stop_channels([key])

    def action_stop_all_playback(self) -> None:
        active_keys = self._active_playback_keys()
        if not active_keys:
            self._set_status("No active player to stop")
            return
        self._stop_channels(active_keys)

    def action_show_now_playing(self) -> None:
        if not self._playing_channels:
            self._set_status("No streams currently playing")
            return
        modal = NowPlayingModal(self)
        modal.set_entries(self._collect_now_playing_entries())
        self._now_playing_modal = modal
        self.push_screen(modal)

    def action_toggle_favorite(self) -> None:
        provider_name, channel, favorite_entry = self._get_selected_entry()
        if self._active_tab == "channels":
            if provider_name is None or channel is None:
                self._set_status("No channel selected")
                return
            existing = self._config.find_favorite(provider_name, channel.url)
            if existing:
                self._config.remove_favorite(provider_name, channel.url)
                self._set_status(f"Removed {channel.name} from favorites")
            else:
                favorite = FavoriteChannel(
                    provider=provider_name,
                    channel_name=channel.name,
                    channel_url=channel.url,
                    group=channel.group,
                    logo=channel.logo,
                )
                self._config.add_favorite(favorite)
                self._set_status(f"Added {channel.name} to favorites")
            self._save_current_config()
        else:
            if favorite_entry is None:
                self._set_status("No favorite selected")
                return
            self._config.remove_favorite(favorite_entry.provider, favorite_entry.channel_url)
            self._set_status(f"Removed {favorite_entry.channel_name} from favorites")
            self._save_current_config()

    def action_delete_provider(self) -> None:
        if not self._require_active_tab(
            "providers", self._PROVIDERS_TAB_REQUIRED_STATUS
        ):
            return
        if self._active_index is None or not self._states:
            self._set_status("No provider selected")
            return
        state = self._states.pop(self._active_index)
        self._remove_provider_from_queue(self._active_index)
        removed_keys = [
            key for key in self._queued_channels if key[0] == state.config.name
        ]
        if removed_keys:
            self._queue_remove_many(removed_keys)
            self._refresh_channel_queue_indicators()
            self._refresh_selected_channels_panel()
        self._config.remove(state.config.name)
        self._save_current_config()
        log.warning("Deleted provider %s", state.config.name)
        location = self._config_destination_label()
        if self._states:
            self._active_index = min(self._active_index, len(self._states) - 1)
            self._editing_name = self._states[self._active_index].config.name
            self._refresh_provider_list()
            self._select_provider(self._active_index)
            self._set_status(
                f"Removed provider {state.config.name} (saved to {location})"
            )
        else:
            self._active_index = None
            self._editing_name = None
            self._refresh_provider_list()
            self._clear_channels("Add a provider to load channels")
            self._set_status(
                f"Removed provider {state.config.name} (saved to {location})"
            )
            self._show_provider_form()
            self.query_one(ProviderForm).clear()
            self.query_one(ProviderForm).focus_name()

    def action_reload_provider(self) -> None:
        if not self._require_active_tab(
            "providers", self._PROVIDERS_TAB_REQUIRED_STATUS
        ):
            return
        if self._active_index is None:
            self._set_status("No provider selected")
            return
        index = self._active_index
        state = self._states[index]
        if state.last_loaded_at is not None:
            now = datetime.now(tz=timezone.utc)
            elapsed = now - state.last_loaded_at
            if elapsed < RECENT_RELOAD_THRESHOLD:
                elapsed_label = _format_timedelta(elapsed)
                log.info(
                    "Reload requested for %s within %s; awaiting confirmation",
                    state.config.name,
                    elapsed_label,
                )
                dialog = ReloadConfirmation(state.config.name, elapsed)
                self.push_screen(
                    dialog,
                    callback=lambda result, *, index=index, elapsed=elapsed: self._handle_reload_confirmation(
                        index, result, elapsed
                    ),
                )
                self._set_status(
                    f"Confirm reload of {state.config.name}? Last loaded {elapsed_label} ago."
                )
                return
        log.info("Reloading provider %s", state.config.name)
        self._load_provider(index, force=True)

    async def on_shutdown(self) -> None:
        self._stop_player_process(suppress_status=True)
        self._cancel_player_monitor()
        self._cleanup_player_resources()
        for task in list(self._player_tasks.values()):
            if not task.done():
                task.cancel()
        self._player_tasks.clear()
        self._player_stop_requested.clear()
        self._playing_channels.clear()
        self._clear_playing_stats()
        self._refresh_playing_channels_panel()
        self._player_process = None

    def _handle_reload_confirmation(
        self, index: int, confirmed: Optional[bool], elapsed: Optional[timedelta]
    ) -> None:
        if not (0 <= index < len(self._states)):
            return
        state = self._states[index]
        elapsed_label = _format_timedelta(elapsed) if elapsed else "an unknown duration"
        if confirmed:
            log.info(
                "Reload confirmed for %s after %s",
                state.config.name,
                elapsed_label,
            )
            self._load_provider(index, force=True)
        else:
            log.info(
                "Reload cancelled for %s after %s",
                state.config.name,
                elapsed_label,
            )
            self._set_status(f"Reload cancelled for {state.config.name}")

    @on(Button.Pressed, "#provider-new")
    def _on_new_pressed(self, _: Button.Pressed) -> None:
        self.action_new_provider()

    @on(Button.Pressed, "#provider-save")
    def _on_save_pressed(self, _: Button.Pressed) -> None:
        self.action_save_provider()

    @on(Button.Pressed, "#provider-delete")
    def _on_delete_pressed(self, _: Button.Pressed) -> None:
        self.action_delete_provider()

    @on(Button.Pressed, "#provider-reset")
    def _on_reset_pressed(self, _: Button.Pressed) -> None:
        if self._editing_name is None:
            self.query_one(ProviderForm).clear()
        else:
            state = self._current_state()
            if state:
                self.query_one(ProviderForm).populate(state.config)
        self._set_status("Form reset")
        log.debug("Provider form reset")

    @on(Button.Pressed, "#channel-play")
    def _on_channel_play(self, _: Button.Pressed) -> None:
        self.action_play_channel()

    @on(Button.Pressed, "#channel-stop")
    def _on_channel_stop(self, _: Button.Pressed) -> None:
        self.action_stop_channel()

    @on(Button.Pressed, "#playing-stop")
    def _on_playing_stop(self, _: Button.Pressed) -> None:
        self.action_stop_all_playback()

    @on(Button.Pressed, "#playing-manage")
    def _on_playing_manage(self, _: Button.Pressed) -> None:
        self.action_show_now_playing()

    @on(Button.Pressed, "#channel-probe")
    def _on_channel_probe(self, _: Button.Pressed) -> None:
        self.action_probe_player()

    @on(Button.Pressed, "#channel-favorite")
    def _on_channel_favorite(self, _: Button.Pressed) -> None:
        self.action_toggle_favorite()

    @on(TabbedContent.TabActivated, "#main-tabs")
    def _on_main_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        pane_id = (event.pane.id or "").removesuffix("-tab")
        if not pane_id:
            return
        self._set_active_tab(pane_id, update_widget=False)

    @on(ListView.Highlighted, "#provider-list")
    def _on_provider_highlighted(self, event: ListView.Highlighted) -> None:
        index = event.list_view.index
        if index is None:
            return
        if index == self._active_index:
            return
        if self._states and 0 <= index < len(self._states):
            self._select_provider(index)
            log.debug("Provider list highlighted index %s", index)

    @on(ListView.Selected, "#provider-list")
    def _on_provider_selected(self, event: ListView.Selected) -> None:
        index = event.index
        if index is None or index == self._active_index:
            return
        if self._states and 0 <= index < len(self._states):
            self._select_provider(index)
            log.debug("Provider list selected index %s", index)

    @on(Input.Changed, "#search")
    def on_search_changed(self, event: Input.Changed) -> None:
        log.debug("Search changed: %s", event.value)
        self._apply_filter(event.value)

    @on(ListView.Highlighted, "#channel-list")
    def on_channel_highlighted(self, event: ListView.Highlighted) -> None:
        if not isinstance(event.item, ChannelListItem):
            self._clear_active_channel_info()
            log.debug("Channel highlighted: %s", None)
            return
        local_index = getattr(event.list_view, "index", None)
        if local_index is None:
            self._clear_active_channel_info()
            log.debug("Channel highlighted: %s", None)
            return
        global_index = self._channel_window_start + local_index
        self._maybe_shift_channel_window(global_index)
        local_index = getattr(event.list_view, "index", None)
        if local_index is None:
            self._clear_active_channel_info()
            log.debug("Channel highlighted: %s", None)
            return
        global_index = self._channel_window_start + local_index
        if not self.filtered_channels or not (0 <= global_index < len(self.filtered_channels)):
            self._clear_active_channel_info()
            log.debug("Channel highlighted: %s", None)
            return
        entry = self.filtered_channels[global_index]
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[1], Channel)
        ):
            provider, channel = entry
        else:
            state = self._current_state()
            provider = state.config.name if state else None
            channel = entry  # type: ignore[assignment]
        self._update_active_channel_info(channel, provider)
        log.debug("Channel highlighted: %s", getattr(channel, "name", None))

    @on(ListView.Highlighted, "#favorites-list")
    def on_favorite_highlighted(self, event: ListView.Highlighted) -> None:
        item = event.item
        if isinstance(item, FavoriteListItem):
            channel = self._favorite_to_channel(item.favorite)
            provider = item.favorite.provider
            self._update_active_channel_info(channel, provider)
            log.debug("Favorite highlighted: %s", item.favorite.channel_name)
        else:
            self._clear_active_channel_info()
            log.debug("Favorite highlighted: %s", None)


__all__ = ["StreamdeckApp"]
