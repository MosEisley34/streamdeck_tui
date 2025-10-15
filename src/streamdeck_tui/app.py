"""Textual application implementing the IPTV management TUI."""
from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow running this module directly via ``python app.py`` during local development.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "streamdeck_tui"

import asyncio
from dataclasses import dataclass
from typing import List, Optional

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.worker import Worker
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Static,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    raise ModuleNotFoundError(
        "The 'textual' package is required to run streamdeck_tui. "
        "Install dependencies with 'pip install -e .[dev]' or 'pip install streamdeck-tui'."
    ) from exc

from .config import AppConfig, ProviderConfig, CONFIG_PATH, save_config
from .playlist import Channel, filter_channels, load_playlist
from .providers import ConnectionStatus, fetch_connection_status


@dataclass
class ProviderState:
    """Runtime state tracked for each provider."""

    config: ProviderConfig
    channels: Optional[List[Channel]] = None
    connection_status: Optional[ConnectionStatus] = None
    last_error: Optional[str] = None
    loading: bool = False


class ChannelListItem(ListItem):
    """Render an IPTV channel in the list."""

    DEFAULT_CSS = """ListItem { padding: 0 1; }"""

    def __init__(self, channel: Channel) -> None:
        super().__init__(Label(channel.name, id="channel-name"))
        self.channel = channel


class ChannelInfo(Static):
    """Display information about the currently selected channel."""

    channel: reactive[Optional[Channel]] = reactive(None)

    def watch_channel(self, channel: Optional[Channel]) -> None:
        if channel is None:
            self.update("Select a channel to view details.")
            return
        lines = [f"[b]{channel.name}[/b]"]
        if channel.group:
            lines.append(f"Group: {channel.group}")
        if channel.logo:
            lines.append(f"Logo: {channel.logo}")
        if channel.raw_attributes:
            attributes = "\n".join(
                f"• {key}: {value}" for key, value in sorted(channel.raw_attributes.items())
            )
            lines.append(f"Attributes:\n{attributes}")
        self.update("\n".join(lines))


class ProviderForm(Static):
    """Form used to edit provider configuration."""

    DEFAULT_CSS = """
    ProviderForm {
        border: solid green;
        padding: 1;
        height: auto;
    }
    ProviderForm Label {
        text-style: bold;
    }
    ProviderForm Input {
        margin-bottom: 1;
    }
    ProviderForm #form-buttons {
        align-horizontal: right;
        padding-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Provider name")
        yield Input(placeholder="Name", id="provider-name")
        yield Label("Playlist URL")
        yield Input(placeholder="https://example.com/playlist.m3u", id="provider-playlist")
        yield Label("API URL (optional)")
        yield Input(placeholder="https://example.com/status", id="provider-api")
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


class StatusBar(Static):
    """A simple status bar widget."""

    status: reactive[str] = reactive("Ready")

    def watch_status(self, status: str) -> None:
        self.update(status)


_DEFAULT_CSS = """Screen {
    layout: vertical;
}

#layout {
    height: 1fr;
    padding: 1 2;
    gap: 2;
}

#providers-pane,
#channels-pane {
    height: 1fr;
    gap: 1;
}

#providers-pane {
    width: 2fr;
}

#channels-pane {
    width: 4fr;
}

#provider-list,
#channel-list {
    height: 1fr;
    border: solid $surface 1;
    padding: 0;
}

#provider-actions {
    gap: 1;
}

ProviderForm {
    width: 1fr;
}

#channel-info {
    border: solid $accent 1;
    padding: 1;
    height: auto;
}

#status {
    padding: 0 1;
    height: 3;
    border-top: solid $surface 1;
}
"""


def _load_stylesheet() -> str:
    """Return the stylesheet bundled with the package or a baked-in fallback."""

    try:
        css_resource = resources.files(__package__).joinpath("streamdeck.css")
        return css_resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return _DEFAULT_CSS


class StreamdeckApp(App[None]):
    """Main Textual application."""

    CSS_PATH = None
    CSS = _load_stylesheet()
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "focus_search", "Search"),
        Binding("escape", "clear_search", "Clear search"),
        Binding("n", "new_provider", "New provider"),
        Binding("ctrl+s", "save_provider", "Save provider"),
        Binding("delete", "delete_provider", "Delete provider"),
        Binding("r", "reload_provider", "Reload provider"),
    ]

    def __init__(
        self,
        config: AppConfig,
        *,
        config_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.css_path = []
        self._config = config
        self._config_path = config_path or CONFIG_PATH
        self._states: list[ProviderState] = [ProviderState(provider) for provider in config.providers]
        self._active_index: Optional[int] = 0 if self._states else None
        self._editing_name: Optional[str] = (
            self._states[0].config.name if self._states else None
        )
        self.filtered_channels: List[Channel] = []
        self._worker: Optional[Worker] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="layout"):
            with Vertical(id="providers-pane"):
                yield Label("Providers", id="providers-title")
                yield ListView(id="provider-list")
                with Horizontal(id="provider-actions"):
                    yield Button("New", id="provider-new", variant="primary")
                    yield Button("Delete", id="provider-delete", variant="warning")
                yield ProviderForm(id="provider-form")
            with Vertical(id="channels-pane"):
                yield Input(placeholder="Search channels…", id="search")
                yield ListView(id="channel-list")
                yield ChannelInfo(id="channel-info")
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_provider_list()
        if self._states:
            self._select_provider(self._active_index or 0)
            self.query_one("#provider-list", ListView).focus()
        else:
            self.query_one(StatusBar).status = "Add a provider to get started"
            self.query_one(ProviderForm).focus_name()

    def _current_state(self) -> Optional[ProviderState]:
        if self._active_index is None:
            return None
        if 0 <= self._active_index < len(self._states):
            return self._states[self._active_index]
        return None

    def _set_status(self, message: str) -> None:
        self.query_one(StatusBar).status = message

    def _provider_label(self, state: ProviderState) -> str:
        if state.loading:
            detail = "loading…"
        elif state.last_error:
            detail = f"error: {state.last_error}"
        elif state.connection_status:
            detail = state.connection_status.as_label()
        elif state.channels is not None:
            detail = f"{len(state.channels)} channels"
        else:
            detail = "not loaded"
        return f"{state.config.name} ({detail})"

    def _refresh_provider_list(self) -> None:
        list_view = self.query_one("#provider-list", ListView)
        previous_index = list_view.index
        list_view.clear()
        for state in self._states:
            list_view.append(ListItem(Label(self._provider_label(state))))
        if self._active_index is not None and self._states:
            list_view.index = self._active_index
        elif previous_index < len(self._states):
            list_view.index = previous_index

    def _clear_channels(self, message: str = "No provider selected") -> None:
        list_view = self.query_one("#channel-list", ListView)
        list_view.clear()
        list_view.append(ListItem(Label(message)))
        self.query_one(ChannelInfo).channel = None
        self.filtered_channels = []

    def _apply_filter(self, query: str) -> None:
        state = self._current_state()
        list_view = self.query_one("#channel-list", ListView)
        list_view.clear()
        info = self.query_one(ChannelInfo)
        if state is None:
            list_view.append(ListItem(Label("No provider selected")))
            info.channel = None
            self.filtered_channels = []
            return
        if state.channels is None:
            if state.loading:
                list_view.append(ListItem(Label("Loading channels…")))
            else:
                list_view.append(ListItem(Label("Channels not loaded")))
            info.channel = None
            self.filtered_channels = []
            return
        channels = filter_channels(state.channels, query)
        self.filtered_channels = channels
        if not channels:
            list_view.append(ListItem(Label("No channels found")))
            info.channel = None
        else:
            for channel in channels:
                list_view.append(ChannelListItem(channel))
            list_view.index = 0
            info.channel = channels[0]
        self._set_status(f"Showing {len(channels)} channels for {state.config.name}")

    def _stop_worker(self) -> None:
        if self._worker and not self._worker.is_finished:
            self._worker.cancel()
        self._worker = None
    def _worker_finished(self) -> None:
        self._worker = None


    def _load_provider(self, index: int) -> None:
        state = self._states[index]
        self._stop_worker()
        state.loading = True
        state.last_error = None
        self._refresh_provider_list()
        self._set_status(f"Loading channels for {state.config.name}…")
        self._worker = self.run_worker(self._fetch_provider(state), name=f"provider:{state.config.name}")

    async def _fetch_provider(self, state: ProviderState) -> None:
        try:
            channels = await asyncio.to_thread(load_playlist, state.config.playlist_url)
        except Exception as exc:  # pragma: no cover - network failures depend on environment
            self.call_from_thread(self._handle_provider_error, state, str(exc))
        else:
            self.call_from_thread(self._handle_channels_loaded, state, channels)
            if state.config.api_url:
                try:
                    status = await fetch_connection_status(state.config.api_url)
                except Exception as exc:  # pragma: no cover - network failures depend on environment
                    self.call_from_thread(self._handle_status_error, state, str(exc))
                else:
                    self.call_from_thread(self._handle_status_success, state, status)
        finally:
            self.call_from_thread(self._worker_finished)

    def _handle_provider_error(self, state: ProviderState, message: str) -> None:
        state.loading = False
        state.last_error = message
        state.channels = None
        if state is self._current_state():
            self._clear_channels(f"Failed to load channels: {message}")
        self._refresh_provider_list()
        self._set_status(f"Failed to load {state.config.name}: {message}")

    def _handle_channels_loaded(self, state: ProviderState, channels: List[Channel]) -> None:
        state.loading = False
        state.last_error = None
        state.channels = channels
        if state is self._current_state():
            self._apply_filter(self.query_one("#search", Input).value)
            self._set_status(
                f"Loaded {len(channels)} channels for {state.config.name}"
            )
        self._refresh_provider_list()

    def _handle_status_success(self, state: ProviderState, status: ConnectionStatus) -> None:
        state.connection_status = status
        state.last_error = None
        if state is self._current_state():
            self._set_status(f"{state.config.name}: {status.as_label()}")
        self._refresh_provider_list()

    def _handle_status_error(self, state: ProviderState, message: str) -> None:
        state.connection_status = None
        state.last_error = message
        if state is self._current_state():
            self._set_status(f"Failed to fetch status: {message}")
        self._refresh_provider_list()

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
        self.query_one(ProviderForm).populate(state.config)
        if state.channels is None:
            self._clear_channels("Loading channels…")
            self._load_provider(index)
        else:
            self._apply_filter(self.query_one("#search", Input).value)
        self._refresh_provider_list()

    def _name_exists(self, name: str) -> bool:
        return any(state.config.name == name for state in self._states)

    def _save_current_config(self) -> None:
        save_config(self._config, self._config_path)

    def action_focus_search(self) -> None:
        self.query_one("#search", Input).focus()

    def action_clear_search(self) -> None:
        search = self.query_one("#search", Input)
        search.value = ""
        self._apply_filter("")

    def action_new_provider(self) -> None:
        form = self.query_one(ProviderForm)
        form.clear()
        form.focus_name()
        self._editing_name = None
        self._set_status("Creating new provider…")

    def action_save_provider(self) -> None:
        self._save_provider()

    def _save_provider(self) -> None:
        form = self.query_one(ProviderForm)
        provider = form.read()
        if not provider.name:
            self._set_status("Provider name is required")
            form.focus_name()
            return
        if not provider.playlist_url:
            self._set_status("Playlist URL is required")
            self.query_one("#provider-playlist", Input).focus()
            return
        if self._editing_name is None:
            if self._name_exists(provider.name):
                self._set_status("A provider with that name already exists")
                return
            state = ProviderState(provider)
            self._states.append(state)
            self._config.add_or_update(provider)
            self._active_index = len(self._states) - 1
            self._editing_name = provider.name
            self._save_current_config()
            self._refresh_provider_list()
            self._select_provider(self._active_index)
            self._set_status(f"Added provider {provider.name}")
        else:
            if self._active_index is None:
                self._set_status("No provider selected")
                return
            state = self._states[self._active_index]
            previous_name = state.config.name
            if provider.name != previous_name and self._name_exists(provider.name):
                self._set_status("A provider with that name already exists")
                return
            state.config = provider
            state.channels = None
            state.connection_status = None
            state.last_error = None
            if provider.name != previous_name:
                self._config.remove(previous_name)
            self._config.add_or_update(provider)
            self._editing_name = provider.name
            self._save_current_config()
            self._refresh_provider_list()
            self._select_provider(self._active_index)
            self._set_status(f"Updated provider {provider.name}")

    def action_delete_provider(self) -> None:
        if self._active_index is None or not self._states:
            self._set_status("No provider selected")
            return
        state = self._states.pop(self._active_index)
        self._config.remove(state.config.name)
        self._save_current_config()
        if self._states:
            self._active_index = min(self._active_index, len(self._states) - 1)
            self._editing_name = self._states[self._active_index].config.name
            self._refresh_provider_list()
            self._select_provider(self._active_index)
            self._set_status(f"Removed provider {state.config.name}")
        else:
            self._active_index = None
            self._editing_name = None
            self._refresh_provider_list()
            self._clear_channels("Add a provider to load channels")
            self._set_status(f"Removed provider {state.config.name}")
            self.query_one(ProviderForm).clear()
            self.query_one(ProviderForm).focus_name()

    def action_reload_provider(self) -> None:
        if self._active_index is None:
            self._set_status("No provider selected")
            return
        state = self._states[self._active_index]
        state.channels = None
        state.connection_status = None
        state.last_error = None
        self._clear_channels("Loading channels…")
        self._load_provider(self._active_index)

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

    @on(ListView.Highlighted, "#provider-list")
    def _on_provider_highlighted(self, event: ListView.Highlighted) -> None:
        if event.index is None:
            return
        if event.index == self._active_index:
            return
        if self._states and 0 <= event.index < len(self._states):
            self._select_provider(event.index)

    @on(Input.Changed, "#search")
    def on_search_changed(self, event: Input.Changed) -> None:
        self._apply_filter(event.value)

    @on(ListView.Highlighted, "#channel-list")
    def on_channel_highlighted(self, event: ListView.Highlighted) -> None:
        info = self.query_one(ChannelInfo)
        item = event.item
        if isinstance(item, ChannelListItem):
            info.channel = item.channel
        else:
            info.channel = None


__all__ = ["StreamdeckApp"]
