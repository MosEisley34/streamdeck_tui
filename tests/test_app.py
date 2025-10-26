"""Unit tests for StreamdeckApp helpers."""

import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)


class DummyLogger:
    """Minimal logger capturing info-level messages for assertions."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - trivial
        self.messages.append(message % args if args else message)

    def debug(self, *args, **kwargs) -> None:  # pragma: no cover - unused in tests
        pass

    def warning(self, *args, **kwargs) -> None:  # pragma: no cover - unused in tests
        pass

    def error(self, *args, **kwargs) -> None:  # pragma: no cover - unused in tests
        pass

    def exception(self, *args, **kwargs) -> None:  # pragma: no cover - unused in tests
        pass


class DummyPlayingPanel:
    """Minimal stub for PlayingChannelsPanel interactions."""

    def __init__(self, selected_key: Optional[tuple[str, str]] = None) -> None:
        self._selected_key = selected_key
        self.entries: list[str] | None = None
        self.keys: list[tuple[str, str]] | None = None

    def update_entries(
        self, entries: Iterable[str], keys: Iterable[tuple[str, str]]
    ) -> None:
        self.entries = list(entries)
        self.keys = list(keys)

    def get_selected_key(self) -> Optional[tuple[str, str]]:
        return self._selected_key


def test_custom_themes_registered() -> None:
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig

    app = StreamdeckApp(AppConfig())

    assert "solarized-dark" in app.available_themes
    assert "solarized-light" in app.available_themes
    assert app.theme == "solarized-dark"


def test_config_theme_used_when_provided() -> None:
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig

    app = StreamdeckApp(AppConfig(theme="solarized-light"))

    assert app.theme == "solarized-light"


def test_search_input_allows_editing_keys_without_forwarding() -> None:
    """Editing keys should remain within the input widget for processing."""

    from textual import events

    from streamdeck_tui.app import SearchInput

    widget = SearchInput()
    event = events.Key("backspace", None)

    asyncio.run(widget.on_key(event))

    assert not event.is_forwarded


def test_tabbed_layout_and_log_viewer() -> None:
    """The main application should expose the tabbed UI and surface log messages."""

    from textual.widgets import TabPane, TabbedContent

    from streamdeck_tui.app import LogViewer, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.logging_utils import get_logger

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    async def run_app() -> None:
        async with app.run_test() as pilot:
            tabbed = app.query_one("#main-tabs", TabbedContent)
            pane_ids = {pane.id for pane in tabbed.query(TabPane)}
            assert {"providers-tab", "channels-tab", "favorites-tab", "logs-tab"} <= pane_ids

            viewer = app.query_one(LogViewer)
            logger = get_logger("tests.tabbed")
            logger.info("Hello from tests")
            await pilot.pause()

            messages = viewer.get_messages()
            assert any("Hello from tests" in message for message in messages)

    asyncio.run(run_app())


def test_provider_form_hidden_until_new_action() -> None:
    """The provider form should start hidden and reveal after requesting a new provider."""

    from textual.widgets import ListView

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    app._states[0].channels = []

    async def run_app() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            container = app.query_one("#provider-form-container")
            assert not container.display
            provider_list = app.query_one("#provider-list", ListView)
            assert provider_list.has_focus

            app.action_new_provider()
            await pilot.pause()

            assert container.display
            assert app.provider_form_visible

    asyncio.run(run_app())


def test_providers_do_not_auto_reload_on_start(monkeypatch) -> None:
    """Providers should wait for an explicit reload after the UI mounts."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    load_calls: list[tuple[int, bool]] = []

    def fake_load(self, index: int, *, force: bool = False) -> None:
        load_calls.append((index, force))

    messages: list[str] = []

    def fake_clear(self, message: str = "No provider selected") -> None:
        messages.append(message)

    statuses: list[str] = []

    def capture_status(self, message: str) -> None:
        statuses.append(message)

    monkeypatch.setattr(StreamdeckApp, "_load_provider", fake_load, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_clear_channels", fake_clear, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_set_status", capture_status, raising=False)

    async def run_app() -> None:
        async with app.run_test():
            await asyncio.sleep(0)

    asyncio.run(run_app())

    assert not load_calls
    assert messages and messages[-1] == "Press R to load channels"
    assert any("Press R to load channels" in status for status in statuses)


def test_cached_channels_loaded_on_startup(tmp_path, monkeypatch) -> None:
    """Cached channels should be restored before the UI renders."""

    from streamdeck_tui import database
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    monkeypatch.setattr(database, "CHANNEL_DATABASE_PATH", tmp_path / "channels.sqlite")
    provider = ProviderConfig(name="Cached", playlist_url="http://example.com")
    channel = Channel(
        name="Cached Channel",
        url="http://example.com/cached",
        group="News",
        logo=None,
        raw_attributes={"tvg-id": "cached"},
    )
    database.save_channels(provider.name, [channel])

    app = StreamdeckApp(AppConfig(providers=[provider]))

    assert app._states
    assert app._states[0].channels is not None
    assert app._states[0].channels[0].name == "Cached Channel"


def test_action_quit_closes_app() -> None:
    """The quit action should stop the application without errors."""

    from streamdeck_tui.app import ChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig

    async def run_app() -> bool:
        app = StreamdeckApp(AppConfig())
        async with app.run_test() as pilot:
            assert app.is_running
            await app.action_quit()
            await pilot.pause()
        return app.is_running

    is_running = asyncio.run(run_app())
    assert not is_running


def test_provider_actions_require_active_tab() -> None:
    """Provider shortcuts should be ignored when the Providers tab is inactive."""

    import types

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    statuses: list[str] = []

    async def run_app() -> tuple[int, int, Optional[str], Optional[str]]:
        app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

        async with app.run_test() as pilot:
            def capture_status(self: StreamdeckApp, message: str) -> None:
                statuses.append(message)

            app._set_status = types.MethodType(capture_status, app)  # type: ignore[assignment]
            app._set_active_tab("channels")
            await pilot.pause()

            initial_count = len(app._states)
            initial_editing = app._editing_name

            app.action_new_provider()
            app.action_delete_provider()
            await pilot.pause()

            return initial_count, len(app._states), initial_editing, app._editing_name

    initial_count, final_count, initial_editing, final_editing = asyncio.run(run_app())

    assert initial_count == final_count
    assert initial_editing == final_editing
    assert StreamdeckApp._PROVIDERS_TAB_REQUIRED_STATUS in statuses


def test_refresh_provider_list_handles_none_previous_index(monkeypatch) -> None:
    """Refreshing the provider list should not compare None indexes."""

    from streamdeck_tui.app import ProviderState, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")
    app._active_index = None
    app._states = [ProviderState(provider)]

    class DummyListView:
        def __init__(self) -> None:
            self.index = None
            self.appended = []

        def clear(self) -> None:  # pragma: no cover - behaviour verified by append count
            self.appended.clear()

        def append(self, item) -> None:  # pragma: no cover - container accepts any object
            self.appended.append(item)

    dummy = DummyListView()

    def fake_query_one(self, selector: str, expected_type=None):
        assert selector == "#provider-list"
        return dummy

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    app._refresh_provider_list()

    assert dummy.index == 0
    assert dummy.appended

    app._states = []
    dummy.index = None
    dummy.appended.clear()

    app._refresh_provider_list()

    assert dummy.index is None


@pytest.mark.parametrize(
    ("percent", "expected_colour"),
    [(0.25, "green"), (0.8, "yellow"), (0.95, "red")],
)
def test_connection_usage_bar_colour_thresholds(percent: float, expected_colour: str) -> None:
    """ConnectionUsageBar should emit coloured markup according to usage."""

    from streamdeck_tui.app import ConnectionUsageBar

    bar = ConnectionUsageBar()
    active = int(round(percent * 100))
    bar.update_status(
        active_connections=active,
        max_connections=100,
        percent=percent,
        message="Peak"
    )

    markup = bar.last_markup
    assert f"[{expected_colour}]" in markup
    assert f"{active}/100" in markup
    assert "Peak" in markup


def test_connection_usage_bar_fallback_when_missing_data() -> None:
    """The usage bar should fall back to a helpful message when data is missing."""

    from streamdeck_tui.app import ConnectionUsageBar

    bar = ConnectionUsageBar()
    bar.update_status(
        active_connections=None,
        max_connections=None,
        percent=None,
        message="Status unavailable"
    )

    markup = bar.last_markup
    assert "Status unavailable" in markup
    assert "[dim]" in markup


def test_connection_strip_updates_with_status() -> None:
    """Connection strip should display current and maximum connections per provider."""

    from rich.console import Console

    from streamdeck_tui.app import ProviderConnectionStrip, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.providers import ConnectionStatus

    provider = ProviderConfig(name="Alpha", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    state.connection_status = ConnectionStatus(
        active_connections=3, max_connections=10, message="Operating normally"
    )
    state.channels = []

    async def run_app() -> str:
        async with app.run_test() as pilot:
            await pilot.pause()
            app._update_connection_strip()
            strip = app.query_one("#connection-strip", ProviderConnectionStrip)
            renderable = strip.render()
            console = Console(record=True)
            console.print(renderable)
            return console.export_text()

    rendered = asyncio.run(run_app())

    assert "3/10" in rendered
    assert "Operating normally" in rendered


def test_update_playing_info_handles_missing_widget(monkeypatch) -> None:
    """Updating the playing info should tolerate missing UI widgets."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig

    app = StreamdeckApp(AppConfig())

    def raise_missing(*_args, **_kwargs):  # pragma: no cover - behaviour asserted by lack of crash
        raise RuntimeError("missing widget")

    monkeypatch.setattr(StreamdeckApp, "query_one", raise_missing, raising=False)

    app._update_playing_info(None, None)
    app._clear_playing_info()


def test_load_provider_skips_recent_refresh(monkeypatch) -> None:
    """_load_provider should skip automatic reloads performed too soon."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import ChannelInfo, PlayingChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    dummy_logger = DummyLogger()
    monkeypatch.setattr(app_module, "log", dummy_logger)

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    state = app._states[0]
    state.last_loaded_at = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    state.channels = [Channel(name="Demo", url="http://example.com/stream")]

    statuses: list[str] = []

    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)
    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_clear_channels", lambda self, message: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_stop_worker", lambda self: None, raising=False)

    run_started = False

    def fake_run_worker(self, *args, **kwargs):  # pragma: no cover - behaviour checked by flag
        nonlocal run_started
        run_started = True
        return None

    monkeypatch.setattr(StreamdeckApp, "run_worker", fake_run_worker, raising=False)

    app._load_provider(0)

    assert not run_started
    assert any("Skipped automatic reload" in status for status in statuses)
    assert any(
        "Skipping automatic reload" in message for message in dummy_logger.messages
    )


def test_playlist_candidates_include_xtream_variants() -> None:
    """Xtream providers should expose HTTPS and legacy playlist fallbacks."""

    from streamdeck_tui.app import MAX_PLAYLIST_ATTEMPTS, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig, build_xtream_urls

    playlist, api = build_xtream_urls("http://portal.example.com", "user", "pass")
    provider = ProviderConfig(
        name="Xtream",
        playlist_url=playlist,
        api_url=api,
        xtream_base_url="http://portal.example.com",
        xtream_username="user",
        xtream_password="pass",
    )
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")
    state = app._states[0]

    candidates = app._playlist_candidates_for_state(state)

    assert candidates[0].playlist_url == playlist
    assert len(candidates) <= MAX_PLAYLIST_ATTEMPTS
    assert any(candidate.scheme == "https" for candidate in candidates[1:])
    assert any(candidate.type_param == "m3u" for candidate in candidates[1:])


def test_provider_label_includes_channels_and_status(monkeypatch) -> None:
    """Provider labels should show counts alongside all status details."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel
    from streamdeck_tui.providers import ConnectionStatus

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(
        StreamdeckApp,
        "_update_provider_progress_widget",
        lambda self, state=None: None,
        raising=False,
    )

    app._active_index = None
    state = app._states[0]
    channels = [
        Channel(name="Demo 1", url="http://example.com/1"),
        Channel(name="Demo 2", url="http://example.com/2"),
    ]
    app._handle_channels_loaded(state, channels)
    state.connection_status = ConnectionStatus(
        active_connections=3, max_connections=5, message="All good"
    )

    label = app._provider_label(state)

    assert "2 channels" in label
    assert "active 3" in label
    assert "max 5" in label
    assert "All good" in label

    state.loading = True
    state.loading_progress = 0.5
    loading_label = app._provider_label(state)
    assert "loading 50%" in loading_label
    assert loading_label.index("loading 50%") < loading_label.index("2 channels")


def test_vod_channels_filtered_when_disabled(monkeypatch) -> None:
    """Providers with VOD disabled should drop movie and series entries."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(
        name="Demo",
        playlist_url="http://example.com",
        enable_vod=False,
    )
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(
        StreamdeckApp,
        "_update_provider_progress_widget",
        lambda self, state=None: None,
        raising=False,
    )
    monkeypatch.setattr(
        StreamdeckApp,
        "_update_connection_usage_widget",
        lambda self, state=None: None,
        raising=False,
    )
    monkeypatch.setattr("streamdeck_tui.app.save_config", lambda config, path: None, raising=False)

    app._active_index = None
    state = app._states[0]
    channels = [
        Channel(name="Live", url="http://example.com/live/user/pass/1.ts", group="News"),
        Channel(
            name="Movie",
            url="http://example.com/movie/user/pass/2.mkv",
            group="English VOD",
        ),
        Channel(
            name="Series",
            url="http://example.com/series/user/pass/3.mkv",
            group="Series",
            raw_attributes={"tvg-type": "series"},
        ),
    ]

    app._handle_channels_loaded(state, channels)

    assert state.channels is not None
    assert [channel.name for channel in state.channels] == ["Live"]
    assert state.last_channel_count == 1


def test_provider_last_channel_count_persists_after_error(monkeypatch) -> None:
    """The last successful channel count should survive load failures."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")

    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(
        StreamdeckApp,
        "_update_provider_progress_widget",
        lambda self, state=None: None,
        raising=False,
    )

    app._active_index = None
    state = app._states[0]
    channels = [Channel(name="Demo", url="http://example.com/1")]
    app._handle_channels_loaded(state, channels)
    assert state.last_channel_count == 1

    app._handle_provider_error(state, "boom")

    assert state.last_channel_count == 1
    label = app._provider_label(state)
    assert "error: boom" in label
    assert "1 channels" in label


def test_action_reload_provider_requires_confirmation(monkeypatch) -> None:
    """Manual reloads within the guard window should prompt for confirmation."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import ChannelInfo, PlayingChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    dummy_logger = DummyLogger()
    monkeypatch.setattr(app_module, "log", dummy_logger)

    provider = ProviderConfig(
        name="Demo",
        playlist_url="http://example.com",
        api_url="http://example.com/status",
    )
    app = StreamdeckApp(AppConfig(providers=[provider]))
    app._active_index = 0
    state = app._states[0]
    state.last_loaded_at = datetime.now(tz=timezone.utc) - timedelta(minutes=30)

    load_calls: list[tuple[int, bool]] = []

    def fake_load(self, index: int, *, force: bool = False) -> None:
        load_calls.append((index, force))

    monkeypatch.setattr(StreamdeckApp, "_load_provider", fake_load, raising=False)

    statuses: list[str] = []
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)

    captured: dict[str, object] = {}

    def fake_push(self, screen, callback=None):  # pragma: no cover - trivial passthrough
        captured["screen"] = screen
        captured["callback"] = callback

    monkeypatch.setattr(StreamdeckApp, "push_screen", fake_push, raising=False)

    app.action_reload_provider()

    assert "screen" in captured
    assert load_calls == []
    assert any("Confirm reload" in status for status in statuses)
    assert any(
        "awaiting confirmation" in message for message in dummy_logger.messages
    )

    callback = captured["callback"]
    assert callable(callback)

    callback(False)

    assert load_calls == []
    assert any("Reload cancelled" in message for message in dummy_logger.messages)

    statuses.clear()
    dummy_logger.messages.clear()

    app.action_reload_provider()

    callback = captured["callback"]
    assert callable(callback)

    callback(True)

    assert load_calls[-1] == (0, True)
    assert any("Reload confirmed" in message for message in dummy_logger.messages)


def test_apply_filter_reuses_cached_search_index(monkeypatch) -> None:
    from streamdeck_tui.app import ChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel, build_search_index

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]

    channels = [
        Channel(name="ESPN 1080", url="http://example.com/espn"),
        Channel(name="News", url="http://example.com/news"),
    ]

    state.channels = channels
    state.search_index = build_search_index(channels)
    state.loading = False
    app._active_index = 0
    app._active_tab = "favorites"

    recorded_indices: list[object] = []

    def fake_filter(
        channels_arg,
        query_arg,
        search_index_arg=None,
        *,
        return_indices=False,
    ):
        recorded_indices.append(search_index_arg)
        if return_indices:
            return list(range(len(channels_arg)))
        return list(channels_arg)

    monkeypatch.setattr("streamdeck_tui.app.filter_channels", fake_filter)
    monkeypatch.setattr(StreamdeckApp, "_cancel_channel_render", lambda self: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_start_channel_render", lambda self, channels: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: None, raising=False)

    class DummyChannelInfo:
        channel = None
        provider = None

    def fake_query_one(self, selector, expected_type=None):
        if selector is ChannelInfo:
            return DummyChannelInfo()
        raise AssertionError(f"Unexpected selector {selector!r}")

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    app._apply_filter("ESPN 1080 / 1080 ESPN")
    app._apply_filter("espn")

    assert recorded_indices == [state.search_index, state.search_index]


def test_apply_filter_streams_channel_batches() -> None:
    """Rendering large channel lists should stream results in batches."""

    import types
    from contextlib import nullcontext

    from streamdeck_tui.app import ChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Bulk", playlist_url="http://example.com")
    channels = [
        Channel(name=f"Channel {index}", url=f"http://example.com/{index}")
        for index in range(6000)
    ]
    app = StreamdeckApp(AppConfig(providers=[provider]))

    class DummyListView:
        def __init__(self) -> None:
            self.items: list[object] = []
            self.index: int | None = None

        def clear(self) -> None:
            self.items.clear()

        def append(self, item: object) -> None:
            self.items.append(item)

        def batch_update(self):
            return nullcontext()

    class DummyChannelInfo:
        def __init__(self) -> None:
            self.channel: Channel | None = None
            self.provider: str | None = None

    list_view = DummyListView()
    info = DummyChannelInfo()

    def fake_query_one(self, selector, expected_type=None):
        if selector == "#channel-list":
            return list_view
        if selector in {ChannelInfo, info.__class__}:
            return info
        raise AssertionError(f"Unexpected selector: {selector!r}")

    def fake_call_later(self, callback, *args, **kwargs):
        asyncio.get_running_loop().call_soon(callback, *args, **kwargs)
        return True

    app.query_one = types.MethodType(fake_query_one, app)
    app.call_later = types.MethodType(fake_call_later, app)
    app.CHANNEL_RENDER_BATCH_DELAY = 0.01
    app._active_tab = "channels"

    async def run_app() -> None:
        state = app._states[0]
        state.channels = channels
        app._channel_render_generation += 1
        generation = app._channel_render_generation
        provider_name = state.config.name
        render_task = asyncio.create_task(
            app._render_channel_list(channels, generation, provider_name)
        )

        await asyncio.sleep(0.05)
        assert app._channel_first_batch_size == app.CHANNEL_RENDER_BATCH_SIZE
        assert app._channel_rendered_count >= app._channel_first_batch_size
        assert app._channel_rendered_count < len(channels)
        assert info.channel is channels[0]
        assert info.provider == provider_name
        assert not render_task.done()

        await asyncio.wait_for(render_task, timeout=5.0)
        assert app._channel_rendered_count == len(channels)

    asyncio.run(run_app())


def test_action_reload_provider_without_recent_load(monkeypatch) -> None:
    """Manual reloads after the guard window should proceed immediately."""

    from streamdeck_tui.app import ChannelInfo, PlayingChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(
        name="Demo",
        playlist_url="http://example.com",
        api_url="http://example.com/status",
    )
    app = StreamdeckApp(AppConfig(providers=[provider]))
    app._active_index = 0
    state = app._states[0]
    state.last_loaded_at = datetime.now(tz=timezone.utc) - timedelta(hours=7)

    load_calls: list[tuple[int, bool]] = []

    def fake_load(self, index: int, *, force: bool = False) -> None:
        load_calls.append((index, force))

    monkeypatch.setattr(StreamdeckApp, "_load_provider", fake_load, raising=False)

    def fake_push(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("push_screen should not be called for stale reloads")

    monkeypatch.setattr(StreamdeckApp, "push_screen", fake_push, raising=False)

    app.action_reload_provider()

    assert load_calls == [(0, True)]



def test_fetch_provider_success_on_app_thread(monkeypatch) -> None:
    """Directly running the worker coroutine should clear loading state."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel
    from streamdeck_tui.providers import ConnectionStatus

    provider = ProviderConfig(
        name="Demo",
        playlist_url="http://example.com",
        api_url="http://example.com/status",
    )
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    state.loading = True
    app._worker = object()

    monkeypatch.setattr(StreamdeckApp, "_apply_filter", lambda self, query: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)

    dummy_search = type("DummySearch", (), {"value": ""})()

    def fake_query_one(self, selector: str, expected_type=None):  # pragma: no cover - trivial stub
        if selector == "#search":
            return dummy_search
        raise AssertionError(f"Unexpected selector: {selector}")

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    channels = [Channel(name="Demo", url="http://example.com/stream")]

    monkeypatch.setattr(
        app_module,
        "load_playlist",
        lambda url, **_: list(channels),
        raising=False,
    )

    async def fake_status(_: str, **__: object) -> ConnectionStatus:
        return ConnectionStatus(message="OK")

    monkeypatch.setattr(app_module, "fetch_connection_status", fake_status, raising=False)

    asyncio.run(app._fetch_provider(state))

    assert state.loading is False
    assert state.last_error is None
    assert state.channels == channels
    assert state.connection_status is not None
    assert state.connection_status.message == "OK"
    assert state.last_loaded_at is not None
    assert app._worker is None


def test_fetch_provider_error_on_app_thread(monkeypatch) -> None:
    """Errors raised while loading should surface through the handler."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")
    state = app._states[0]
    state.loading = True
    app._worker = object()

    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_clear_channels", lambda self, message="": None, raising=False)

    def failing_loader(_: str, **__: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "load_playlist", failing_loader, raising=False)

    asyncio.run(app._fetch_provider(state))

    assert state.loading is False
    assert state.channels is None
    assert state.last_error == "boom"
    assert state.connection_status is None
    assert state.last_loaded_at is None
    assert app._worker is None


def test_toggle_favorite_updates_config(monkeypatch) -> None:
    """Toggling favorites should add and remove entries from the configuration."""

    from streamdeck_tui.app import ChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")
    state = app._states[0]
    channel = Channel(name="Demo Channel", url="http://example.com/stream")
    state.channels = [channel]
    app.filtered_channels = [channel]
    app._active_index = 0
    app._active_tab = "channels"

    statuses: list[str] = []
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)

    dummy_list = type("DummyList", (), {"index": 0})()
    dummy_info = type("DummyInfo", (), {"channel": None, "provider": None})()
    dummy_search = type("DummySearch", (), {"value": ""})()

    def fake_query_one(self, selector, expected_type=None):
        if selector == "#channel-list":
            return dummy_list
        if selector in {"ChannelInfo", ChannelInfo}:
            return dummy_info
        if selector == "#search":
            return dummy_search
        raise AssertionError(f"Unexpected selector: {selector}")

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    refresh_calls: list[bool] = []
    monkeypatch.setattr(StreamdeckApp, "_refresh_favorites_view", lambda self: refresh_calls.append(True), raising=False)

    saved: list[bool] = []

    def fake_save_config(config, path):  # pragma: no cover - trivial stub
        saved.append(True)

    monkeypatch.setattr("streamdeck_tui.app.save_config", fake_save_config, raising=False)

    app.action_toggle_favorite()

    assert len(app._config.favorites) == 1
    assert app._config.favorites[0].channel_url == channel.url
    assert saved
    assert refresh_calls
    assert any("Added" in status for status in statuses)

    saved.clear()
    refresh_calls.clear()
    app.action_toggle_favorite()

    assert app._config.favorites == []
    assert saved
    assert refresh_calls
    assert any("Removed" in status for status in statuses)


def test_action_play_channel_handles_launch_failure(monkeypatch) -> None:
    """Failures launching the player should update the status and not track a process."""

    from streamdeck_tui.app import ChannelInfo, PlayingChannelInfo, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]), preferred_player="mpv")
    state = app._states[0]
    channel = Channel(name="Demo Channel", url="http://example.com/stream")
    state.channels = [channel]
    app.filtered_channels = [channel]
    app._active_index = 0
    app._active_tab = "channels"

    statuses: list[str] = []
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)

    dummy_list = type("DummyList", (), {"index": 0})()
    dummy_info = type("DummyInfo", (), {"channel": None, "provider": None})()
    dummy_playing = type("DummyPlaying", (), {"channel": None, "provider": None})()
    dummy_search = type("DummySearch", (), {"value": ""})()

    def fake_query_one(self, selector, expected_type=None):
        if selector == "#channel-list":
            return dummy_list
        if selector in {"ChannelInfo", ChannelInfo}:
            return dummy_info
        if selector in {"PlayingChannelInfo", PlayingChannelInfo}:
            return dummy_playing
        if selector == "#search":
            return dummy_search
        raise AssertionError(f"Unexpected selector: {selector}")

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    captured_preferred: list[Optional[str]] = []

    async def failing_launch(_: Channel, *, preferred: Optional[str] = None):  # pragma: no cover - trivial async stub
        captured_preferred.append(preferred)
        raise RuntimeError("player failed")

    monkeypatch.setattr("streamdeck_tui.app.launch_player", failing_launch, raising=False)

    app.action_play_channel()

    assert app._player_process is None
    assert captured_preferred == ["mpv"]
    assert any("Failed to launch player" in status for status in statuses)


def test_channel_list_shows_provider_names() -> None:
    from textual.widgets import ListView

    from streamdeck_tui.app import ChannelListItem, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel, build_search_index
    from streamdeck_tui.providers import ConnectionStatus

    provider = ProviderConfig(name="Demo Provider", playlist_url="http://example.com")
    channel = Channel(name="Sample", url="http://example.com/sample", group="News")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    state.channels = [channel]
    state.search_index = build_search_index(state.channels)
    state.connection_status = ConnectionStatus(active_connections=2, max_connections=5)

    expected_color = app._provider_color(provider.name)

    async def run_app() -> "Text":
        async with app.run_test() as pilot:
            app._set_active_tab("channels")
            app._rebuild_all_channels()
            app._apply_filter("")
            await pilot.pause()
            list_view = app.query_one("#channel-list", ListView)
            item = next(
                child for child in list_view.children if isinstance(child, ChannelListItem)
            )
            return item._label.render()

    rendered = asyncio.run(run_app())
    text = getattr(rendered, "plain", str(rendered))
    assert "Demo Provider" in text
    assert "Sample" in text
    assert "News" in text
    assert "Connections: 2/5" in text
    spans = getattr(rendered, "spans", ())
    assert any(span.style == expected_color for span in spans)


def test_stop_all_playback_stops_everything(monkeypatch) -> None:
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig

    app = StreamdeckApp(AppConfig())
    active_key = ("Provider", "http://example.com/stream")
    app._player_handles[active_key] = object()

    stopped: list[tuple[tuple[str, str] | None, bool]] = []

    def fake_stop(
        self: StreamdeckApp,
        key=None,
        *,
        user_requested: bool = False,
        suppress_status: bool = False,
    ) -> None:
        stopped.append((key, user_requested))

    monkeypatch.setattr(StreamdeckApp, "_stop_player_process", fake_stop, raising=False)

    app.action_stop_all_playback()

    assert stopped == [(active_key, True)]


def test_refresh_playing_panel_sorts_by_average_bitrate(monkeypatch) -> None:
    from streamdeck_tui.app import PlayingChannelsPanel, StreamdeckApp
    from streamdeck_tui.config import AppConfig
    from streamdeck_tui.playlist import Channel
    from streamdeck_tui.stats import StreamStats

    app = StreamdeckApp(AppConfig())

    fast_channel = Channel(name="Fast", url="http://example.com/fast")
    slow_channel = Channel(name="Slow", url="http://example.com/slow")
    fast_key = ("Provider B", fast_channel.url)
    slow_key = ("Provider A", slow_channel.url)

    app._playing_channels[fast_key] = ("Provider B", fast_channel)
    app._playing_channels[slow_key] = ("Provider A", slow_channel)
    app._latest_stats[fast_key] = StreamStats(
        live_bitrate=2_500_000, average_bitrate=4_000_000
    )
    app._latest_stats[slow_key] = StreamStats(
        live_bitrate=1_500_000, average_bitrate=2_000_000
    )
    app._playback_started_at[fast_key] = datetime(2024, 1, 2, tzinfo=timezone.utc)
    app._playback_started_at[slow_key] = datetime(2024, 1, 1, tzinfo=timezone.utc)

    panel = DummyPlayingPanel()

    def fake_query_optional_widget(self, query, widget_type=None):
        if query is PlayingChannelsPanel:
            return panel
        return None

    monkeypatch.setattr(
        StreamdeckApp,
        "_query_optional_widget",
        fake_query_optional_widget,
        raising=False,
    )

    refreshed: list[bool] = []

    monkeypatch.setattr(
        StreamdeckApp,
        "_refresh_selected_channels_panel",
        lambda self: refreshed.append(True),
        raising=False,
    )

    app._refresh_playing_channels_panel()

    assert panel.keys == [fast_key, slow_key]
    assert panel.entries is not None
    assert any("[bright_green]Live" in entry for entry in panel.entries)
    assert any("[bright_yellow]Avg" in entry for entry in panel.entries)
    assert refreshed


def test_action_stop_channel_targets_selected_playing_entry(monkeypatch) -> None:
    from streamdeck_tui.app import PlayingChannelsPanel, StreamdeckApp
    from streamdeck_tui.config import AppConfig
    from streamdeck_tui.playlist import Channel

    app = StreamdeckApp(AppConfig())
    channel = Channel(name="Demo", url="http://example.com/demo")
    key = ("Provider", channel.url)
    app._playing_channels[key] = ("Provider", channel)
    app._player_handles[key] = object()

    panel = DummyPlayingPanel(selected_key=key)

    def fake_query_optional_widget(self, query, widget_type=None):
        if query is PlayingChannelsPanel:
            return panel
        return None

    monkeypatch.setattr(
        StreamdeckApp,
        "_query_optional_widget",
        fake_query_optional_widget,
        raising=False,
    )

    focused_widget = SimpleNamespace(id="playing-channels-list")
    monkeypatch.setattr(
        StreamdeckApp,
        "focused",
        property(lambda self: focused_widget),
        raising=False,
    )

    stopped: list[list[tuple[str, str]]] = []

    def capture_stop(self, keys):
        stopped.append(list(keys))

    monkeypatch.setattr(
        StreamdeckApp, "_stop_channels", capture_stop, raising=False
    )

    app.action_stop_channel()

    assert stopped == [[key]]


def test_action_stop_channel_handles_child_focus(monkeypatch) -> None:
    from streamdeck_tui.app import PlayingChannelsPanel, StreamdeckApp
    from streamdeck_tui.config import AppConfig
    from streamdeck_tui.playlist import Channel

    app = StreamdeckApp(AppConfig())
    channel = Channel(name="Demo", url="http://example.com/demo")
    key = ("Provider", channel.url)
    app._playing_channels[key] = ("Provider", channel)
    app._player_handles[key] = object()

    panel = DummyPlayingPanel(selected_key=key)

    def fake_query_optional_widget(self, query, widget_type=None):
        if query is PlayingChannelsPanel:
            return panel
        return None

    monkeypatch.setattr(
        StreamdeckApp,
        "_query_optional_widget",
        fake_query_optional_widget,
        raising=False,
    )

    parent = SimpleNamespace(id="playing-channels-list", parent=None)
    focused_widget = SimpleNamespace(id=None, parent=parent)
    monkeypatch.setattr(
        StreamdeckApp,
        "focused",
        property(lambda self: focused_widget),
        raising=False,
    )

    stopped: list[list[tuple[str, str]]] = []

    def capture_stop(self, keys):
        stopped.append(list(keys))

    monkeypatch.setattr(
        StreamdeckApp, "_stop_channels", capture_stop, raising=False
    )

    app.action_stop_channel()

    assert stopped == [[key]]


def test_search_down_arrow_focuses_channel_list() -> None:
    from textual.widgets import Input, ListView

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel, build_search_index

    provider = ProviderConfig(name="Demo Provider", playlist_url="http://example.com")
    channel = Channel(name="Sample", url="http://example.com/sample")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    state.channels = [channel]
    state.search_index = build_search_index(state.channels)

    async def run_app() -> bool:
        async with app.run_test() as pilot:
            app._set_active_tab("channels")
            app._rebuild_all_channels()
            app._apply_filter("")
            await pilot.pause()
            search = app.query_one("#search", Input)
            app.action_focus_search()
            await pilot.pause()
            assert search.has_focus
            await pilot.press("down")
            await pilot.pause()
            list_view = app.query_one("#channel-list", ListView)
            return list_view.has_focus

    has_focus = asyncio.run(run_app())
    assert has_focus


def test_tab_specific_bindings_expose_scope_metadata() -> None:
    """Bindings restricted to tabs should report their scope and stay hidden by default."""

    from streamdeck_tui.app import StreamdeckApp

    binding = next(b for b in StreamdeckApp.BINDINGS if b.action == "new_provider")

    assert not binding.show
    assert StreamdeckApp._binding_tab_scope(binding) == frozenset({"providers"})


def test_footer_updates_with_active_tab_bindings() -> None:
    """Switching tabs should refresh the footer with the relevant shortcuts."""

    from textual.widgets._footer import FooterKey

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    app._states[0].channels = []

    async def run_app() -> None:
        async with app.run_test() as pilot:
            footer = app.query_one("#app-footer")

            await pilot.pause()

            app.action_switch_tab("channels")
            await pilot.pause()

            def footer_actions() -> set[str]:
                return {widget.action for widget in footer.query(FooterKey)}

            assert "focus_search" in footer_actions()
            assert "new_provider" not in footer_actions()

            app.action_switch_tab("providers")
            await pilot.pause()

            assert "new_provider" in footer_actions()
            assert "focus_search" not in footer_actions()

    asyncio.run(run_app())
