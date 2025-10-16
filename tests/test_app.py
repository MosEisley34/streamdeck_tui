"""Unit tests for StreamdeckApp helpers."""

import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone

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


def test_tabbed_layout_and_log_viewer() -> None:
    """The main application should expose the tabbed UI and surface log messages."""

    from textual.widgets import TabPane, TabbedContent

    from streamdeck_tui.app import LogViewer, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.logging_utils import get_logger

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))

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


def test_refresh_provider_list_handles_none_previous_index(monkeypatch) -> None:
    """Refreshing the provider list should not compare None indexes."""

    from streamdeck_tui.app import ProviderState, StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
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


def test_load_provider_skips_recent_refresh(monkeypatch) -> None:
    """_load_provider should skip automatic reloads performed too soon."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    dummy_logger = DummyLogger()
    monkeypatch.setattr(app_module, "log", dummy_logger)

    provider = ProviderConfig(name="Test", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))

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


def test_action_reload_provider_requires_confirmation(monkeypatch) -> None:
    """Manual reloads within the guard window should prompt for confirmation."""

    from streamdeck_tui import app as app_module
    from streamdeck_tui.app import StreamdeckApp
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

    async def run_app() -> None:
        state = app._states[0]
        state.channels = channels
        app._channel_render_generation += 1
        generation = app._channel_render_generation
        render_task = asyncio.create_task(
            app._render_channel_list(channels, generation)
        )

        await asyncio.sleep(0.05)
        assert app._channel_first_batch_size == app.CHANNEL_RENDER_BATCH_SIZE
        assert app._channel_rendered_count >= app._channel_first_batch_size
        assert app._channel_rendered_count < len(channels)
        assert info.channel is channels[0]
        assert not render_task.done()

        await asyncio.wait_for(render_task, timeout=5.0)
        assert app._channel_rendered_count == len(channels)

    asyncio.run(run_app())


def test_action_reload_provider_without_recent_load(monkeypatch) -> None:
    """Manual reloads after the guard window should proceed immediately."""

    from streamdeck_tui.app import StreamdeckApp
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

    monkeypatch.setattr(app_module, "load_playlist", lambda url: list(channels), raising=False)

    async def fake_status(_: str) -> ConnectionStatus:
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
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    state.loading = True
    app._worker = object()

    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_refresh_provider_list", lambda self: None, raising=False)
    monkeypatch.setattr(StreamdeckApp, "_clear_channels", lambda self, message="": None, raising=False)

    def failing_loader(_: str) -> None:
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

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    channel = Channel(name="Demo Channel", url="http://example.com/stream")
    state.channels = [channel]
    app.filtered_channels = [channel]
    app._active_index = 0
    app._active_tab = "channels"

    statuses: list[str] = []
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)

    dummy_list = type("DummyList", (), {"index": 0})()
    dummy_info = type("DummyInfo", (), {"channel": None})()
    dummy_search = type("DummySearch", (), {"value": ""})()

    def fake_query_one(self, selector: str, expected_type=None):
        if selector == "#channel-list":
            return dummy_list
        if selector == "ChannelInfo":
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

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig
    from streamdeck_tui.playlist import Channel

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
    app = StreamdeckApp(AppConfig(providers=[provider]))
    state = app._states[0]
    channel = Channel(name="Demo Channel", url="http://example.com/stream")
    state.channels = [channel]
    app.filtered_channels = [channel]
    app._active_index = 0
    app._active_tab = "channels"

    statuses: list[str] = []
    monkeypatch.setattr(StreamdeckApp, "_set_status", lambda self, message: statuses.append(message), raising=False)

    dummy_list = type("DummyList", (), {"index": 0})()
    dummy_info = type("DummyInfo", (), {"channel": None})()
    dummy_search = type("DummySearch", (), {"value": ""})()

    def fake_query_one(self, selector: str, expected_type=None):
        if selector == "#channel-list":
            return dummy_list
        if selector == "ChannelInfo":
            return dummy_info
        if selector == "#search":
            return dummy_search
        raise AssertionError(f"Unexpected selector: {selector}")

    monkeypatch.setattr(StreamdeckApp, "query_one", fake_query_one, raising=False)

    async def failing_launch(_: Channel):  # pragma: no cover - trivial async stub
        raise RuntimeError("player failed")

    monkeypatch.setattr("streamdeck_tui.app.launch_player", failing_launch, raising=False)

    app.action_play_channel()

    assert app._player_process is None
    assert any("Failed to launch player" in status for status in statuses)
