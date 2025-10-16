"""Unit tests for StreamdeckApp helpers."""

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

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
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


def test_action_reload_provider_without_recent_load(monkeypatch) -> None:
    """Manual reloads after the guard window should proceed immediately."""

    from streamdeck_tui.app import StreamdeckApp
    from streamdeck_tui.config import AppConfig, ProviderConfig

    provider = ProviderConfig(name="Demo", playlist_url="http://example.com")
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
