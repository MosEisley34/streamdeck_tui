"""Unit tests for StreamdeckApp helpers."""

import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)


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
