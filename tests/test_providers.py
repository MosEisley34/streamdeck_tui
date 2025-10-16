import asyncio

from streamdeck_tui.providers import ConnectionStatus, fetch_connection_status


def test_fetch_connection_status(monkeypatch):
    def fake_fetch(url: str, timeout: float):
        assert url == "https://example.com/status"
        assert timeout == 10.0
        return {"active_connections": 3, "max_connections": 10, "message": "ok"}

    monkeypatch.setattr("streamdeck_tui.providers._fetch_status", fake_fetch)

    status = asyncio.run(fetch_connection_status("https://example.com/status"))
    assert status.active_connections == 3
    assert status.max_connections == 10
    assert status.as_label() == "3/10 connections"


def test_connection_status_label_fallback():
    status = ConnectionStatus(message="All good")
    assert status.as_label() == "All good"
