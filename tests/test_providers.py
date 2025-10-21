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


def test_fetch_connection_status_string_counts(monkeypatch):
    def fake_fetch(url: str, timeout: float):
        assert url == "https://example.com/status"
        assert timeout == 10.0
        return {
            "active_connections": "5",
            "max_connections": "20",
            "message": "string values",
        }

    monkeypatch.setattr("streamdeck_tui.providers._fetch_status", fake_fetch)

    status = asyncio.run(fetch_connection_status("https://example.com/status"))
    assert status.active_connections == 5
    assert status.max_connections == 20
    assert status.as_label() == "5/20 connections"


def test_fetch_connection_status_active_cons(monkeypatch):
    def fake_fetch(url: str, timeout: float):
        assert url == "https://example.com/status"
        assert timeout == 10.0
        return {"active_cons": 7, "max_connections": 9}

    monkeypatch.setattr("streamdeck_tui.providers._fetch_status", fake_fetch)

    status = asyncio.run(fetch_connection_status("https://example.com/status"))
    assert status.active_connections == 7
    assert status.max_connections == 9
    assert status.as_label() == "7/9 connections"


def test_fetch_connection_status_max_cons(monkeypatch):
    def fake_fetch(url: str, timeout: float):
        assert url == "https://example.com/status"
        assert timeout == 10.0
        return {"active_connections": 11, "max_cons": 42}

    monkeypatch.setattr("streamdeck_tui.providers._fetch_status", fake_fetch)

    status = asyncio.run(fetch_connection_status("https://example.com/status"))
    assert status.active_connections == 11
    assert status.max_connections == 42
    assert status.as_label() == "11/42 connections"


def test_fetch_connection_status_nested_user_info(monkeypatch):
    def fake_fetch(url: str, timeout: float):
        assert url == "https://example.com/status"
        assert timeout == 10.0
        return {"user_info": {"active_cons": 4, "max_connections": "8"}}

    monkeypatch.setattr("streamdeck_tui.providers._fetch_status", fake_fetch)

    status = asyncio.run(fetch_connection_status("https://example.com/status"))
    assert status.active_connections == 4
    assert status.max_connections == 8
    assert status.as_label() == "4/8 connections"
