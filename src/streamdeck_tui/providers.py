"""Provider utilities for Streamdeck TUI."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional
from urllib import error, request


@dataclass(slots=True)
class ConnectionStatus:
    """Status information returned from a provider API."""

    active_connections: Optional[int] = None
    max_connections: Optional[int] = None
    message: Optional[str] = None

    def as_label(self) -> str:
        """Return a human readable label."""

        if self.active_connections is not None and self.max_connections is not None:
            return f"{self.active_connections}/{self.max_connections} connections"
        if self.active_connections is not None:
            return f"{self.active_connections} connections"
        return self.message or "No status"


def _fetch_status(url: str, timeout: float) -> dict[str, object]:
    with request.urlopen(url, timeout=timeout) as response:  # type: ignore[call-arg]
        payload = response.read().decode("utf8")
    return json.loads(payload)


async def fetch_connection_status(url: str, *, timeout: float = 10.0) -> ConnectionStatus:
    """Fetch connection status JSON from ``url`` asynchronously."""

    try:
        payload = await asyncio.to_thread(_fetch_status, url, timeout)
    except (error.URLError, json.JSONDecodeError) as exc:  # pragma: no cover - network errors
        raise RuntimeError(str(exc)) from exc
    status = ConnectionStatus()
    if isinstance(payload, dict):
        active = payload.get("active_connections")
        maximum = payload.get("max_connections")
        if isinstance(active, int):
            status.active_connections = active
        if isinstance(maximum, int):
            status.max_connections = maximum
        message = payload.get("message")
        if isinstance(message, str):
            status.message = message
    return status


__all__ = ["ConnectionStatus", "fetch_connection_status"]
