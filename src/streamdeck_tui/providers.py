"""Provider utilities for Streamdeck TUI."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional
from urllib import error, request

from .logging_utils import get_logger

log = get_logger(__name__)


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
    log.debug("Fetching provider status from %s (timeout=%s)", url, timeout)
    with request.urlopen(url, timeout=timeout) as response:  # type: ignore[call-arg]
        payload = response.read().decode("utf8")
    log.debug("Received status payload: %s", payload)
    return json.loads(payload)


async def fetch_connection_status(url: str, *, timeout: float = 10.0) -> ConnectionStatus:
    """Fetch connection status JSON from ``url`` asynchronously."""

    log.info("Requesting connection status from %s", url)
    try:
        payload = await asyncio.to_thread(_fetch_status, url, timeout)
    except (error.URLError, json.JSONDecodeError) as exc:  # pragma: no cover - network errors
        log.error("Failed to fetch status from %s: %s", url, exc)
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
    log.info(
        "Parsed status from %s: active=%s max=%s message=%s",
        url,
        status.active_connections,
        status.max_connections,
        status.message,
    )
    return status


__all__ = ["ConnectionStatus", "fetch_connection_status"]
