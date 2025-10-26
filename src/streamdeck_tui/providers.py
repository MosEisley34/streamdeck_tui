"""Provider utilities for Streamdeck TUI."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Optional
from urllib import error, request

from .logging_utils import get_logger

log = get_logger(__name__)


def _coerce_connection_count(value: object) -> Optional[int]:
    """Convert ``value`` into an integer connection count if possible."""

    if isinstance(value, bool):  # Guard against ``True``/``False`` being treated as 1/0.
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                numeric = float(text)
            except ValueError:
                return None
            if numeric.is_integer():
                return int(numeric)
    return None


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


def _fetch_status(
    url: str, timeout: float, *, user_agent: Optional[str] = None
) -> dict[str, object]:
    log.debug("Fetching provider status from %s (timeout=%s)", url, timeout)
    req = request.Request(url)
    if user_agent:
        req.add_header("User-Agent", user_agent)
    with request.urlopen(req, timeout=timeout) as response:  # type: ignore[call-arg]
        payload = response.read().decode("utf8")
    log.debug("Received status payload: %s", payload)
    return json.loads(payload)


async def fetch_connection_status(
    url: str, *, timeout: float = 10.0, user_agent: Optional[str] = None
) -> ConnectionStatus:
    """Fetch connection status JSON from ``url`` asynchronously."""

    log.info("Requesting connection status from %s", url)
    try:
        payload = await asyncio.to_thread(
            _fetch_status, url, timeout, user_agent=user_agent
        )
    except (error.URLError, json.JSONDecodeError) as exc:  # pragma: no cover - network errors
        log.error("Failed to fetch status from %s: %s", url, exc)
        raise RuntimeError(str(exc)) from exc
    status = ConnectionStatus()
    if isinstance(payload, dict):
        def lookup_counts(container: dict[str, object], keys: tuple[str, ...]) -> Optional[int]:
            for key in keys:
                value = container.get(key)
                if value is None:
                    continue
                coerced = _coerce_connection_count(value)
                if coerced is not None:
                    return coerced
            return None

        active_keys = ("active_connections", "active_cons")
        maximum_keys = ("max_connections", "max_cons")

        coerced_active = lookup_counts(payload, active_keys)
        coerced_maximum = lookup_counts(payload, maximum_keys)

        if coerced_active is None or coerced_maximum is None:
            nested_sections = ("user_info",)
            for section in nested_sections:
                nested = payload.get(section)
                if not isinstance(nested, dict):
                    continue
                if coerced_active is None:
                    coerced_active = lookup_counts(nested, active_keys)
                if coerced_maximum is None:
                    coerced_maximum = lookup_counts(nested, maximum_keys)
                if coerced_active is not None and coerced_maximum is not None:
                    break
        if coerced_active is not None:
            status.active_connections = coerced_active
        if coerced_maximum is not None:
            status.max_connections = coerced_maximum
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
