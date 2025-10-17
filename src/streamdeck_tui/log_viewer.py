"""Textual widget displaying log output within the application."""

from __future__ import annotations

from collections import deque
import logging
from logging import LogRecord
from typing import Deque, Iterable, Optional, Tuple

from textual.message import Message
from textual.widgets import Static

from .logging_utils import register_log_viewer


class LogViewer(Static):
    """Simple log output widget that keeps a rolling buffer of messages."""

    def __init__(
        self,
        *,
        max_lines: int = 500,
        id: Optional[str] = None,
    ) -> None:
        super().__init__("", id=id, markup=False)
        self._messages: Deque[str] = deque(maxlen=max_lines)
        self._formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._refresh_view()

    def on_mount(self) -> None:  # pragma: no cover - requires UI integration
        register_log_viewer(self)

    def on_unmount(self) -> None:  # pragma: no cover - defensive cleanup
        register_log_viewer(None)

    @property
    def lines(self) -> Tuple[str, ...]:
        """Return the currently buffered log lines."""

        return tuple(self._messages)

    def get_messages(self) -> Tuple[str, ...]:
        """Alias for :attr:`lines` used by the application code."""

        return self.lines

    def clear(self) -> None:
        """Clear all captured log lines."""

        self._messages.clear()
        self._refresh_view()

    def append_message(self, message: str) -> None:
        """Append a single log *message* to the buffer and refresh display."""

        self._messages.append(message)
        self._refresh_view()

    def replace_messages(self, messages: Iterable[str]) -> None:
        """Replace the current buffer with ``messages`` and refresh display."""

        self._messages.clear()
        for message in messages:
            self._messages.append(message)
        self._refresh_view()

    def _render_messages(self) -> str:
        """Return a textual representation of the buffered log messages."""

        if not self._messages:
            return "No log messages yet."
        return "\n".join(self._messages)

    def post_message(
        self,
        message: Message,
        formatted_message: Optional[str] = None,
    ) -> bool:
        """Append a log record to the widget or delegate to ``Static``.

        Parameters
        ----------
        message:
            The message being delivered. When this is a
            :class:`logging.LogRecord`, it will be rendered in the viewer;
            otherwise the call is delegated to :class:`~textual.widgets.Static`.
        formatted_message:
            An optional pre-formatted message. When omitted, the viewer will
            render the record using its internal formatter.
        """

        if isinstance(message, LogRecord):
            rendered = formatted_message or self._formatter.format(message)
            self.append_message(rendered)
            return True
        return super().post_message(message)

    def _refresh_view(self) -> None:
        """Update the rendered text and scroll to the most recent entry."""

        self.update(self._render_messages())
        self.call_after_refresh(self._scroll_to_end)

    def _scroll_to_end(self) -> None:
        """Scroll the widget to the bottom without animation."""

        try:
            self.scroll_end(animate=False)
        except Exception:  # pragma: no cover - scrolling may fail during shutdown
            pass
