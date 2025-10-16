"""Textual widget displaying log output within the application."""

from __future__ import annotations

import logging
from logging import LogRecord
from typing import List, Optional, Tuple

from textual.message import Message
from textual.widgets import Static


class LogViewer(Static):
    """Simple log output widget that keeps a rolling buffer of messages."""

    def __init__(
        self,
        *,
        max_lines: int = 500,
        id: Optional[str] = None,
    ) -> None:
        super().__init__("", id=id, markup=False)
        self._max_lines = max_lines
        self._lines: List[str] = []
        self._formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    @property
    def lines(self) -> Tuple[str, ...]:
        """Return the currently buffered log lines."""

        return tuple(self._lines)

    def clear(self) -> None:
        """Clear all captured log lines."""

        self._lines.clear()
        self.update("")

    def _append(self, message: str) -> None:
        self._lines.append(message)
        if self._max_lines and len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines :]
        self.update("\n".join(self._lines))

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
            self._append(rendered)
            return True
        return super().post_message(message)
