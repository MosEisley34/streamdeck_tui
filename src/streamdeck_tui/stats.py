"""Utilities for tracking stream bitrate and resolution statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class StreamStats:
    """Snapshot of bitrate and resolution information for a stream."""

    live_bitrate: Optional[float] = None
    average_bitrate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None


class StreamStatsAccumulator:
    """Aggregate incoming bitrate and resolution samples."""

    __slots__ = ("_live", "_total", "_samples", "_width", "_height")

    def __init__(self) -> None:
        self._live: Optional[float] = None
        self._total: float = 0.0
        self._samples: int = 0
        self._width: Optional[int] = None
        self._height: Optional[int] = None

    def push_bitrate(self, value: float) -> StreamStats:
        """Record a new instantaneous bitrate sample and return a snapshot."""

        if value <= 0:
            return self.snapshot()
        self._live = value
        self._samples += 1
        self._total += value
        return self.snapshot()

    def set_resolution(self, width: Optional[int], height: Optional[int]) -> StreamStats:
        """Update the last known resolution and return a snapshot."""

        if isinstance(width, int) and width > 0:
            self._width = width
        if isinstance(height, int) and height > 0:
            self._height = height
        return self.snapshot()

    def reset(self) -> None:
        """Clear recorded metrics."""

        self._live = None
        self._total = 0.0
        self._samples = 0
        self._width = None
        self._height = None

    def snapshot(self) -> StreamStats:
        """Return the current statistics snapshot."""

        average = None
        if self._samples:
            average = self._total / self._samples
        return StreamStats(
            live_bitrate=self._live,
            average_bitrate=average,
            width=self._width,
            height=self._height,
        )


__all__ = ["StreamStats", "StreamStatsAccumulator"]
