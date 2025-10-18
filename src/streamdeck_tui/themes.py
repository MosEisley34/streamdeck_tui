"""Custom theme definitions for Streamdeck TUI."""
from __future__ import annotations

from typing import Mapping

from textual.theme import Theme

__all__ = [
    "CUSTOM_THEMES",
    "DEFAULT_THEME_NAME",
]

# Solarized palette reference values.
_SOLARIZED_BASE03 = "#002b36"
_SOLARIZED_BASE02 = "#073642"
_SOLARIZED_BASE01 = "#586e75"
_SOLARIZED_BASE00 = "#657b83"
_SOLARIZED_BASE0 = "#839496"
_SOLARIZED_BASE1 = "#93a1a1"
_SOLARIZED_BASE2 = "#eee8d5"
_SOLARIZED_BASE3 = "#fdf6e3"
_SOLARIZED_YELLOW = "#b58900"
_SOLARIZED_ORANGE = "#cb4b16"
_SOLARIZED_RED = "#dc322f"
_SOLARIZED_MAGENTA = "#d33682"
_SOLARIZED_VIOLET = "#6c71c4"
_SOLARIZED_BLUE = "#268bd2"
_SOLARIZED_CYAN = "#2aa198"
_SOLARIZED_GREEN = "#859900"

_SOLARIZED_DARK = Theme(
    "solarized-dark",
    primary=_SOLARIZED_BLUE,
    secondary=_SOLARIZED_CYAN,
    warning=_SOLARIZED_YELLOW,
    error=_SOLARIZED_RED,
    success=_SOLARIZED_GREEN,
    accent=_SOLARIZED_MAGENTA,
    foreground=_SOLARIZED_BASE1,
    background=_SOLARIZED_BASE03,
    surface=_SOLARIZED_BASE02,
    panel=_SOLARIZED_BASE02,
    boost=_SOLARIZED_BASE2,
    dark=True,
)

_SOLARIZED_LIGHT = Theme(
    "solarized-light",
    primary=_SOLARIZED_BLUE,
    secondary=_SOLARIZED_VIOLET,
    warning=_SOLARIZED_YELLOW,
    error=_SOLARIZED_RED,
    success=_SOLARIZED_GREEN,
    accent=_SOLARIZED_ORANGE,
    foreground=_SOLARIZED_BASE00,
    background=_SOLARIZED_BASE3,
    surface=_SOLARIZED_BASE2,
    panel=_SOLARIZED_BASE2,
    boost=_SOLARIZED_BASE01,
    dark=False,
)

CUSTOM_THEMES: Mapping[str, Theme] = {
    _SOLARIZED_DARK.name: _SOLARIZED_DARK,
    _SOLARIZED_LIGHT.name: _SOLARIZED_LIGHT,
}
"""Themes bundled with the application keyed by their names."""

DEFAULT_THEME_NAME = _SOLARIZED_DARK.name
"""Default theme to apply when none is specified explicitly."""
