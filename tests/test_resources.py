"""Resource configuration tests for streamdeck_tui."""

import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)

from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_app_loads_packaged_stylesheet() -> None:
    """The application should resolve the packaged stylesheet at runtime."""

    assert StreamdeckApp.CSS.strip().startswith("#layout")
    assert StreamdeckApp.CSS_PATH == []
    app = StreamdeckApp(AppConfig())
    try:
        assert app.stylesheet is not None
    finally:
        app.exit(result=None)
