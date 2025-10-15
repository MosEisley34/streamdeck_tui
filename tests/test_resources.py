"""Resource configuration tests for streamdeck_tui."""

import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)

import importlib.resources as resources

from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_app_loads_packaged_stylesheet() -> None:
    """The application should resolve the packaged stylesheet at runtime."""

    assert StreamdeckApp.CSS.strip().startswith("#layout")
    assert StreamdeckApp.CSS_PATH == []
    assert not any(
        resource.name == "streamdeck.css"
        for resource in resources.files("streamdeck_tui").iterdir()
    ), "No external stylesheet should be packaged"
    app = StreamdeckApp(AppConfig())
    try:
        assert app.stylesheet is not None
    finally:
        app.exit(result=None)
