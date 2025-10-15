"""Resource configuration tests for streamdeck_tui."""

import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)

from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_app_loads_packaged_stylesheet() -> None:
    """The application should resolve the packaged stylesheet at runtime."""

    assert StreamdeckApp.CSS == ""
    assert StreamdeckApp.CSS_PATH == "streamdeck.css"
    app = StreamdeckApp(AppConfig())
    try:
        css_paths = list(app.css_path)
        assert css_paths, "CSS path list should not be empty"
        assert any(path.name == "streamdeck.css" for path in css_paths)
    finally:
        app.exit(result=None)
