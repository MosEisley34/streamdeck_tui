"""Resource configuration tests for streamdeck_tui."""

import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)

import importlib.resources as resources

from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_app_uses_packaged_stylesheet() -> None:
    """The application should resolve the packaged stylesheet at runtime."""

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert css_resource.is_file(), "Expected stylesheet to be packaged with the app"
    css_text = css_resource.read_text(encoding="utf-8").strip()

    assert css_text.startswith("#layout")
    assert StreamdeckApp.CSS.strip() == css_text
    assert StreamdeckApp.CSS_PATH == []

    app = StreamdeckApp(AppConfig())
    try:
        assert app.stylesheet is not None
    finally:
        app.exit(result=None)
