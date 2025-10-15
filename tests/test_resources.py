"""Resource configuration tests for streamdeck_tui."""

import importlib.util
import importlib.resources as resources

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)


def test_app_uses_inline_stylesheet() -> None:
    """The application should rely solely on the inline stylesheet."""

    import streamdeck_tui.app as app_module
    from streamdeck_tui.config import AppConfig

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert not css_resource.is_file(), "Stylesheet resource should no longer be packaged"

    assert app_module.StreamdeckApp.CSS.strip() == app_module._INLINE_DEFAULT_CSS.strip()
    assert app_module.StreamdeckApp.CSS_PATH == []

    app = app_module.StreamdeckApp(AppConfig())
    try:
        assert app.stylesheet is not None
    finally:
        app.exit(result=None)
