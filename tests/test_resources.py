"""Resource configuration tests for streamdeck_tui."""

import importlib
import importlib.resources as resources
import importlib.util

import pytest

if importlib.util.find_spec("textual") is None:  # pragma: no cover - optional dependency
    pytest.skip("textual is not installed", allow_module_level=True)


def test_app_uses_packaged_stylesheet() -> None:
    """The application should resolve the packaged stylesheet at runtime."""

    import streamdeck_tui.app as app_module
    from streamdeck_tui.config import AppConfig

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert css_resource.is_file(), "Expected stylesheet to be packaged with the app"
    css_text = css_resource.read_text(encoding="utf-8").strip()

    assert css_text.startswith("#layout")
    assert app_module.StreamdeckApp.CSS.strip() == css_text
    assert app_module.StreamdeckApp.CSS_PATH == []

    app = app_module.StreamdeckApp(AppConfig())
    try:
        assert app.stylesheet is not None
    finally:
        app.exit(result=None)


def test_stylesheet_fallback_when_packaged_css_is_unsupported(tmp_path) -> None:
    """Inline CSS should be used if the packaged file has unsupported tokens."""

    import streamdeck_tui.app as app_module

    css_path = tmp_path / "streamdeck.css"
    css_path.write_text("#layout {\n    gap: 1;\n    border: solid $surface 1;\n}\n", encoding="utf-8")

    original_files = app_module.resources.files

    class _DummyTraversable:
        def joinpath(self, _name: str):
            return css_path

    def fake_files(_package: str):
        return _DummyTraversable()

    app_module.resources.files = fake_files  # type: ignore[attr-defined]
    try:
        reloaded = importlib.reload(app_module)
        assert reloaded.DEFAULT_CSS.strip() == reloaded._INLINE_DEFAULT_CSS.strip()
        assert reloaded.StreamdeckApp.CSS.strip() == reloaded._INLINE_DEFAULT_CSS.strip()
    finally:
        app_module.resources.files = original_files  # type: ignore[attr-defined]
        importlib.reload(app_module)
