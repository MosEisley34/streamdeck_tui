"""Resource packaging tests for streamdeck_tui."""

from importlib import resources
from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_css_resource_is_packaged() -> None:
    """The Textual stylesheet should be present in the installed package."""

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert css_resource.is_file(), "streamdeck.css should be included with the package"
    assert StreamdeckApp.CSS.strip() == css_resource.read_text(encoding="utf-8").strip()


def test_app_uses_inline_stylesheet() -> None:
    """The application should not require an external CSS path to start."""

    assert getattr(StreamdeckApp, "CSS_PATH", None) is None
    app = StreamdeckApp(AppConfig())
    try:
        assert not app.css_path
    finally:
        app.exit(result=None)
