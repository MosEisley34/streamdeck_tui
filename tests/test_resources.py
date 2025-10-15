"""Resource configuration tests for streamdeck_tui."""

from streamdeck_tui.app import StreamdeckApp
from streamdeck_tui.config import AppConfig


def test_app_uses_default_styles() -> None:
    """The application should rely on Textual defaults without custom CSS."""

    assert StreamdeckApp.CSS == ""
    app = StreamdeckApp(AppConfig())
    try:
        assert not app.css_path
    finally:
        app.exit(result=None)
