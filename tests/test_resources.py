"""Resource packaging tests for streamdeck_tui."""

from importlib import resources
from pathlib import Path

from streamdeck_tui.app import StreamdeckApp


def test_css_resource_is_packaged() -> None:
    """The Textual stylesheet should be present in the installed package."""

    css_resource = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert css_resource.is_file(), "streamdeck.css should be included with the package"
    assert Path(StreamdeckApp.CSS_PATH).is_file(), "StreamdeckApp should resolve the stylesheet path"
