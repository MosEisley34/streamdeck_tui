"""Resource packaging tests for streamdeck_tui."""

from importlib import resources


def test_css_resource_is_packaged() -> None:
    """The Textual stylesheet should be present in the installed package."""

    css_path = resources.files("streamdeck_tui").joinpath("streamdeck.css")
    assert css_path.is_file(), "streamdeck.css should be included with the package"
