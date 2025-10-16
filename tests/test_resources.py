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


def test_warn_if_legacy_stylesheet_no_file(monkeypatch) -> None:
    """No warning is emitted when the legacy stylesheet is absent."""

    from streamdeck_tui import cli

    calls = []

    def fake_warning(*args, **kwargs) -> None:  # pragma: no cover - defensive
        calls.append((args, kwargs))

    monkeypatch.setattr(cli.log, "warning", fake_warning)

    def fake_files(package: str) -> object:
        assert package == "streamdeck_tui"

        class _Path:
            def joinpath(self, name: str) -> "_Path":
                return self

            def is_file(self) -> bool:
                return False

        return _Path()

    monkeypatch.setattr(cli.resources, "files", fake_files)

    cli.warn_if_legacy_stylesheet()

    assert not calls


def test_warn_if_legacy_stylesheet_when_present(monkeypatch) -> None:
    """A helpful warning is logged if the legacy stylesheet is discovered."""

    from streamdeck_tui import cli

    class DummyPath:
        def __init__(self, *, exists: bool, path: str) -> None:
            self._exists = exists
            self._path = path

        def joinpath(self, name: str) -> "DummyPath":
            return DummyPath(exists=self._exists, path=f"{self._path}/{name}")

        def is_file(self) -> bool:
            return self._exists

        def __str__(self) -> str:  # pragma: no cover - defensive
            return self._path

    def fake_files(package: str) -> DummyPath:
        assert package == "streamdeck_tui"
        return DummyPath(exists=True, path="/tmp/legacy")

    monkeypatch.setattr(cli.resources, "files", fake_files)

    calls = []

    def fake_warning(message: str, path: object) -> None:
        calls.append(message)

    monkeypatch.setattr(cli.log, "warning", fake_warning)

    cli.warn_if_legacy_stylesheet()

    assert calls and "legacy stylesheet" in calls[0]
