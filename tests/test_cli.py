import pytest

from streamdeck_tui import __version__
from streamdeck_tui.cli import parse_args


def test_parse_args_version_flag_exits_and_prints_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_args(["--version"])

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"{__version__}\n"
    assert captured.err == ""
