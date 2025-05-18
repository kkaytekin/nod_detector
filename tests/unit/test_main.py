"""Tests for the __main__ module."""

import sys
from unittest.mock import MagicMock, patch

from nod_detector import __main__


@patch("nod_detector.__main__.app")
def test_main(mock_app: MagicMock) -> None:
    """Test that main() calls the app function."""
    with patch.object(sys, "argv", ["nod_detector"]):
        __main__.main()
    mock_app.assert_called_once()


@patch("nod_detector.__main__.app")
@patch("nod_detector.__main__.sys")
def test_main_module(mock_sys: MagicMock, mock_app: MagicMock) -> None:
    """Test that __main__ module calls main() when run directly."""
    mock_sys.modules = {"__main__": __main__}
    with patch.object(__main__, "__name__", "__main__"):
        __main__.main()
    mock_app.assert_called_once()


@patch("nod_detector.__main__.app")
def test_main_with_args(mock_app: MagicMock) -> None:
    """Test that main() passes command line arguments to the app."""
    test_args = ["nod_detector", "--help"]
    with patch.object(sys, "argv", test_args):
        __main__.main()
    mock_app.assert_called_once()
