"""Tests for the nod-detector CLI."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the module under test
from nod_detector.cli import app


# Fixture for the test runner
@pytest.fixture
def runner():
    """Create a CliRunner instance for testing."""
    return CliRunner()


# Fixture for the test video file
@pytest.fixture
def test_video(tmp_path):
    """Create a test video file."""
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    return video_path


def test_cli_help(runner):
    """Test that the --help flag works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_cli_defaults(runner, test_video):
    """Test the CLI with default parameters."""
    with patch("nod_detector.cli.VideoProcessingPipeline") as mock_pipeline:
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.process.return_value = {"status": "success"}
        mock_pipeline.return_value = mock_instance

        # Call the CLI with minimal arguments
        result = runner.invoke(app, [str(test_video)])

        # Verify the result
        assert result.exit_code == 0
        mock_instance.process.assert_called_once()
        args, kwargs = mock_instance.process.call_args
        assert args[0] == str(test_video)
        assert kwargs["visualize"] is False
        assert kwargs["debug"] is False
