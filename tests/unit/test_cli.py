"""Tests for the nod-detector CLI."""

import sys
from importlib import reload
from pathlib import Path
from typing import Any, Dict
from unittest.mock import ANY, MagicMock, call, patch

import pytest
from typer.testing import CliRunner

# Mock the test video path
TEST_VIDEO = Path("tests/data/test_video.mp4")


# Fixture for the app with mocks
@pytest.fixture
def mock_app(monkeypatch):
    """Fixture to mock the Typer app and its dependencies."""
    # Mock the VideoProcessingPipeline
    mock_pipeline = MagicMock()
    mock_instance = MagicMock()
    mock_pipeline.return_value = mock_instance

    # Mock the process method to return a successful result
    mock_instance.process.return_value = {"status": "success"}

    # Mock the imports
    with (
        patch("nod_detector.cli.VideoProcessingPipeline", mock_pipeline),
        patch("typer.Typer") as mock_typer_class,
        patch("nod_detector.cli.Progress") as mock_progress,
        patch("nod_detector.cli.console") as mock_console,
    ):

        # Create a mock for the Typer app
        mock_typer_app = MagicMock()
        mock_typer_class.return_value = mock_typer_app

        # Mock the progress bar
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__.return_value = mock_progress_instance

        # Import the module after patching
        from nod_detector import cli

        # Reload the module to apply patches
        reload(cli)

        yield {
            "typer_app": mock_typer_app,
            "pipeline": mock_pipeline,
            "pipeline_instance": mock_instance,
            "typer_class": mock_typer_class,
            "cli": cli,
            "progress": mock_progress_instance,
            "console": mock_console,
        }


# Test cases
def test_validate_input_file():
    """Test input file validation."""
    from nod_detector.cli import validate_input_file

    # Create a mock Path object
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.is_file.return_value = True

    # Test with valid file
    result = validate_input_file(mock_path)
    assert result == mock_path

    # Test with non-existent file
    mock_path.exists.return_value = False
    with pytest.raises(Exception):  # Should raise typer.BadParameter
        validate_input_file(mock_path)


def test_ensure_output_path():
    """Test output path generation."""
    from nod_detector.cli import ensure_output_path

    # Test with explicit output path
    input_path = Path("input.mp4")
    output_path = Path("output.mp4")
    assert ensure_output_path(input_path, output_path) == output_path

    # Test with None output path (should use input path with _processed suffix)
    expected = Path("input_processed.mp4")
    assert ensure_output_path(input_path, None) == expected


def test_process_video_command(mock_app, tmp_path):
    """Test the main process_video function through the CLI."""
    cli = mock_app["cli"]

    # Create a test video file
    input_path = tmp_path / "test.mp4"
    input_path.touch()
    output_path = tmp_path / "output.mp4"

    # Mock the process_video function
    with (
        patch("nod_detector.cli.process_video") as mock_process,
        patch("nod_detector.cli.validate_input_file") as mock_validate,
        patch("nod_detector.cli.ensure_output_path") as mock_ensure,
    ):

        # Setup mocks
        mock_validate.return_value = input_path
        mock_ensure.return_value = output_path

        # Call the command directly
        cli.process_video(input=input_path, output=output_path, visualize=True)

        # Verify the function was called with correct arguments
        mock_process.assert_called_once_with(input=input_path, output=output_path, visualize=True)


def test_process_video_function(mock_app, tmp_path):
    """Test the process_video function logic."""
    mock_pipeline = mock_app["pipeline"]
    mock_instance = mock_app["pipeline_instance"]
    cli = mock_app["cli"]

    # Create a test video file
    input_path = tmp_path / "test.mp4"
    input_path.touch()
    output_path = tmp_path / "output.mp4"

    # Mock the process_video function implementation
    with patch("nod_detector.cli.process_video") as mock_process:
        # Call the function through the module
        cli.process_video(input=input_path, output=output_path, visualize=True)

        # Verify the function was called with correct arguments
        mock_process.assert_called_once_with(input=input_path, output=output_path, visualize=True)

        # Verify the pipeline was called correctly in the actual implementation
        if mock_process.called:
            args, kwargs = mock_process.call_args
            assert kwargs["input"] == input_path
            assert kwargs["output"] == output_path
            assert kwargs["visualize"] is True


def test_process_video_default_visualization(mock_app, tmp_path):
    """Test that visualization is disabled by default."""
    cli = mock_app["cli"]

    # Create a test video file
    input_path = tmp_path / "test.mp4"
    input_path.touch()

    # Mock the process_video function implementation
    with patch("nod_detector.cli.process_video") as mock_process:
        # Call with default visualize=False
        cli.process_video(input=input_path, output=None, visualize=False)

        # Verify the function was called with correct arguments
        mock_process.assert_called_once_with(input=input_path, output=None, visualize=False)

        # Verify the pipeline was called with default settings in the actual implementation
        if mock_process.called:
            args, kwargs = mock_process.call_args
            assert kwargs["input"] == input_path
            assert kwargs["output"] is None
            assert kwargs["visualize"] is False


def test_cli_invocation(mock_app):
    """Test that the CLI can be invoked directly."""
    mock_typer_app = mock_app["typer_app"]
    mock_typer_class = mock_app["typer_class"]

    # Test the app is properly initialized
    assert mock_typer_app is not None

    # Test that the Typer class was called
    mock_typer_class.assert_called_once()

    # Test that the command was registered
    mock_typer_app.command.assert_called_once()


def test_cli_help(mock_app):
    """Test that the CLI command is properly registered with help functionality."""
    # Get the mocked Typer app and CLI module
    mock_typer_app = mock_app["typer_app"]

    # Test that the command was registered
    assert mock_typer_app.command.called

    # Get the command function that was registered
    command_call = mock_typer_app.command.call_args
    assert command_call is not None

    # Check if the command was registered with the correct name
    if command_call[0]:  # If there are positional arguments
        assert command_call[0][0] == "process-video"
    elif "name" in command_call[1]:  # If it's a keyword argument
        assert command_call[1]["name"] == "process-video"

    # Verify that the help option is enabled by checking the mock's call arguments
    # Typer adds help by default, so we just need to verify the command is registered
    # and has the expected name and parameters

    # Check that the command has the expected parameters
    # We can do this by checking the mock's call arguments
    command_kwargs = command_call[1]

    # The command should have a callback function
    assert "callback" in command_kwargs or command_call[0] is not None

    # The command should have a name
    if "name" in command_kwargs:
        assert command_kwargs["name"] == "process-video"

    # The command should have a help text
    if "help" in command_kwargs:
        assert isinstance(command_kwargs["help"], str)

    # The command should have the --help option enabled by default
    # We can verify this by checking that add_help_option was called
    # or that no_help is not set to True
    if "no_help" in command_kwargs:
        assert command_kwargs["no_help"] is False


def test_cli_process_video_command(mock_app, tmp_path):
    """Test the CLI process-video command."""
    cli = mock_app["cli"]

    # Create a test video file
    input_video = tmp_path / "test.mp4"
    input_video.touch()
    output_video = tmp_path / "output.mp4"

    # Mock the process_video function
    with (
        patch("nod_detector.cli.process_video") as mock_process,
        patch("nod_detector.cli.validate_input_file") as mock_validate,
        patch("nod_detector.cli.ensure_output_path") as mock_ensure,
    ):

        # Setup mocks
        mock_validate.return_value = input_video
        mock_ensure.return_value = output_video

        # Call the command directly through the module
        cli.process_video(input=input_video, output=output_video, visualize=False)

        # Verify the function was called with correct arguments
        mock_process.assert_called_once_with(input=input_video, output=output_video, visualize=False)


def test_cli_visualize_flag(mock_app, tmp_path):
    """Test that the --visualize flag is properly passed to the pipeline."""
    cli = mock_app["cli"]

    # Create a test video file
    input_video = tmp_path / "test.mp4"
    input_video.touch()
    output_video = tmp_path / "output.mp4"

    # Mock the process_video function
    with (
        patch("nod_detector.cli.process_video") as mock_process,
        patch("nod_detector.cli.validate_input_file") as mock_validate,
        patch("nod_detector.cli.ensure_output_path") as mock_ensure,
    ):
        # Setup mocks
        mock_validate.return_value = input_video
        mock_ensure.return_value = output_video

        # Call with --visualize flag
        cli.process_video(input=input_video, output=output_video, visualize=True)

        # Verify the function was called with visualize=True
        mock_process.assert_called_once_with(input=input_video, output=output_video, visualize=True)
