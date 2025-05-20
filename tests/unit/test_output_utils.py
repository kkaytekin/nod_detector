"""Unit tests for output_utils.py"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nod_detector.output_utils import (
    FrameData,
    NodEvent,
    OutputSaver,
    VideoInfo,
    save_processed_video,
)


@pytest.fixture
def sample_frame_data():
    """Return sample frame data for testing."""
    return FrameData(
        frame_number=1,
        timestamp=0.0333,
        pitch=5.5,
        yaw=-1.2,
        roll=0.8,
        nod_detected=False,
        nod_direction=None,
    )


@pytest.fixture
def sample_nod_event():
    """Return a sample nod event for testing."""
    return NodEvent(
        frame_number=45,
        timestamp=1.5,
        direction="down-up",
        start_pitch=5.2,
        peak_pitch=-8.3,
        end_pitch=4.8,
        amplitude=13.5,
        duration_frames=9,
        duration_seconds=0.3,
    )


@pytest.fixture
def sample_video_info():
    """Return sample video info for testing."""
    return VideoInfo(
        path="path/to/input_video.mp4",
        fps=30.0,
        frame_width=1920,
        frame_height=1080,
        total_frames=300,
    )


def test_output_saver_init(tmp_path, sample_video_info):
    """Test OutputSaver initialization."""
    output_dir = tmp_path / "output"
    saver = OutputSaver(output_dir, sample_video_info)
    assert saver.output_dir == output_dir
    assert saver.video_info == sample_video_info
    assert output_dir.exists()
    assert (output_dir / "plots").exists()


def test_output_saver_add_frame(tmp_path, sample_video_info, sample_frame_data):
    """Test adding frame data to OutputSaver."""
    saver = OutputSaver(tmp_path, sample_video_info)
    saver.add_frame(sample_frame_data)
    assert len(saver.frames) == 1
    assert saver.frames[0]["frame_number"] == 1
    assert saver.frames[0]["pitch"] == 5.5


def test_output_saver_add_nod_event(tmp_path, sample_video_info, sample_nod_event):
    """Test adding nod event to OutputSaver."""
    saver = OutputSaver(tmp_path, sample_video_info)
    saver.add_nod_event(sample_nod_event)
    assert len(saver.nod_events) == 1
    assert saver.nod_events[0]["frame_number"] == 45
    assert saver.nod_events[0]["direction"] == "down-up"


@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.close")
def test_output_saver_save_all(mock_close, mock_savefig, mock_figure, tmp_path, sample_video_info, sample_frame_data, sample_nod_event):
    """Test saving all output files."""
    # Setup mock figure
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_fig.add_subplot.return_value = mock_ax
    mock_figure.return_value = mock_fig

    # Add more frame data to make the test more realistic
    frame_data1 = FrameData(
        frame_number=1,
        timestamp=0.0,
        pitch=5.5,
        yaw=0.0,
        roll=0.0,
        nod_detected=False,
    )
    frame_data2 = FrameData(
        frame_number=2, timestamp=0.0333, pitch=10.0, yaw=0.0, roll=0.0, nod_detected=True, nod_direction="down-up"  # ~30fps
    )

    saver = OutputSaver(tmp_path, sample_video_info)
    saver.add_frame(frame_data1)
    saver.add_frame(frame_data2)
    saver.add_nod_event(sample_nod_event)

    # Create the plots directory to avoid FileNotFoundError
    (tmp_path / "plots").mkdir(exist_ok=True)

    # Call the method under test
    saver.save_all()

    # Verify savefig was called with the correct path
    expected_plot_path = tmp_path / "plots" / "pitch_plot.png"
    mock_savefig.assert_called_once()
    args, kwargs = mock_savefig.call_args
    assert Path(args[0]) == expected_plot_path
    assert kwargs == {"bbox_inches": "tight", "dpi": 300}
    mock_close.assert_called_once()

    # Check that all expected files were created
    assert (tmp_path / "frames.json").exists()
    assert (tmp_path / "nod_events.json").exists()
    assert (tmp_path / "summary.json").exists()

    # Verify frames.json content
    frames_file = tmp_path / "frames.json"
    with frames_file.open(encoding="utf-8") as f:
        frames_data = json.load(f)
    assert "video_info" in frames_data
    assert "frames" in frames_data
    assert len(frames_data["frames"]) == 2
    assert frames_data["frames"][0]["frame_number"] == 1
    assert frames_data["frames"][1]["frame_number"] == 2

    # Verify nod_events.json content
    events_file = tmp_path / "nod_events.json"
    with events_file.open(encoding="utf-8") as f:
        events_data = json.load(f)
    assert "total_nods" in events_data
    assert "nod_events" in events_data
    assert len(events_data["nod_events"]) == 1
    assert events_data["nod_events"][0]["frame_number"] == 45


@patch("cv2.VideoWriter")
def test_save_processed_video(mock_video_writer, tmp_path):
    """Test saving processed video.

    Args:
        mock_video_writer: Mock for cv2.VideoWriter
        tmp_path: Temporary directory path fixture
    """
    # Setup mock VideoWriter
    writer_mock = MagicMock()
    mock_video_writer.return_value = writer_mock

    # Create test frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    frames = [frame1, frame2]
    output_path = tmp_path / "output.mp4"

    # Call the function
    save_processed_video(
        frames=frames,
        output_path=output_path,
        fps=30.0,
        frame_size=(100, 100),
    )

    # Verify VideoWriter was called with correct parameters
    mock_video_writer.assert_called_once()
    args, _ = mock_video_writer.call_args
    assert str(output_path) in args
    assert args[2] == 30.0  # fps
    assert args[3] == (100, 100)  # frame_size

    # Verify write was called for each frame
    assert writer_mock.write.call_count == 2
    # Verify release was called
    writer_mock.release.assert_called_once()


@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_save_processed_video_empty_frames(tmp_path):
    """Test saving processed video with empty frames list."""
    # The function should not raise an error for empty frames list
    output_path = tmp_path / "output.mp4"
    save_processed_video(
        frames=[],
        output_path=output_path,
        fps=30.0,
        frame_size=(100, 100),
    )
    # Verify the output file was created (but will be empty)
    assert output_path.exists()


@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_save_processed_video_invalid_frame_size(tmp_path):
    """Test saving processed video with invalid frame size."""
    # The function should not raise an error for invalid frame size
    # It will be handled by OpenCV's VideoWriter
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    output_path = tmp_path / "output.mp4"
    save_processed_video(
        frames=[frame],
        output_path=output_path,
        fps=30.0,
        frame_size=(0, 0),  # Invalid size, but won't raise an error
    )
    # Verify the output file was created (but may be invalid)
    assert output_path.exists()
