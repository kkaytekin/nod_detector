"""
Unit tests for the VideoProcessingPipeline class.
"""

from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import cv2
import numpy as np
import pytest

from nod_detector.pipeline.video_processing_pipeline import (
    FrameResult,
    VideoProcessingPipeline,
)


@pytest.fixture
def mock_rerun():
    """Fixture to mock rerun functionality."""
    with (
        patch("rerun.init"),
        patch("rerun.set_time_sequence"),
        patch("rerun.log"),
        patch("rerun.Image"),
        patch("rerun.Scalar"),
        patch("rerun.TextLog"),
    ):
        yield


@pytest.fixture
def sample_frame():
    """Return a sample video frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_result():
    """Return a sample frame result for testing."""
    return FrameResult(
        frame_number=0,
        timestamp=0.0,
        detections=[],
        head_pose={"pitch": 5.0, "yaw": 0.0, "roll": 0.0},
        pose_landmarks=None,
        face_landmarks=None,
        nod_detected=False,
    )


def test_video_processing_pipeline_init():
    """Test that the VideoProcessingPipeline initializes correctly."""
    # Test with default config
    pipeline = VideoProcessingPipeline()
    assert pipeline.config == {}
    assert pipeline.frame_count == 0

    # Test with custom config
    config = {"output_dir": "test_output"}
    pipeline = VideoProcessingPipeline(config=config)
    assert pipeline.config == config
    assert pipeline.output_dir.name == "test_output"


@pytest.mark.skip(reason="Temporarily disabled for debugging")
@patch("cv2.VideoCapture")
@patch("cv2.cvtColor")
@patch("rerun.init")
@patch("rerun.log")
@patch("rerun.set_time_sequence")
@patch("rerun.Image")
@patch("rerun.Scalar")
@patch("rerun.TextLog")
def test_video_processing_pipeline_process(
    mock_text_log,
    mock_scalar,
    mock_image,
    mock_set_time_sequence,
    mock_log,
    mock_rr_init,
    mock_cvt_color,
    mock_video_capture,
    sample_video_path,
    sample_frame,
    sample_frame_result,
):
    """Test the video processing pipeline process method."""
    # Setup mock VideoCapture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [
        (True, sample_frame),
        (False, None),  # Simulate end of video
    ]
    mock_cap.get.side_effect = [30.0, 640, 480, 100]  # fps, width, height, frame_count
    mock_video_capture.return_value = mock_cap

    # Setup mock color conversion
    mock_cvt_color.return_value = sample_frame

    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    # Mock the detector
    mock_detector = MagicMock()
    mock_detector.process_frame.return_value = (
        {"head_pose": {"pitch": 5.0, "yaw": 0.0, "roll": 0.0}, "pose_landmarks": {}, "face_landmarks": {}, "detections": [], "timestamp": 0.0},
        sample_frame,
    )
    pipeline.detector = mock_detector

    # Process video
    results = pipeline.process(sample_video_path, visualize=True, debug=True)

    # Verify results
    assert "video_info" in results
    assert "frame_results" in results
    assert len(results["frame_results"]) == 1  # Only one frame in debug mode

    # Verify video capture was called with correct path
    mock_video_capture.assert_called_once_with(sample_video_path)
    mock_cap.release.assert_called_once()


@patch("cv2.VideoCapture")
def test_video_processing_invalid_video(mock_video_capture):
    """Test that an error is raised when an invalid video path is provided."""
    # Setup mock to simulate video open failure
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    pipeline = VideoProcessingPipeline()
    with pytest.raises(IOError):
        pipeline.process("nonexistent_video.mp4")


@patch("rerun.set_time_sequence")
@patch("rerun.log")
@patch("rerun.Image")
@patch("rerun.Scalar")
@patch("rerun.TextLog")
def test_visualize(
    mock_text_log,
    mock_scalar,
    mock_image,
    mock_log,
    mock_set_time_sequence,
    sample_frame,
    sample_frame_result,
):
    """Test the _visualize method."""
    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    # Call visualize method
    pipeline._visualize(sample_frame, sample_frame_result)

    # Verify rerun was called with correct parameters
    mock_set_time_sequence.assert_called_once_with("frame", 0)

    # Verify error handling with invalid frame
    pipeline._visualize(None, sample_frame_result)  # Should not raise
    pipeline._visualize(np.array([]), sample_frame_result)  # Should not raise
