"""
Unit tests for the VideoProcessingPipeline class.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from nod_detector.pipeline.video_processing_pipeline import VideoProcessingPipeline


def test_video_processing_pipeline_init():
    """Test that the VideoProcessingPipeline initializes correctly."""
    # Test with default config
    pipeline = VideoProcessingPipeline()
    assert pipeline.config == {}
    assert pipeline.frame_count == 0

    # Test with custom config
    config = {"test_param": 123}
    pipeline = VideoProcessingPipeline(config=config)
    assert pipeline.config == config


@patch("cv2.VideoCapture")
def test_video_processing_pipeline_process(mock_video_capture, sample_video_path):
    """Test the video processing pipeline with a mock video capture."""
    # Setup mock video capture
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = [
        30.0,
        1920,
        1080,
        100,
    ]  # fps, width, height, frame_count

    # Mock frame data
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, test_frame),
        (False, None),
    ]  # One frame then done

    mock_video_capture.return_value = mock_cap

    # Initialize and run pipeline
    pipeline = VideoProcessingPipeline()
    results = pipeline.process(sample_video_path)

    # Verify results
    assert "video_info" in results
    assert "frame_results" in results
    assert "detections" in results
    assert results["video_info"]["fps"] == 30.0
    assert results["video_info"]["frame_width"] == 1920
    assert results["video_info"]["frame_height"] == 1080
    assert results["video_info"]["total_frames"] == 100
    assert len(results["frame_results"]) == 1  # Should process one frame

    # Verify video capture was called correctly
    mock_video_capture.assert_called_once_with(sample_video_path)
    mock_cap.release.assert_called_once()


def test_video_processing_invalid_video():
    """Test that an error is raised when an invalid video path is provided."""
    pipeline = VideoProcessingPipeline()
    with pytest.raises(IOError, match="Could not open video file"):
        pipeline.process("nonexistent_video.mp4")
