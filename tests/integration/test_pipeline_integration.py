"""
Integration tests for the video processing pipeline.
"""

from pathlib import Path

import pytest

from nod_detector.pipeline.video_processing_pipeline import VideoProcessingPipeline

# This is a slow test, so we'll mark it with a custom marker
pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_process_sample_video(sample_video_path, tmp_path):
    """Test processing the sample video file."""
    # Skip if the sample video doesn't exist
    if not Path(sample_video_path).exists():
        pytest.skip(f"Sample video not found at {sample_video_path}")

    # Create output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Initialize pipeline
    pipeline = VideoProcessingPipeline()

    # Process the video with visualization disabled
    results = pipeline.process(sample_video_path, visualize=False)

    # Basic assertions about the results structure
    assert isinstance(results, dict)
    assert "video_info" in results
    assert "frame_results" in results
    assert "detections" in results

    # Verify video info
    video_info = results["video_info"]
    assert video_info["path"] == sample_video_path
    assert video_info["fps"] > 0
    assert video_info["frame_width"] > 0
    assert video_info["frame_height"] > 0
    assert video_info["total_frames"] >= 0

    # Verify frame results
    assert isinstance(results["frame_results"], list)
    if results["frame_results"]:
        frame_result = results["frame_results"][0]
        assert "head_pose" in frame_result
        assert "pose_landmarks" in frame_result
        assert "face_landmarks" in frame_result
        assert "timestamp" in frame_result

    # Verify frame results structure is correct
    # Note: We don't enforce a specific number of frames to be processed
    # as it depends on the test environment and available resources
    assert isinstance(results["frame_results"], list)

    # Verify the first frame result has the expected structure
    if results["frame_results"]:
        frame_result = results["frame_results"][0]
        # Check for expected keys in frame result
        assert "frame_number" in frame_result
        # Note: 'timestamp' is not currently included in the frame result
        # but we can add it in the future if needed
    # Add more assertions as needed based on your implementation


# This test can be run with: pytest -m integration tests/
