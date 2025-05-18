"""Unit tests for MediaPipe components."""

from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pytest

from src.nod_detector.mediapipe_components import (
    HeadPose,
    Landmark3D,
    MediaPipeDetector,
    load_results_from_json,
    save_results_to_json,
)


def test_landmark3d_initialization():
    """Test Landmark3D initialization and to_dict method."""
    landmark = Landmark3D(x=0.1, y=0.2, z=0.3, visibility=0.9)
    assert landmark.x == 0.1
    assert landmark.y == 0.2
    assert landmark.z == 0.3
    assert landmark.visibility == 0.9
    assert landmark.to_dict() == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "visibility": 0.9,
    }


def test_head_pose_initialization():
    """Test HeadPose initialization and to_dict method."""
    pose = HeadPose(pitch=10.0, yaw=-5.0, roll=2.5)
    assert pose.pitch == 10.0
    assert pose.yaw == -5.0
    assert pose.roll == 2.5
    assert pose.to_dict() == {"pitch": 10.0, "yaw": -5.0, "roll": 2.5}


def test_mediapipe_detector_initialization():
    """Test MediaPipeDetector initialization and context manager."""
    with MediaPipeDetector(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        static_image_mode=True,
        refine_face_landmarks=False,
    ) as detector:
        assert detector is not None
        assert detector.min_detection_confidence == 0.7
        assert detector.min_tracking_confidence == 0.6
        assert detector.static_image_mode is True
        assert detector.refine_face_landmarks is False


@patch("cv2.cvtColor")
@patch("mediapipe.solutions.pose.Pose")
@patch("mediapipe.solutions.face_mesh.FaceMesh")
@patch("mediapipe.solutions.drawing_utils.draw_landmarks")
def test_mediapipe_detector_process_frame(mock_draw_landmarks, mock_face_mesh, mock_pose, mock_cvt_color):
    """Test MediaPipeDetector process_frame method with mocked MediaPipe."""

    # Setup mock cvtColor to return a copy of the input
    def cvt_color_side_effect(frame, _):
        return frame.copy()

    mock_cvt_color.side_effect = cvt_color_side_effect

    # Setup mock pose results
    mock_pose_instance = MagicMock()
    mock_pose.return_value = mock_pose_instance

    # Create a mock landmark with required attributes
    mock_landmark = MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5
    mock_landmark.z = 0.0
    mock_landmark.visibility = 0.9
    mock_landmark.presence = 1.0

    # Setup pose results
    mock_pose_results = MagicMock()
    mock_pose_results.pose_landmarks = MagicMock()
    mock_pose_results.pose_landmarks.landmark = [mock_landmark] * 33  # 33 pose landmarks
    mock_pose_instance.process.return_value = mock_pose_results

    # Setup face mesh results
    mock_face_mesh_instance = MagicMock()
    mock_face_mesh.return_value = mock_face_mesh_instance

    mock_face_landmarks = MagicMock()
    mock_face_landmarks.landmark = [mock_landmark] * 478  # 478 face landmarks
    mock_face_results = MagicMock()
    mock_face_results.multi_face_landmarks = [mock_face_landmarks]
    mock_face_mesh_instance.process.return_value = mock_face_results

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initialize detector and process frame
    with MediaPipeDetector() as detector:
        results, annotated_frame = detector.process_frame(test_frame)

    # Verify results
    assert "pose_landmarks" in results
    assert "face_landmarks" in results
    assert "head_pose" in results

    # Verify pose landmarks
    assert len(results["pose_landmarks"]) == 33
    assert results["pose_landmarks"][0] == {
        "x": 0.5,
        "y": 0.5,
        "z": 0.0,
        "visibility": 0.9,
    }

    # Verify face landmarks
    assert len(results["face_landmarks"]) == 478

    # Verify head pose (mocked calculation)
    assert "pitch" in results["head_pose"]
    assert "yaw" in results["head_pose"]
    assert "roll" in results["head_pose"]

    # Verify annotated frame
    assert annotated_frame.shape == test_frame.shape
    assert annotated_frame.dtype == np.uint8

    # Verify draw_landmarks was called
    assert mock_draw_landmarks.called


def test_save_and_load_results(tmp_path):
    """Test saving and loading results to/from JSON."""
    # Create test data
    test_results = {
        "frame_number": 1,
        "landmarks": {
            "nose": {"x": 0.5, "y": 0.5, "z": 0.0},
            "left_eye": {"x": 0.4, "y": 0.45, "z": 0.1},
        },
        "head_pose": {"pitch": 5.0, "yaw": 0.0, "roll": 2.0},
    }

    # Save to file
    output_path = tmp_path / "test_results.json"
    save_results_to_json(test_results, output_path)

    # Verify file exists
    assert output_path.exists()

    # Load from file
    loaded_results = load_results_from_json(output_path)

    # Verify loaded data matches original
    assert loaded_results == test_results


def test_mediapipe_detector_with_real_image():
    """Test MediaPipeDetector with a real image (requires MediaPipe installation)."""
    # Skip this test if running in CI environment
    import os

    if os.getenv("CI"):
        pytest.skip("Skipping test that requires MediaPipe in CI environment")

    # Create a blank test image
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Initialize detector and process frame
    with MediaPipeDetector() as detector:
        results, annotated_frame = detector.process_frame(test_frame)

    # Verify basic structure of results
    assert isinstance(results, dict)
    assert "pose_landmarks" in results
    assert "face_landmarks" in results
    assert "head_pose" in results

    # Verify annotated frame
    assert annotated_frame.shape == test_frame.shape
    assert annotated_frame.dtype == np.uint8


@patch("cv2.cvtColor")
@patch("mediapipe.solutions.drawing_utils.draw_landmarks")
@patch("mediapipe.solutions.face_mesh.FaceMesh")
@patch("mediapipe.solutions.pose.Pose")
def test_mediapipe_detector_error_handling(mock_pose, mock_face_mesh, mock_draw_landmarks, mock_cvt_color):
    """Test MediaPipeDetector error handling with invalid input."""
    # Setup mock processors
    mock_pose_instance = MagicMock()
    mock_face_mesh_instance = MagicMock()
    mock_pose.return_value = mock_pose_instance
    mock_face_mesh.return_value = mock_face_mesh_instance

    # Configure cvtColor to return a copy of the input frame
    def cvt_color_side_effect(frame, _):
        return frame.copy()

    mock_cvt_color.side_effect = cvt_color_side_effect

    # Configure draw_landmarks to handle empty frames
    def draw_landmarks_side_effect(image, *args, **kwargs):
        if image.size == 0:
            raise ValueError("Cannot draw on empty image")
        return image

    mock_draw_landmarks.side_effect = draw_landmarks_side_effect

    with MediaPipeDetector() as detector:
        # Replace the actual processors with our mocks
        detector.pose = mock_pose_instance
        detector.face_mesh = mock_face_mesh_instance

        # Test with None input - should raise ValueError
        with pytest.raises(ValueError, match="Input frame cannot be None"):
            detector.process_frame(None)  # type: ignore

        # Test with invalid image shape (1D array)
        with pytest.raises(ValueError, match="Input frame must be a 3-channel BGR image"):
            invalid_frame = np.zeros((100,), dtype=np.uint8)
            detector.process_frame(invalid_frame)  # type: ignore

        # Test with empty image (0x0x3 shape) - should handle gracefully
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        results, processed_frame = detector.process_frame(empty_frame)  # type: ignore
        assert isinstance(results, dict)
        assert processed_frame.shape == empty_frame.shape

        # Test with valid shape but empty data - should handle gracefully
        valid_shape_empty_data = np.zeros((100, 100, 3), dtype=np.uint8)
        results, processed_frame = detector.process_frame(valid_shape_empty_data)  # type: ignore
        assert isinstance(results, dict)
        assert processed_frame.shape == valid_shape_empty_data.shape
