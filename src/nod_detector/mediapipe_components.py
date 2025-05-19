"""MediaPipe components for 3D pose and face landmark detection.

This module provides functionality to detect and track 3D pose and face landmarks
using MediaPipe. It includes classes for detecting landmarks and calculating
head pose angles.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Initialize MediaPipe components
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class Landmark3D:
    """Represents a 3D landmark with x, y, z coordinates and visibility.

    Attributes:
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        visibility: Confidence score [0, 1] that the landmark is visible
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        """Convert landmark to dictionary."""
        return asdict(self)


@dataclass
class HeadPose:
    """Represents head pose angles in degrees.

    Attributes:
        pitch: Nodding (up/down) rotation around X-axis
        yaw: Shaking (left/right) rotation around Y-axis
        roll: Tilting (ear to shoulder) rotation around Z-axis
    """

    pitch: float  # Nodding (up/down)
    yaw: float  # Shaking (left/right)
    roll: float  # Tilting (ear to shoulder)

    def to_dict(self) -> dict[str, float]:
        """Convert head pose to dictionary."""
        return asdict(self)


class MediaPipeDetector:
    """MediaPipe detector for 3D pose and face landmarks."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        refine_face_landmarks: bool = True,
    ) -> None:
        """Initialize the MediaPipe detector.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: Whether to treat input as static images or video
            refine_face_landmarks: Whether to refine face landmarks
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.refine_face_landmarks = refine_face_landmarks

        # Initialize MediaPipe models
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=2,  # Highest complexity for best accuracy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,  # We only need one face
            refine_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Initialize results storage with proper type hints
        self.pose_landmarks: Optional[Dict[int, Dict[str, float]]] = None
        self.face_landmarks: Optional[Dict[int, Dict[str, float]]] = None
        self.head_pose: Optional[Dict[str, float]] = None

    def process_frame(self, frame: npt.NDArray[np.uint8]) -> Tuple[Dict[str, Any], npt.NDArray[np.uint8]]:
        """Process a single frame to detect pose and face landmarks.

        Args:
            frame: Input frame in BGR format

        Returns:
            Tuple containing:
                - Dictionary with detection results
                - Annotated frame with landmarks drawn
        """
        if frame is None:
            raise ValueError("Input frame cannot be None")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be a 3-channel BGR image")

        # Check for empty frame
        if frame.size == 0 or 0 in frame.shape:
            logger.warning("Received empty frame")
            return {"head_pose": None, "pose_landmarks": None, "face_landmarks": None}, frame

        # Log frame info for debugging
        try:
            logger.debug(f"Processing frame - Shape: {frame.shape}, Type: {frame.dtype}, Range: {frame.min()}-{frame.max()}")
        except ValueError as e:
            logger.warning(f"Could not get frame stats: {e}")
            return {"head_pose": None, "pose_landmarks": None, "face_landmarks": None}, frame

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Check if the frame is valid
        if rgb_frame.size == 0:
            logger.error("Empty frame after BGR to RGB conversion")
            return {"error": "Empty frame"}, frame

        # Initialize results dictionary with proper types
        results: Dict[str, Any] = {
            "pose_landmarks": None,
            "face_landmarks": None,
            "head_pose": None,
            "timestamp": None,
        }

        # Initialize variables to store intermediate results
        pose_landmarks_result: Optional[Dict[int, Dict[str, float]]] = None
        face_landmarks_result: Optional[Dict[int, Dict[str, float]]] = None
        head_pose_result: Optional[Dict[str, float]] = None
        annotated_frame = frame.copy()

        try:
            # Process face mesh first to get face landmarks
            logger.debug("Processing face mesh...")
            try:
                face_results = self.face_mesh.process(rgb_frame)
                logger.debug(f"Face mesh results: {face_results}")
                if face_results.multi_face_landmarks:
                    logger.debug(f"Found {len(face_results.multi_face_landmarks)} face(s)")
                    logger.debug(f"First face has {len(face_results.multi_face_landmarks[0].landmark)} landmarks")
                else:
                    logger.debug("No faces detected in the frame")
            except Exception as e:
                logger.error(f"Error in face mesh processing: {str(e)}", exc_info=True)
                face_results = None
            face_landmarks_result = None

            if face_results.multi_face_landmarks:
                logger.debug(f"Found {len(face_results.multi_face_landmarks)} face(s) in the frame")
                if face_results.multi_face_landmarks[0].landmark:
                    logger.debug(f"First face has {len(face_results.multi_face_landmarks[0].landmark)} landmarks")
                    # Just take the first face for now
                    face_landmarks = face_results.multi_face_landmarks[0]
                    face_landmarks_result = self._extract_landmarks(face_landmarks.landmark)
                    results["face_landmarks"] = face_landmarks_result

                    # Calculate head pose using face landmarks
                    logger.debug("Calculating head pose...")
                    head_pose_result = self._calculate_head_pose(face_landmarks.landmark)
                    results["head_pose"] = head_pose_result
                    logger.debug(f"Head pose result: {head_pose_result}")
                else:
                    logger.warning("First face has no landmarks")
            else:
                logger.warning("No faces detected in the frame")

            # Process pose
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks and pose_results.pose_landmarks.landmark:
                pose_landmarks_result = self._extract_landmarks(pose_results.pose_landmarks.landmark)
                results["pose_landmarks"] = pose_landmarks_result

            # Annotate frame
            if pose_results or face_results:
                annotated_frame = self._annotate_frame(frame.copy(), pose_results, face_results)

            # Update instance variables
            self.pose_landmarks = pose_landmarks_result
            self.face_landmarks = face_landmarks_result
            self.head_pose = head_pose_result

            return results, annotated_frame

        except Exception as e:
            logger.error("Error processing frame: %s", str(e))
            # Return the original frame if there's an error
            return results, frame.copy()

    def _extract_landmarks(self, landmarks: Any) -> Dict[int, Dict[str, float]]:
        """Extract landmarks to a dictionary.

        Args:
            landmarks: MediaPipe landmarks object

        Returns:
            Dictionary mapping landmark index to coordinates
        """
        return {
            i: {"x": lm.x, "y": lm.y, "z": getattr(lm, "z", 0), "visibility": getattr(lm, "visibility", 1.0)} for i, lm in enumerate(landmarks)
        }

    def _calculate_pitch_angle(self, landmarks: Any) -> float:
        """Calculate pitch angle from facial landmarks.

        Pitch is calculated using the vertical relationship between facial landmarks:
        - Nose tip (1) and chin (199) for vertical alignment
        - Left (33) and right (263) eye landmarks for horizontal reference

        Args:
            landmarks: MediaPipe face landmarks.

        Returns:
            Pitch angle in degrees. Positive values indicate head up, negative down.
        """
        try:
            # Define landmark indices (MediaPipe Face Mesh)
            NOSE_TIP = 1
            CHIN = 199
            LEFT_EYE = 33
            RIGHT_EYE = 263

            # Get required landmarks
            nose = landmarks[NOSE_TIP]
            chin = landmarks[CHIN]
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]

            # Calculate vertical distance between nose and chin
            vertical_distance = abs(nose.y - chin.y)

            # Calculate horizontal reference (distance between eyes)
            horizontal_reference = abs(left_eye.x - right_eye.x)

            # Avoid division by zero
            if horizontal_reference == 0 or vertical_distance == 0:
                return 0.0

            # Calculate pitch angle using arctangent of the ratio
            # of vertical distance to horizontal reference
            pitch_rad = np.arctan2(vertical_distance, horizontal_reference)

            # Convert to degrees
            pitch_deg = np.degrees(pitch_rad)

            # Determine direction (up or down)
            if nose.y < chin.y:  # Nose is above chin
                pitch_deg = -pitch_deg

            return float(pitch_deg)

        except Exception as e:
            logger.error(f"Error calculating pitch angle: {e}")
            return 0.0

    def _calculate_head_pose(self, landmarks: Any) -> Dict[str, float]:
        """Calculate head pose (pitch, yaw, roll) from facial landmarks.

        This method uses a simplified approach to calculate head pose angles.
        - Pitch is calculated using vertical relationships between facial landmarks
        - Yaw and roll are set to 0.0 as they are not needed for nod detection

        Args:
            landmarks: MediaPipe face landmarks.

        Returns:
            Dictionary containing pitch, yaw, and roll angles in degrees.
        """
        # Initialize results with default values
        results = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        try:
            # Calculate pitch using our simplified method
            results["pitch"] = self._calculate_pitch_angle(landmarks)

            # For nod detection, we only need pitch, so we can set yaw and roll to 0
            results["yaw"] = 0.0
            results["roll"] = 0.0

            logger = logging.getLogger(__name__)
            logger.debug(f"Calculated head pose - Pitch: {results['pitch']:.2f}°")
            return results

        except Exception as e:
            logger.error(f"Error in _calculate_head_pose: {e}")
            return results

    def _annotate_frame(
        self,
        frame: npt.NDArray[np.uint8],
        pose_results: Any,
        face_results: Any,
    ) -> npt.NDArray[np.uint8]:
        """Annotate frame with detected landmarks and pose.

        Args:
            frame: Input frame
            pose_results: MediaPipe pose results
            face_results: MediaPipe face mesh results

        Returns:
            Annotated frame
        """
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        return frame

    def close(self) -> None:
        """Release resources."""
        self.pose.close()
        self.face_mesh.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def save_results_to_json(results: dict[str, Any], output_path: str | Path) -> None:
    """Save detection results to a JSON file.

    Args:
        results: Detection results dictionary
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_numpy_types(x) for x in obj]
        return obj

    # Convert all numpy types in the results
    json_serializable = convert_numpy_types(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_serializable, f, indent=2)


def load_results_from_json(file_path: str | Path) -> dict[str, Any]:
    """Load detection results from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing the loaded results
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
