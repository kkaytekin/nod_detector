"""
Video processing pipeline for the nod detection system.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr

from ..mediapipe_components import MediaPipeDetector
from ..nod_detection import NodDetector
from ..output_utils import save_processed_video
from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class FrameResult(TypedDict, total=False):
    """Type definition for frame processing results."""

    frame_number: int
    timestamp: float
    detections: List[Any]
    head_pose: Optional[Dict[str, float]]
    pose_landmarks: Optional[Dict[str, Any]]
    face_landmarks: Optional[Dict[str, Any]]
    nod_detected: bool


class VideoInfo(TypedDict):
    """Type definition for video information."""

    path: str
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int


class ProcessingResults(Dict[str, Any]):
    """Type definition for processing results.

    This extends Dict[str, Any] to satisfy the type constraint in BasePipeline.
    The actual structure is defined by the TypedDict below.
    """

    pass


class ProcessingResultsDict(TypedDict):
    """Typed dictionary for processing results."""

    video_info: VideoInfo
    frame_results: List[FrameResult]
    detections: List[Any]


class VideoProcessingPipeline(BasePipeline["ProcessingResults"]):
    """Pipeline for processing video frames for nod detection.

    This pipeline handles:
    - Video frame extraction
    - Frame preprocessing
    - Person detection (to be implemented)
    - Head pose estimation (to be implemented)
    - Nod detection (to be implemented)
    - Result visualization using rerun.io
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the video processing pipeline.

        Args:
            config: Configuration dictionary for the pipeline.
        """
        super().__init__()
        self.config = config or {}
        self.frame_count = 0
        self.detector = MediaPipeDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False,
            refine_face_landmarks=True,
        )
        # Initialize NodDetector with configuration from config or use defaults
        self.nod_detector = NodDetector(
            min_amplitude=float(self.config.get("min_amplitude", 5.0)),
            min_peak_distance=int(self.config.get("min_peak_distance", 30)),
            fps=int(self.config.get("fps", 30)),
        )
        self.output_dir = Path(self.config.get("output_dir", "output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nod_count = 0
        self.last_nod_frame = -10  # Track the last frame where a nod was detected
        self.nod_display_frames = 5  # Number of frames to show nod detection

    def _visualize_frame(
        self,
        frame: npt.NDArray[np.uint8],  # type: ignore[type-arg]  # OpenCV frame type
        frame_number: int,
        video_info: VideoInfo,
        results: Optional[FrameResult] = None,
    ) -> None:
        """Visualize the frame with keypoints and results.

        Args:
            frame: The video frame to visualize as a numpy array of uint8.
            frame_number: Current frame number.
            video_info: Dictionary containing video metadata.
            results: Optional frame processing results.
        """
        if results is None:
            return

        # Set time for rerun visualization
        rr.set_time_seconds("time", frame_number / video_info["fps"])
        rr.set_time_sequence("frame", frame_number)

        # Create a copy of the frame to draw on
        if frame is None or frame.size == 0:
            return

        frame_copy = frame.copy()
        height, width = frame_copy.shape[:2]

        # Draw pose landmarks if available
        if "pose_landmarks" in results and results["pose_landmarks"]:
            pose_landmarks = results["pose_landmarks"]
            if isinstance(pose_landmarks, dict):
                # Draw connections between keypoints
                connections = [
                    # Face outline
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (0, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    # Torso
                    (11, 12),
                    (11, 23),
                    (23, 24),
                    (12, 24),
                    # Left arm
                    (11, 13),
                    (13, 15),
                    (15, 17),
                    (15, 19),
                    (15, 21),
                    (17, 19),
                    # Right arm
                    (12, 14),
                    (14, 16),
                    (16, 18),
                    (16, 20),
                    (16, 22),
                    (18, 20),
                    # Left leg
                    (23, 25),
                    (25, 27),
                    (27, 29),
                    (29, 31),
                    (27, 31),
                    # Right leg
                    (24, 26),
                    (26, 28),
                    (28, 30),
                    (28, 32),
                    (30, 32),
                ]

                # Draw connections
                for start_idx, end_idx in connections:
                    if str(start_idx) in pose_landmarks and str(end_idx) in pose_landmarks:
                        start = pose_landmarks[str(start_idx)]
                        end = pose_landmarks[str(end_idx)]
                        if start and end:
                            start_pt = (
                                int(start["x"] * width),
                                int(start["y"] * height),
                            )
                            end_pt = (int(end["x"] * width), int(end["y"] * height))
                            cv2.line(frame_copy, start_pt, end_pt, (0, 255, 0), 2)

                # Draw keypoints
                for idx, landmark in pose_landmarks.items():
                    if isinstance(landmark, dict):
                        # Convert normalized coordinates to pixel coordinates
                        x = int(landmark.get("x", 0) * width)
                        y = int(landmark.get("y", 0) * height)

                        # Draw keypoints with different colors for different body parts
                        color = (0, 255, 0)  # Green for body
                        if int(idx) in [11, 12, 13, 14, 15, 16]:  # Shoulders and elbows
                            color = (255, 255, 0)  # Cyan for arms
                        elif int(idx) in [23, 24, 25, 26, 27, 28]:  # Hips and knees
                            color = (0, 255, 255)  # Yellow for legs

                        cv2.circle(frame_copy, (x, y), 5, color, -1)

        # Draw face landmarks if available
        if "face_landmarks" in results and results["face_landmarks"]:
            face_landmarks = results["face_landmarks"]
            if isinstance(face_landmarks, dict):
                # Draw face mesh
                connections = [
                    # Face oval
                    (10, 338),
                    (338, 297),
                    (297, 332),
                    (332, 284),
                    (284, 251),
                    (251, 389),
                    (389, 356),
                    (356, 454),
                    (454, 323),
                    (323, 361),
                    (361, 288),
                    (288, 397),
                    (397, 365),
                    (365, 379),
                    (379, 378),
                    (378, 400),
                    (400, 377),
                    (377, 152),
                    (152, 148),
                    (148, 176),
                    (176, 149),
                    (149, 150),
                    (150, 136),
                    (136, 172),
                    (172, 58),
                    (58, 132),
                    (132, 93),
                    (93, 234),
                    (234, 127),
                    (127, 162),
                    (162, 21),
                    (21, 54),
                    (54, 103),
                    (103, 67),
                    (67, 109),
                    (109, 10),
                ]

                for start_idx, end_idx in connections:
                    if str(start_idx) in face_landmarks and str(end_idx) in face_landmarks:
                        start = face_landmarks[str(start_idx)]
                        end = face_landmarks[str(end_idx)]
                        if start and end:
                            start_pt = (
                                int(start["x"] * width),
                                int(start["y"] * height),
                            )
                            end_pt = (int(end["x"] * width), int(end["y"] * height))
                            cv2.line(frame_copy, start_pt, end_pt, (0, 0, 255), 1)

                # Draw face keypoints
                for _landmark in face_landmarks.values():
                    if isinstance(_landmark, dict):
                        x = int(_landmark.get("x", 0) * width)
                        y = int(_landmark.get("y", 0) * height)
                        cv2.circle(frame_copy, (x, y), 1, (0, 0, 255), -1)

        # Add status text with background for better visibility
        if "nod_detected" in results:
            nod_direction = results.get("nod_direction", "")
            if results["nod_detected"]:
                direction = nod_direction.upper() if isinstance(nod_direction, str) else ""
                status = f"NOD DETECTED: {direction}"
                color = (0, 0, 255)  # Red for detected nod
            else:
                status = "No nod detected"
                color = (0, 255, 0)  # Green for no nod

            # Add nod count to status
            status = f"Nods: {self.nod_count} | {status}"

            # Add background for status text
            text_size = cv2.getTextSize(f"Status: {status}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.rectangle(frame_copy, (10, 10), (30 + text_size[0], 50), (0, 0, 0), -1)

            # Add status text
            cv2.putText(
                frame_copy,
                f"Status: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )

        # Add frame info with background
        info_text = f"Frame: {frame_number}/{video_info['total_frames']} | FPS: {video_info['fps']:.1f}"
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(
            frame_copy,
            (10, frame_copy.shape[0] - 30),
            (20 + text_size[0], frame_copy.shape[0] - 10),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame_copy,
            info_text,
            (20, frame_copy.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _process(self, input_data: str, **kwargs: Any) -> ProcessingResults:
        """Process the input video and detect nodding behavior.

        Args:
            input_data: Path to the input video file.
            **kwargs: Additional keyword arguments.
                visualize: Whether to visualize the results. Defaults to False.
                debug: If True, process only the first 10 frames. Defaults to False.

        Returns:
            ProcessingResults containing the processing results.

        Raises:
            IOError: If the input video file cannot be opened.
        """
        video_path = input_data
        visualize = kwargs.get("visualize", False)
        debug = kwargs.get("debug", False)

        # Initialize Rerun for visualization if enabled
        if visualize:
            video_name = Path(video_path).stem
            try:
                # Initialize Rerun with more detailed settings
                rr.init(
                    application_id=f"NodDetector/{video_name}",
                    spawn=True,
                    default_enabled=True,
                )
                logger.info("Rerun visualization initialized. Open the Rerun viewer to see the results.")

            except Exception as e:
                logger.error(f"Failed to initialize Rerun: {e}")
                visualize = False  # Disable visualization if initialization fails

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_info: VideoInfo = {
            "path": video_path,
            "fps": float(fps),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "total_frames": total_frames,
        }

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {frame_width}x{frame_height} at {fps:.2f} FPS")
        logger.info(f"Total frames: {total_frames}")

        # Prepare the final results dictionary with proper typing
        results: ProcessingResultsDict = {
            "video_info": video_info,
            "frame_results": [],
            "detections": [],
        }

        # List to store processed frames for video output
        processed_frames: List[npt.NDArray[np.uint8]] = []

        # Set max frames based on debug mode
        max_frames = 10 if debug else float("inf")
        if debug:
            logger.info(f"Debug mode: Processing only first {max_frames} frames")
        else:
            logger.info("Processing entire video")

        try:
            frame_count = 0
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Process the frame
                    frame_result, processed_frame = self.detector.process_frame(frame)
                    frame_result["frame_number"] = frame_count
                    frame_result["timestamp"] = frame_count / fps

                    # Update nod detection using NodDetector
                    head_pose = frame_result.get("head_pose", {})
                    if isinstance(head_pose, dict):
                        pitch = head_pose.get("pitch", 0.0) if isinstance(head_pose.get("pitch"), (int, float)) else 0.0

                        # Update nod detector with current pitch and frame number
                        nod_detected, nod_direction = self.nod_detector.update(pitch, frame_count)

                        # Update frame result with nod detection info
                        frame_result["nod_detected"] = nod_detected
                        frame_result["nod_direction"] = nod_direction if nod_detected else ""
                        frame_result["pitch"] = pitch  # Ensure pitch is in frame_result

                        # Save frame results to JSON file
                        self._save_frame_results(frame_result, frame_count)

                        # Update nod count
                        if nod_detected:
                            self.nod_count += 1
                            logger.info(f"Nod detected at frame {frame_count}: {nod_direction}")

                        # Update last nod frame if nod was detected
                        if nod_detected:
                            self.last_nod_frame = frame_count

                    # Create a copy of the frame for visualization
                    vis_frame = frame.copy()

                    # Convert processed frame to BGR if needed
                    if len(processed_frame.shape) == 2:  # Grayscale
                        vis_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                    elif processed_frame.shape[2] == 4:  # RGBA
                        vis_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)
                    elif processed_frame.shape[2] == 1:  # Single channel
                        vis_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                    else:  # Already BGR or other 3-channel format
                        vis_frame = processed_frame.copy()

                    # Calculate text scale and thickness based on frame dimensions
                    frame_height = vis_frame.shape[0]
                    text_scale = frame_height / 500  # Scale text size based on frame height
                    text_thickness = max(1, int(frame_height / 400))  # Scale thickness

                    # Visualize the frame with landmarks and results
                    self._visualize_frame(vis_frame, frame_count, video_info, frame_result)

                    # Visualize with Rerun if enabled
                    if visualize:
                        self._visualize(vis_frame, frame_result)

                    # Show nod detection indicator for several frames after detection
                    if frame_count - self.last_nod_frame < self.nod_display_frames:
                        cv2.putText(
                            vis_frame,
                            f"NOD DETECTED: {nod_direction.upper()}",
                            (int(frame_height * 0.05), int(frame_height * 0.1)),  # Position relative to frame height
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_scale * 1.5,  # Larger text for nod detection
                            (0, 0, 255),  # Red color
                            text_thickness * 2,  # Thicker text
                            cv2.LINE_AA,
                        )

                    # Add nod count (top left)
                    cv2.putText(
                        vis_frame,
                        f"NOD COUNT: {self.nod_count}",
                        (int(frame_height * 0.03), int(frame_height * 0.05)),  # Top left
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale,
                        (0, 255, 0),  # Green color
                        text_thickness,
                        cv2.LINE_AA,
                    )

                    # Add frame number (top right)
                    frame_text = f"FRAME: {frame_count}"
                    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, text_thickness)[0]
                    text_x = vis_frame.shape[1] - text_size[0] - int(frame_height * 0.03)  # Right align
                    cv2.putText(
                        vis_frame,
                        frame_text,
                        (text_x, int(frame_height * 0.05)),  # Top right
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_scale * 0.8,
                        (255, 255, 255),  # White color
                        text_thickness,
                        cv2.LINE_AA,
                    )

                    # Ensure frame has correct dimensions
                    if (vis_frame.shape[1], vis_frame.shape[0]) != (frame_width, frame_height):
                        vis_frame = cv2.resize(vis_frame, (frame_width, frame_height))

                    # Add the frame to the list for video output
                    processed_frames.append(vis_frame)
                    frame_count += 1

                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        logger.info(f"Collected {frame_count}/{total_frames} frames for output video")

                    # Log progress every 100 frames
                    if frame_count % 100 == 0:
                        logger.info(f"Processed {frame_count}/{total_frames} frames")

                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
                    # Create an error result for this frame
                    error_result = self._get_error_result(frame_count, e)
                    results["frame_results"].append(error_result)
                    continue

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise
        finally:
            try:
                # Release resources
                if "cap" in locals() and cap.isOpened():
                    cap.release()

                # Save the processed video
                if "processed_frames" in locals() and processed_frames:
                    output_video_path = self.output_dir / f"{Path(video_path).stem}_processed.mp4"
                    logger.info(f"Saving processed video to: {output_video_path}")
                    logger.info(f"Number of frames to save: {len(processed_frames)}")
                    logger.info(f"Frame size: {frame_width}x{frame_height}")
                    logger.info(f"FPS: {fps}")

                    # Ensure output directory exists
                    self.output_dir.mkdir(parents=True, exist_ok=True)

                    # Save the video
                    try:
                        save_processed_video(
                            frames=processed_frames, output_path=output_video_path, fps=fps, frame_size=(frame_width, frame_height)
                        )
                        logger.info(f"Successfully saved processed video to: {output_video_path}")

                        # Verify the output file was created
                        if output_video_path.exists():
                            logger.info(f"Output video size: {output_video_path.stat().st_size / (1024 * 1024):.2f} MB")
                        else:
                            logger.error("Output video file was not created")

                    except Exception as e:
                        logger.error(f"Error saving video: {e}", exc_info=True)

                if visualize:
                    try:
                        # Add a small delay to ensure all frames are processed
                        import time

                        time.sleep(1)

                        # Try to close Rerun if possible
                        try:
                            if hasattr(rr, "disconnect"):
                                rr.disconnect()
                            elif hasattr(rr, "shutdown"):
                                rr.shutdown()
                        except Exception as e:
                            logger.debug(f"Could not properly close Rerun: {e}")

                    except Exception as e:
                        logger.warning(f"Error during cleanup: {e}")
                    finally:
                        # Always try to clean up OpenCV windows
                        cv2.destroyAllWindows()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")

        # Convert to ProcessingResults type to satisfy the return type
        return ProcessingResults(results)

    def _save_frame_results(self, frame_result: FrameResult, frame_number: int) -> None:
        """Save frame results to a JSON file.

        Args:
            frame_result: Results from frame processing.
            frame_number: Current frame number.
        """
        output_file = self.output_dir / f"frame_{frame_number:06d}.json"
        with open(output_file, "w") as f:
            json.dump(frame_result, f, indent=2)

    def _get_error_result(self, frame_number: int, error: Exception) -> FrameResult:
        """Create an error result dictionary.

        Args:
            frame_number: Current frame number
            error: The exception that occurred

        Returns:
            FrameResult with error information
        """
        error_result = FrameResult(
            frame_number=frame_number,
            timestamp=0.0,
            detections=[],
            head_pose=None,
            pose_landmarks=None,
            face_landmarks=None,
            nod_detected=False,
        )
        self._save_frame_results(error_result, frame_number)
        return error_result

    def _visualize(self, frame: npt.NDArray[np.uint8], frame_result: FrameResult) -> None:
        """Visualize the processing results using Rerun.

        Args:
            frame: Input frame.
            frame_result: Results from frame processing.
        """
        if frame is None or frame.size == 0:
            return

        frame_number = frame_result["frame_number"]

        try:
            # Log the raw frame to Rerun
            rr.set_time_sequence("frame", frame_number)

            # Convert from BGR (OpenCV) to RGB (Rerun)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Log the RGB frame to Rerun
            rr.log("video/rgb", rr.Image(rgb_frame))

            # Log head pose information
            head_pose = frame_result.get("head_pose")
            if head_pose:
                pitch = head_pose.get("pitch", 0.0)

                # Log head pose data as scalars
                rr.log("metrics/pitch_abs", rr.Scalar(pitch))

                # Log nod detection status and count
                nod_detected = frame_result.get("nod_detected", False)
                nod_direction = self.nod_detector.get_last_direction()
                nod_count = self.nod_detector.get_nod_count()

                # Update the nod count display
                rr.log("metrics/nod_count", rr.Scalar(nod_count))

                # Log nod detection event
                if nod_detected:
                    rr.log(
                        "text/nod_detected",
                        rr.TextLog(f"Nod detected: {nod_direction}"),
                    )

                # Create a more informative status text
                status_text = f"Frame: {frame_number}\n" f"Pitch: {pitch:.1f}°\n" f"Nod Count: {nod_count}"

                # Add detection status
                if nod_detected and nod_direction:
                    status_text += f"\nNOD DETECTED: {nod_direction.upper()}"

                # Add current direction
                if hasattr(self.nod_detector, "current_direction"):
                    current_direction = self.nod_detector.current_direction
                    if current_direction:
                        status_text += f"\nDirection: {current_direction}"

                # Log the status text
                rr.log("text/status", rr.TextLog(status_text))

                # Log pitch history for visualization
                if hasattr(self.nod_detector, "pitch_history") and self.nod_detector.pitch_history:
                    # Create time points for the pitch history
                    history_length = len(self.nod_detector.pitch_history)
                    time_points = list(range(max(0, history_length - 100), history_length))
                    pitch_values = self.nod_detector.pitch_history[-len(time_points) :]

                    # Log as a time series
                    for t, p in zip(time_points, pitch_values):
                        rr.set_time_sequence("pitch_history", t)
                        rr.log("metrics/pitch_window", rr.Scalar(p))

        except Exception as e:
            logger.warning(f"Error in visualization: {e}")
        finally:
            # Always try to clean up OpenCV windows
            cv2.destroyAllWindows()

    def reset(self) -> None:
        """Reset the pipeline state."""
        super().reset()
        self.frame_count = 0
        self.nod_count = 0
        self.nod_detector.reset()
