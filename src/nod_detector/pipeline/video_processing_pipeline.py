"""
Video processing pipeline for the nod detection system.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class FrameResult(TypedDict, total=False):
    """Type definition for frame processing results."""

    frame_number: int
    detections: List[Any]
    head_pose: Optional[Dict[str, float]]
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

    def _visualize_frame(
        self,
        frame: npt.NDArray[np.uint8],
        frame_number: int,
        video_info: VideoInfo,
        results: Optional[FrameResult] = None,
    ) -> None:
        """Visualize the frame and results using rerun.io.

        Args:
            frame: The video frame to visualize.
            frame_number: Current frame number.
            video_info: Dictionary containing video metadata.
            results: Optional frame processing results.
        """
        # Log the RGB frame to rerun
        rr.set_time_seconds("time", time.time())
        rr.set_time_sequence("frame", frame_number)
        # Convert BGR to RGB for visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log("video/frame", rr.Image(frame_rgb))
        # Log video info as text
        info_text = f"""
        Frame: {frame_number}/{video_info['total_frames']}
        FPS: {video_info['fps']:.2f}
        Resolution: {video_info['frame_width']}x{video_info['frame_height']}
        """
        rr.log("video/info", rr.TextDocument(info_text.strip()))
        # Log head pose if available
        if results and "head_pose" in results and results["head_pose"]:
            pose = results["head_pose"]
            rr.log(
                "head_pose/direction",
                rr.Arrows3D(
                    origins=[[0, 0, 0]],
                    vectors=[[pose.get("pitch", 0), pose.get("yaw", 0), pose.get("roll", 0)]],
                    colors=[[255, 0, 0]],
                    radii=0.1,
                ),
            )

    def _process(self, input_data: Any, **kwargs: Any) -> ProcessingResults:
        """
        Process the input video to detect nodding behavior.

        Args:
            input_data: Input data to process (expected to be a video path).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict containing processing results.
        """
        video_path = str(input_data)  # Convert input to string

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize rerun
            video_name = Path(video_path).stem
            rr.init(f"Nod Detector - {video_name}", spawn=True)
            # Set up the visualization
            rr.log("world", rr.ViewCoordinates.RDF)  # Right-Down-Forward for computer vision
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Initialize results dictionary with proper typing
            results_dict: ProcessingResultsDict = {
                "video_info": {
                    "path": video_path,
                    "fps": fps,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "total_frames": total_frames,
                },
                "frame_results": [],
                "detections": [],
            }

            # Create the ProcessingResults instance
            results = ProcessingResults(results_dict)

            # Process each frame
            frame_results: List[FrameResult] = []
            self.frame_count = 0
            # Create video info dictionary
            video_info: VideoInfo = {
                "path": video_path,
                "fps": fps,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "total_frames": total_frames,
            }

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame (to be implemented)
                frame_result: FrameResult = {
                    "frame_number": self.frame_count,
                    "detections": [],
                    "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},  # Placeholder - will be updated with actual values
                    "nod_detected": False,
                }
                # Visualize the current frame with results
                self._visualize_frame(frame, self.frame_count, video_info, frame_result)
                frame_results.append(frame_result)
                self.frame_count += 1
                # Add a small delay to keep the visualization at the video's frame rate
                time.sleep(1.0 / (fps * 1.5))  # Slightly faster than real-time

            return results

        except Exception as e:
            logger.error("Error processing frame %d: %s", self.frame_count, str(e))
            raise
        finally:
            cap.release()

    def _process_frame(self, frame: npt.NDArray[np.uint8], frame_number: int) -> FrameResult:
        """Process a single frame.

        Args:
            frame: Input frame as a numpy array.
            frame_number: Current frame number.

        Returns:
            Dictionary containing frame processing results.
        """
        # TODO: Implement frame processing logic
        return {
            "frame_number": frame_number,
            "detections": [],
            "head_pose": None,
            "nod_detected": False,
        }

    def _visualize(self, frame: npt.NDArray[np.uint8], frame_result: FrameResult) -> None:
        """Visualize the processing results on the frame.

        Args:
            frame: Input frame.
            frame_result: Results from frame processing.
        """
        # TODO: Implement visualization
        # This method intentionally left blank as visualization is not yet implemented
        pass

    def reset(self) -> None:
        """Reset the pipeline state."""
        super().reset()
        self.frame_count = 0
