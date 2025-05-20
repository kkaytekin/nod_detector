"""
Utility functions for saving nod detection results and visualizations.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data class for storing frame information."""

    frame_number: int
    timestamp: float
    pitch: float
    yaw: float = 0.0
    roll: float = 0.0
    nod_detected: bool = False
    nod_direction: Optional[str] = None


@dataclass
class NodEvent:
    """Data class for storing nod event information."""

    frame_number: int
    timestamp: float
    direction: str  # 'up-down' or 'down-up'
    start_pitch: float
    peak_pitch: float
    end_pitch: float
    amplitude: float
    duration_frames: int
    duration_seconds: float


class VideoInfo(TypedDict):
    """Type definition for video information."""

    path: str
    fps: float
    frame_width: int
    frame_height: int
    total_frames: int


class OutputSaver:
    """Handles saving nod detection results to files."""

    def __init__(self, output_dir: Union[str, Path], video_info: VideoInfo) -> None:
        """Initialize the output saver.

        Args:
            output_dir: Directory to save output files
            video_info: Dictionary containing video metadata
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_info = video_info

        # Initialize data storage
        self.frames: List[Dict[str, Any]] = []
        self.nod_events: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

        # Create plots directory
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def add_frame(self, frame_data: FrameData) -> None:
        """Add frame data to be saved.

        Args:
            frame_data: Frame data to add
        """
        self.frames.append(asdict(frame_data))

    def add_nod_event(self, nod_event: NodEvent) -> None:
        """Add a nod event to be saved.

        Args:
            nod_event: Nod event to add
        """
        self.nod_events.append(asdict(nod_event))

    def save_all(self) -> None:
        """Save all output files."""
        self._save_frames_json()
        self._save_nod_events_json()
        self._save_summary_json()
        self._generate_pitch_plot()

    def _save_frames_json(self) -> None:
        """Save frame data to a JSON file."""
        output = {"video_info": self.video_info, "frames": self.frames}
        self._save_json(output, self.output_dir / "frames.json")

    def _save_nod_events_json(self) -> None:
        """Save nod events to a JSON file."""
        if not self.nod_events:
            return

        # Calculate average amplitude and duration
        amplitudes = [e["amplitude"] for e in self.nod_events]
        durations = [e["duration_frames"] for e in self.nod_events]

        output = {
            "total_nods": len(self.nod_events),
            "average_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.0,
            "average_duration_frames": float(np.mean(durations)) if durations else 0.0,
            "nod_events": self.nod_events,
        }
        self._save_json(output, self.output_dir / "nod_events.json")

    def _save_summary_json(self) -> None:
        """Save summary statistics to a JSON file."""
        # Calculate processing time
        processing_time = (datetime.now() - self.start_time).total_seconds()

        # Calculate nod statistics
        if self.nod_events:
            avg_duration = sum(nod["duration_seconds"] for nod in self.nod_events) / len(self.nod_events)
            avg_amplitude = sum(abs(nod["amplitude"]) for nod in self.nod_events) / len(self.nod_events)
        else:
            avg_duration = 0
            avg_amplitude = 0

        # Calculate nod statistics
        if self.nod_events:
            avg_duration = sum(nod["duration_seconds"] for nod in self.nod_events) / len(self.nod_events)
            avg_amplitude = sum(abs(nod["amplitude"]) for nod in self.nod_events) / len(self.nod_events)
        else:
            avg_duration = 0
            avg_amplitude = 0

        output = {
            "video_duration_seconds": self.video_info["total_frames"] / self.video_info["fps"],
            "nod_count": len(self.nod_events),
            "processing_time_seconds": processing_time,
            "processing_speed_fps": len(self.frames) / processing_time if processing_time > 0 else 0,
            "first_frame_timestamp": self.frames[0]["timestamp"] if self.frames else 0,
            "last_frame_timestamp": self.frames[-1]["timestamp"] if self.frames else 0,
            "average_fps": len(self.frames) / processing_time if processing_time > 0 and self.frames else 0,
            "nod_statistics": {
                "total_nods": len(self.nod_events),
                "avg_nod_duration_seconds": avg_duration,
                "avg_nod_amplitude": avg_amplitude,
            },
        }

        # Save to JSON file
        self._save_json(output, self.output_dir / "summary.json")

    def _generate_pitch_plot(self) -> None:
        """Generate and save a plot of pitch angle over time."""
        if not self.frames:
            return

        try:
            import matplotlib.pyplot as plt

            timestamps = [frame["timestamp"] for frame in self.frames]
            pitches = [frame["pitch"] for frame in self.frames]

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, pitches, label="Pitch Angle")
            plt.xlabel("Time (s)")
            plt.ylabel("Pitch Angle (degrees)")
            plt.title("Head Pitch Angle Over Time")
            plt.grid(True, linestyle="--", alpha=0.7)

            # Add markers for detected nods
            for event in self.nod_events:
                plt.axvline(x=event["timestamp"], color="r", linestyle=":", alpha=0.5)
                plt.text(
                    event["timestamp"],
                    max(pitches) * 0.9,
                    f"{event['direction']}",
                    rotation=90,
                    va="top",
                    ha="right",
                )

            plt.legend()

            # Save the plot
            plt.savefig(self.plots_dir / "pitch_plot.png", dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved pitch plot to {self.plots_dir / 'pitch_plot.png'}")

        except ImportError as e:
            logger.warning(f"Could not import matplotlib: {e}")
        except Exception as e:
            logger.warning(f"Error generating pitch plot: {e}")

    @staticmethod
    def _save_json(data: Any, path: Path) -> None:
        """Save data to a JSON file with nice formatting.

        Args:
            data: Data to save (must be JSON serializable)
            path: Path to save the file
        """

        def default_serializer(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=default_serializer)
            logger.info("Saved output to %s", path)
        except Exception as e:
            logger.error("Failed to save %s: %s", path, str(e), exc_info=True)


def save_processed_video(frames: List[npt.NDArray[np.uint8]], output_path: Union[str, Path], fps: float, frame_size: Tuple[int, int]) -> None:
    """Save processed frames as a video file.

    Args:
        frames: List of video frames as numpy arrays in BGR format
        output_path: Path to save the output video
        fps: Frames per second for the output video
        frame_size: Tuple of (width, height) for the output video
    """
    output_path = Path(output_path)
    logger.debug(f"Preparing to save video to: {output_path.absolute()}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate frames
    if not frames:
        logger.error("No frames provided to save")
        return

    # Log frame information
    logger.debug(f"Number of frames: {len(frames)}")
    logger.debug(f"Frame size: {frame_size}")
    logger.debug(f"FPS: {fps}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path), fourcc, float(fps), (int(frame_size[0]), int(frame_size[1]))  # Ensure fps is float  # Ensure frame size is int
    )

    if not out.isOpened():
        logger.error(f"Could not open video writer for file: {output_path}")
        return

    try:
        # Write frames
        for i, frame in enumerate(frames):
            # Ensure frame is in BGR format and has correct dimensions
            if frame is None:
                logger.warning(f"Frame {i} is None, skipping")
                continue

            if len(frame.shape) == 2:  # Convert grayscale to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # Convert RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 1:  # Single channel to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Resize frame if necessary
            if (frame.shape[1], frame.shape[0]) != (frame_size[0], frame_size[1]):
                frame = cv2.resize(frame, frame_size)

            out.write(frame)

        logger.info(f"Successfully wrote {len(frames)} frames to {output_path}")

    except Exception as e:
        logger.error(f"Error writing video frames: {e}", exc_info=True)

    finally:
        try:
            out.release()
            logger.debug("Video writer released successfully")
        except Exception as e:
            logger.error(f"Error releasing video writer: {e}", exc_info=True)
