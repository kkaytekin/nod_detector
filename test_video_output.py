"""
Test script to verify video output functionality.
"""

from pathlib import Path

import cv2
import numpy as np

from nod_detector.output_utils import save_processed_video


def create_test_video():
    """Create a simple test video with a moving rectangle."""
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Create a simple video with a moving rectangle
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    num_frames = fps * duration

    # Create frames
    frames = []
    for i in range(num_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw a moving rectangle
        x = int((i / num_frames) * (width - 100))
        cv2.rectangle(frame, (x, 200), (x + 100, 300), (0, 255, 0), -1)

        # Add frame number
        cv2.putText(frame, f"Frame {i + 1}/{num_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frames.append(frame)

    # Save the video
    output_path = output_dir / "test_output.mp4"
    save_processed_video(frames=frames, output_path=output_path, fps=fps, frame_size=(width, height))
    print(f"Test video saved to: {output_path}")


if __name__ == "__main__":
    create_test_video()
