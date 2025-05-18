"""Nod Detector - A system for detecting nodding behavior in videos.

This package provides tools for detecting nodding behavior in video files using
computer vision and machine learning techniques. It includes a command-line
interface for easy usage and can be integrated into other Python applications.

Example:
    To use the command-line interface:
    ```bash
    python -m nod_detector --help
    ```

    Or programmatically:
    ```python
    from nod_detector.pipeline import VideoProcessingPipeline

    pipeline = VideoProcessingPipeline()
    results = pipeline.process("path/to/video.mp4")
    ```
"""

__version__ = "0.1.0"

# Import key components to make them available at package level
from nod_detector.pipeline import VideoProcessingPipeline  # noqa: F401

__all__ = ["VideoProcessingPipeline"]
