"""
Pipeline module for the nod detection system.
This module contains the base pipeline and related components.
"""

from .base_pipeline import BasePipeline
from .video_processing_pipeline import VideoProcessingPipeline

__all__ = ["BasePipeline", "VideoProcessingPipeline"]
