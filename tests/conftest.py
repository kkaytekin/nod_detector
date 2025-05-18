"""
Pytest configuration and fixtures for nod_detector tests.
"""

import sys
from pathlib import Path

import pytest

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture to get the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_video_path(test_data_dir):
    """Fixture to get the path to the sample video file."""
    video_path = test_data_dir / "10515012-hd_3840_2160_24fps.mp4"
    if not video_path.exists():
        pytest.skip(f"Sample video not found at {video_path}")
    return str(video_path)
