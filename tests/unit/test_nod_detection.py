"""Unit tests for the NodDetector class."""

from src.nod_detector.nod_detection import NodDetector


def test_initialization():
    """Test NodDetector initialization with default parameters."""
    detector = NodDetector()
    assert detector.get_nod_count() == 0
    assert detector.get_last_direction() is None


def test_down_up_nod_detection():
    """Test detection of a down-up nod."""
    detector = NodDetector(min_amplitude=2.0, min_peak_distance=5, fps=30)

    # Down motion (pitch increasing)
    for i in range(5):
        pitch = i * 2  # 0, 2, 4, 6, 8
        detected, _ = detector.update(pitch, i)

    # Up motion (pitch decreasing)
    nod_detected = False
    for i in range(5, 10):
        pitch = 16 - i * 2  # 6, 4, 2, 0, -2
        detected, direction = detector.update(pitch, i)

        # Should detect a nod at the peak (around frame 5)
        if detected:
            assert direction == "down-up"
            nod_detected = True

    assert nod_detected, "No nod was detected in the expected range"

    assert detector.get_nod_count() == 1


def test_up_down_nod_detection():
    """Test detection of an up-down nod."""
    detector = NodDetector(min_amplitude=2.0, min_peak_distance=5, fps=30)

    # Up motion (pitch decreasing)
    for i in range(5):
        pitch = 10 - i * 2  # 10, 8, 6, 4, 2
        detected, _ = detector.update(pitch, i)

    # Down motion (pitch increasing)
    nod_detected = False
    for i in range(5, 10):
        pitch = i * 2 - 8  # 2, 4, 6, 8, 10
        detected, direction = detector.update(pitch, i)

        # Should detect a nod at the trough (around frame 5)
        if detected:
            assert direction == "up-down"
            nod_detected = True

    assert nod_detected, "No nod was detected in the expected range"

    assert detector.get_nod_count() == 1


def test_min_amplitude_threshold():
    """Test that nods below the minimum amplitude are ignored."""
    # Set a high amplitude threshold
    detector = NodDetector(min_amplitude=20.0, min_peak_distance=5, fps=30)

    # Generate a small nod (amplitude = 4)
    for i in range(5):
        pitch = i  # 0, 1, 2, 3, 4
        detected, _ = detector.update(pitch, i)
    for i in range(5, 10):
        pitch = 8 - i  # 3, 2, 1, 0, -1
        detected, _ = detector.update(pitch, i)

    assert detector.get_nod_count() == 0


def test_min_peak_distance():
    """Test that nods too close together are ignored."""
    detector = NodDetector(min_amplitude=2.0, min_peak_distance=10, fps=30)

    # First nod (should be detected)
    for i in range(5):
        pitch = i * 2  # 0, 2, 4, 6, 8
        detector.update(pitch, i)
    for i in range(5, 10):
        pitch = 16 - i * 2  # 6, 4, 2, 0, -2
        detector.update(pitch, i)

    # Second nod too soon (should be ignored)
    for i in range(10, 13):  # Only go up to frame 12 (distance 8 < 10)
        pitch = i * 2 - 20  # 0, 2, 4
        detector.update(pitch, i)
    for i in range(13, 18):  # Start going down before completing the nod
        pitch = 24 - i * 2  # -2, -4, -6, -8, -10
        detector.update(pitch, i)

    # Third nod after enough time has passed (should be detected)
    for i in range(18, 23):
        pitch = i * 2 - 40  # -4, -2, 0, 2, 4
        detector.update(pitch, i)
    for i in range(23, 28):
        pitch = 50 - i * 2  # 4, 2, 0, -2, -4
        detector.update(pitch, i)

    assert detector.get_nod_count() == 2, "Expected exactly two nods (one at the start and one at the end)"


def test_reset():
    """Test resetting the detector state."""
    detector = NodDetector(min_amplitude=2.0, min_peak_distance=5, fps=30)

    # Generate and detect a nod
    for i in range(5):
        pitch = i * 2  # 0, 2, 4, 6, 8
        detector.update(pitch, i)
    for i in range(5, 10):
        pitch = 16 - i * 2  # 6, 4, 2, 0, -2
        detector.update(pitch, i)

    assert detector.get_nod_count() == 1

    # Reset and verify state is cleared
    detector.reset()
    assert detector.get_nod_count() == 0
    assert detector.get_last_direction() is None

    # Should be able to detect nods again
    for i in range(5):
        pitch = i * 2  # 0, 2, 4, 6, 8
        detector.update(pitch, i)
    for i in range(5, 10):
        pitch = 16 - i * 2  # 6, 4, 2, 0, -2
        detector.update(pitch, i)

    assert detector.get_nod_count() == 1
