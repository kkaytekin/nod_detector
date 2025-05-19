"""Nod detection module for detecting head nods in video frames."""

from typing import List, Optional, Tuple


class NodDetector:
    """
    A class to detect nodding behavior based on pitch angle changes over time.

    The detector looks for patterns of up-down or down-up head movements
    that exceed a minimum amplitude and are sufficiently separated in time.
    """

    def __init__(self, min_amplitude: float = 5.0, min_peak_distance: int = 30, fps: int = 30) -> None:  # in frames
        """
        Initialize the NodDetector.

        Args:
            min_amplitude: Minimum pitch change (in degrees) to consider as a nod
            min_peak_distance: Minimum frames between detected nods
            fps: Frames per second of the video (used for time-based calculations)
        """
        self.min_amplitude = min_amplitude
        self.min_peak_distance = min_peak_distance
        self.fps = fps

        # State variables
        self.pitch_history: List[float] = []
        self.velocity_history: List[float] = []
        self.last_extremum: Optional[str] = None  # 'min' or 'max'
        self.last_extremum_value: float = 0.0
        self.last_extremum_frame: int = 0
        self.last_nod_frame: int = 0
        self.last_direction: Optional[str] = None  # 'up' or 'down'
        self.current_direction: Optional[str] = None
        self.nod_count: int = 0
        self.last_detection_direction: Optional[str] = None

    def _calculate_velocity(self, current_pitch: float, previous_pitch: float) -> float:
        """Calculate the velocity between two consecutive pitch values."""
        return current_pitch - previous_pitch

    def _detect_direction_change(self, current_frame: int) -> Tuple[bool, str]:
        """
        Detect if there's a direction change in the pitch movement.

        Args:
            current_frame: Current frame number

        Returns:
            Tuple[bool, str]: (detected, direction)
        """
        if len(self.velocity_history) < 2 or len(self.pitch_history) < 3:
            return False, ""

        current_vel = self.velocity_history[-1]
        _ = self.velocity_history[-2]  # Used in debug print below
        current_pitch = self.pitch_history[-1]

        # Calculate the pitch change from the last extremum or start
        if self.last_extremum_frame > 0:
            pitch_change = abs(current_pitch - self.last_extremum_value)
            frames_since_last_nod = current_frame - self.last_extremum_frame
            too_close = frames_since_last_nod < self.min_peak_distance
        else:
            pitch_change = abs(current_pitch - self.pitch_history[0])
            too_close = False

        # Debug output
        print(
            f"Frame {current_frame}: pitch={current_pitch:.1f}, vel={current_vel:.1f}, "
            f"pitch_change={pitch_change:.1f}, last_extremum={self.last_extremum}"
        )

        if too_close:
            print(
                f"  Too close to last nod at frame {self.last_extremum_frame} " f"(distance={frames_since_last_nod} < {self.min_peak_distance})"
            )
            return False, ""

        # For testing purposes, detect direction changes more aggressively
        if len(self.pitch_history) >= 3:
            # Check for local maximum (down-up nod)
            is_local_max = self.pitch_history[-2] > self.pitch_history[-3] and self.pitch_history[-2] >= self.pitch_history[-1]

            # Check for local minimum (up-down nod)
            is_local_min = self.pitch_history[-2] < self.pitch_history[-3] and self.pitch_history[-2] <= self.pitch_history[-1]

            print(
                f"  is_local_max={is_local_max}, is_local_min={is_local_min}, "
                f"pitch_change={pitch_change:.1f} >= {self.min_amplitude}? {pitch_change >= self.min_amplitude}"
            )

            # Only detect a new extremum if we've moved enough from the last one
            if pitch_change >= self.min_amplitude:
                if is_local_max and (self.last_extremum != "max" or current_frame - self.last_extremum_frame > 1):
                    # Only detect a new nod if we're not too close to the last nod
                    if current_frame - self.last_nod_frame >= self.min_peak_distance or self.last_nod_frame == 0:
                        print(f"  Detected DOWN-UP nod at frame {current_frame - 1}")
                        self.last_extremum = "max"
                        self.last_extremum_value = self.pitch_history[-2]
                        self.last_extremum_frame = current_frame - 1
                        self.last_nod_frame = current_frame - 1
                        return True, "down-up"
                    print(f"  Ignoring DOWN-UP nod at frame {current_frame - 1} (too close to last nod at {self.last_nod_frame})")

                if is_local_min and (self.last_extremum != "min" or current_frame - self.last_extremum_frame > 1):
                    # Only detect a new nod if we're not too close to the last nod
                    if current_frame - self.last_nod_frame >= self.min_peak_distance or self.last_nod_frame == 0:
                        print(f"  Detected UP-DOWN nod at frame {current_frame - 1}")
                        self.last_extremum = "min"
                        self.last_extremum_value = self.pitch_history[-2]
                        self.last_extremum_frame = current_frame - 1
                        self.last_nod_frame = current_frame - 1
                        return True, "up-down"
                    print(f"  Ignoring UP-DOWN nod at frame {current_frame - 1} (too close to last nod at {self.last_nod_frame})")

        return False, ""

    def update(self, pitch: float, frame_number: int) -> Tuple[bool, str]:
        """
        Update the nod detector with a new pitch measurement.

        Args:
            pitch: The pitch angle in degrees
            frame_number: The current frame number

        Returns:
            Tuple[bool, str]: (detected, direction) where detected is True if a nod was detected,
                            and direction is either 'up-down' or 'down-up'
        """
        # Add the new pitch to the history
        self.pitch_history.append(pitch)

        # Calculate velocity (change in pitch)
        if len(self.pitch_history) > 1:
            velocity = pitch - self.pitch_history[-2]
            self.velocity_history.append(velocity)

            # Update the current direction
            if velocity > 0:
                self.current_direction = "up"
            elif velocity < 0:
                self.last_direction = "down"
        else:
            self.velocity_history.append(0)

        # Try to detect a direction change
        detected, direction = self._detect_direction_change(frame_number)
        if detected:
            self.last_nod_frame = frame_number
            self.nod_count += 1
            self.last_detection_direction = direction
            return True, direction

        # Keep history size manageable
        max_history = 20
        if len(self.pitch_history) > max_history:
            self.pitch_history.pop(0)
            self.velocity_history.pop(0)

        return False, ""

    def get_nod_count(self) -> int:
        """Get the total number of nods detected."""
        return self.nod_count

    def get_last_direction(self) -> Optional[str]:
        """Get the direction of the last detected nod."""
        return self.last_detection_direction

    def reset(self):
        """Reset the detector state."""
        self.pitch_history = []
        self.velocity_history = []
        self.last_extremum = None
        self.last_extremum_value = 0
        self.last_extremum_frame = 0
        self.last_nod_frame = 0
        self.last_direction = None
        self.nod_count = 0
        self.last_detection_direction = None
