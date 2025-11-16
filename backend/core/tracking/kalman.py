"""
Kalman Filter implementations for target tracking
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class TrackState:
    """State of a tracked target"""
    position: Tuple[float, float, float]  # lat, lon, alt
    velocity: Tuple[float, float, float]  # velocities in each dimension
    covariance: np.ndarray  # State covariance matrix
    timestamp: float
    track_id: str


class KalmanTracker:
    """
    Linear Kalman Filter for target position tracking.

    State vector: [x, y, z, vx, vy, vz]
    - x, y, z: position
    - vx, vy, vz: velocity

    Assumes constant velocity model.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 10.0,
        initial_position: Optional[Tuple[float, float, float]] = None,
        track_id: str = "target_0",
    ):
        """
        Initialize Kalman tracker.

        Args:
            process_noise: Process noise (model uncertainty)
            measurement_noise: Measurement noise (sensor uncertainty in meters)
            initial_position: Initial position guess
            track_id: Unique identifier for this track
        """

        self.track_id = track_id
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State dimension: [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.measurement_dim = 3

        # Initialize state
        if initial_position:
            self.state = np.array([
                initial_position[0],  # x
                initial_position[1],  # y
                initial_position[2],  # z
                0, 0, 0,  # vx, vy, vz (initially zero)
            ])
        else:
            self.state = np.zeros(self.state_dim)

        # State covariance (uncertainty)
        self.P = np.eye(self.state_dim) * 1000  # High initial uncertainty

        # Transition matrix (constant velocity model)
        self.F = None  # Will be set in predict() based on dt

        # Measurement matrix (we observe position only)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        # Process noise covariance
        self.Q = np.eye(self.state_dim) * self.process_noise

        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * (self.measurement_noise ** 2)

        self.last_update_time = None
        self.num_updates = 0

        logger.debug(f"KalmanTracker initialized: track_id={track_id}")

    def predict(self, dt: float) -> np.ndarray:
        """
        Predict next state.

        Args:
            dt: Time step (seconds)

        Returns:
            Predicted state
        """

        # Build state transition matrix for this time step
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

        # Predict state
        self.state = self.F @ self.state

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state

    def update(
        self,
        measurement: Tuple[float, float, float],
        measurement_covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Update state with a new measurement.

        Args:
            measurement: Measured position (x, y, z)
            measurement_covariance: Measurement covariance (if known)

        Returns:
            Updated state
        """

        z = np.array(measurement)

        # Use custom measurement covariance if provided
        R = measurement_covariance if measurement_covariance is not None else self.R

        # Innovation (measurement residual)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudoinverse")
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        # Update state
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        self.num_updates += 1

        return self.state

    def get_track_state(self) -> TrackState:
        """Get current track state"""

        return TrackState(
            position=tuple(self.state[:3]),
            velocity=tuple(self.state[3:]),
            covariance=self.P.copy(),
            timestamp=self.last_update_time or 0.0,
            track_id=self.track_id,
        )

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position estimate"""
        return tuple(self.state[:3])

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity estimate"""
        return tuple(self.state[3:])

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty (meters, 1-sigma)"""
        # Average of x, y, z uncertainties
        return np.sqrt(np.mean([self.P[i, i] for i in range(3)]))


class EKFTracker:
    """
    Extended Kalman Filter for nonlinear tracking.

    Useful when the measurement model is nonlinear (e.g., range/bearing,
    or when using geodetic coordinates).

    For simplicity, this implementation assumes the same state/measurement
    model as the linear KF, but can be extended for nonlinear models.
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 10.0,
        initial_position: Optional[Tuple[float, float, float]] = None,
        track_id: str = "target_0",
    ):
        """Initialize Extended Kalman Filter"""

        # For now, use the linear KF as base
        # In a full implementation, this would handle Jacobians
        self.kf = KalmanTracker(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_position=initial_position,
            track_id=track_id,
        )

        logger.debug(f"EKFTracker initialized: track_id={track_id}")

    def predict(self, dt: float) -> np.ndarray:
        """Predict (uses linear model for now)"""
        return self.kf.predict(dt)

    def update(
        self,
        measurement: Tuple[float, float, float],
        measurement_covariance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Update (uses linear model for now)"""
        return self.kf.update(measurement, measurement_covariance)

    def get_track_state(self) -> TrackState:
        """Get current track state"""
        return self.kf.get_track_state()

    def get_position(self) -> Tuple[float, float, float]:
        """Get current position"""
        return self.kf.get_position()

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity"""
        return self.kf.get_velocity()

    def get_position_uncertainty(self) -> float:
        """Get position uncertainty"""
        return self.kf.get_position_uncertainty()


def smooth_trajectory(
    positions: list,
    timestamps: list,
    process_noise: float = 0.1,
    measurement_noise: float = 10.0,
) -> list:
    """
    Smooth a trajectory using Kalman filtering.

    Args:
        positions: List of (x, y, z) positions
        timestamps: List of timestamps (seconds)
        process_noise: Process noise parameter
        measurement_noise: Measurement noise parameter

    Returns:
        List of smoothed positions
    """

    if len(positions) < 2:
        return positions

    # Initialize tracker with first position
    tracker = KalmanTracker(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        initial_position=positions[0],
    )

    smoothed = []

    for i, (pos, t) in enumerate(zip(positions, timestamps)):
        if i > 0:
            dt = t - timestamps[i-1]
            tracker.predict(dt)

        tracker.update(pos)
        smoothed.append(tracker.get_position())

    return smoothed
