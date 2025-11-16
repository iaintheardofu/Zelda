"""
Unit tests for TDOA core functionality
"""

import numpy as np
import pytest


def test_gcc_phat_basic():
    """Test basic GCC-PHAT TDOA calculation"""

    from backend.core.tdoa.gcc_phat import gcc_phat

    # Create two signals with known time delay
    sample_rate = 1e6  # 1 MHz
    num_samples = 1024
    true_delay = 10  # samples
    true_delay_seconds = true_delay / sample_rate

    # Generate signal
    t = np.arange(num_samples) / sample_rate
    signal = np.exp(2j * np.pi * 1000 * t)  # 1 kHz complex tone

    # Create delayed version
    sig1 = signal
    sig2 = np.roll(signal, true_delay)

    # Calculate TDOA
    tdoa, confidence = gcc_phat(sig1, sig2, sample_rate)

    # Check result
    assert abs(tdoa - true_delay_seconds) < 1e-6, f"TDOA error: expected {true_delay_seconds}, got {tdoa}"
    assert confidence > 0.5, f"Low confidence: {confidence}"

    print(f"✓ GCC-PHAT test passed: TDOA = {tdoa*1e6:.2f}μs (expected {true_delay_seconds*1e6:.2f}μs)")


def test_multilateration_simple():
    """Test multilateration with a simple geometry"""

    from backend.core.tdoa.multilateration import TDOAMeasurement, multilaterate_taylor_series

    # Speed of light
    c = 299792458.0

    # Receivers in a square (1000m x 1000m)
    receivers = [
        (0, 0, 0),
        (1000, 0, 0),
        (1000, 1000, 0),
        (0, 1000, 0),
    ]

    # Emitter at center
    true_position = (500, 500, 0)

    # Calculate true TDOAs
    ref_pos = np.array(receivers[0])
    ref_range = np.linalg.norm(np.array(true_position) - ref_pos)

    measurements = []
    for i in range(1, len(receivers)):
        rx_pos = np.array(receivers[i])
        rx_range = np.linalg.norm(np.array(true_position) - rx_pos)

        # TDOA (time difference)
        tdoa = (rx_range - ref_range) / c

        measurement = TDOAMeasurement(
            receiver1_pos=receivers[0],
            receiver2_pos=receivers[i],
            tdoa=tdoa,
            confidence=1.0,
        )
        measurements.append(measurement)

    # Perform multilateration
    estimated_pos, residual = multilaterate_taylor_series(measurements)

    # Check error
    error = np.linalg.norm(np.array(estimated_pos) - np.array(true_position))

    assert error < 1.0, f"Multilateration error too large: {error:.2f}m"

    print(f"✓ Multilateration test passed: error = {error:.2f}m")


def test_signal_classifier_basic():
    """Test signal classifier initialization"""

    try:
        from backend.core.ml.signal_classifier import SignalClassifier, ModulationType

        # Initialize classifier (will use random weights)
        classifier = SignalClassifier(signal_length=1024)

        # Create dummy signal
        iq_data = np.random.randn(1024) + 1j * np.random.randn(1024)

        # Classify (result will be random since model is not trained)
        result = classifier.classify(iq_data)

        assert result.modulation in ModulationType
        assert 0 <= result.confidence <= 1.0

        print(f"✓ Signal classifier test passed: {result.modulation.name} (conf={result.confidence:.3f})")

    except ImportError as e:
        print(f"⚠ Skipping ML test (dependency not available): {e}")
        pytest.skip("PyTorch not available")


def test_kalman_tracker():
    """Test Kalman filter tracker"""

    from backend.core.tracking.kalman import KalmanTracker

    # Initialize tracker
    tracker = KalmanTracker(initial_position=(0, 0, 0))

    # Simulate measurements of a moving target
    true_velocity = (10, 5, 0)  # m/s
    dt = 1.0  # second

    positions = []
    for i in range(10):
        # True position
        true_pos = (
            0 + true_velocity[0] * i * dt,
            0 + true_velocity[1] * i * dt,
            0
        )

        # Add noise
        noise = np.random.randn(3) * 2  # 2m noise
        measured_pos = tuple(np.array(true_pos) + noise)

        # Predict and update
        if i > 0:
            tracker.predict(dt)

        tracker.update(measured_pos)

        positions.append(tracker.get_position())

    # Final position should be close to true position
    final_true_pos = (
        0 + true_velocity[0] * 9 * dt,
        0 + true_velocity[1] * 9 * dt,
        0
    )

    error = np.linalg.norm(np.array(positions[-1]) - np.array(final_true_pos))

    assert error < 10, f"Tracking error too large: {error:.2f}m"

    print(f"✓ Kalman tracker test passed: final error = {error:.2f}m")


if __name__ == "__main__":
    # Run tests directly
    print("Running Zelda TDOA tests...\n")

    test_gcc_phat_basic()
    test_multilateration_simple()
    test_signal_classifier_basic()
    test_kalman_tracker()

    print("\n✓ All tests passed!")
