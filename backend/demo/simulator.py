"""
Demo Simulator for Zelda TDOA System

Simulates:
- Multiple receivers at known positions
- Emitters transmitting RF signals
- TDOA measurements and geolocation
- Real-time visualization

This allows testing the entire system without physical hardware.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from loguru import logger


@dataclass
class SimulatedReceiver:
    """Simulated SDR receiver"""
    receiver_id: str
    position: Tuple[float, float, float]  # lat, lon, alt (meters for simplicity)
    sample_rate: float = 2.4e6
    center_freq: float = 100e6


@dataclass
class SimulatedEmitter:
    """Simulated RF emitter"""
    emitter_id: str
    position: Tuple[float, float, float]
    frequency: float = 100e6
    power: float = 1.0  # Watts
    modulation: str = "FM"

    def __post_init__(self):
        # Add some random motion
        self.velocity = (
            np.random.uniform(-5, 5),  # m/s in x
            np.random.uniform(-5, 5),  # m/s in y
            0.0,  # stationary in z
        )


class DemoSystem:
    """
    Complete demo system for Zelda.

    Sets up a simulated environment and processes it through the
    entire TDOA pipeline.
    """

    def __init__(
        self,
        num_receivers: int = 4,
        num_emitters: int = 1,
        area_size: float = 1000.0,  # meters
    ):
        """
        Initialize demo system.

        Args:
            num_receivers: Number of simulated receivers
            num_emitters: Number of simulated emitters
            area_size: Size of simulation area (meters)
        """

        self.area_size = area_size
        self.receivers = self._create_receivers(num_receivers)
        self.emitters = self._create_emitters(num_emitters)
        self.speed_of_light = 299792458.0  # m/s

        logger.info(
            f"Demo system initialized: {num_receivers} receivers, "
            f"{num_emitters} emitters, area={area_size}m"
        )

    def _create_receivers(self, num: int) -> List[SimulatedReceiver]:
        """Create receivers in a good geometric configuration"""

        receivers = []

        if num == 4:
            # Square configuration
            positions = [
                (0, 0, 0),
                (self.area_size, 0, 0),
                (self.area_size, self.area_size, 0),
                (0, self.area_size, 0),
            ]
        elif num == 3:
            # Triangle
            positions = [
                (0, 0, 0),
                (self.area_size, 0, 0),
                (self.area_size/2, self.area_size * 0.866, 0),
            ]
        else:
            # Random positions
            positions = [
                (
                    np.random.uniform(0, self.area_size),
                    np.random.uniform(0, self.area_size),
                    0.0
                )
                for _ in range(num)
            ]

        for i, pos in enumerate(positions):
            receiver = SimulatedReceiver(
                receiver_id=f"rx_{i}",
                position=pos,
            )
            receivers.append(receiver)
            logger.info(f"  Receiver {receiver.receiver_id} at {pos}")

        return receivers

    def _create_emitters(self, num: int) -> List[SimulatedEmitter]:
        """Create emitters at random positions"""

        emitters = []

        for i in range(num):
            position = (
                np.random.uniform(0.2 * self.area_size, 0.8 * self.area_size),
                np.random.uniform(0.2 * self.area_size, 0.8 * self.area_size),
                np.random.uniform(0, 100),  # Some altitude variation
            )

            emitter = SimulatedEmitter(
                emitter_id=f"emitter_{i}",
                position=position,
            )
            emitters.append(emitter)
            logger.info(f"  Emitter {emitter.emitter_id} at {position}")

        return emitters

    def simulate_signal_propagation(
        self,
        emitter: SimulatedEmitter,
        receiver: SimulatedReceiver,
        noise_level: float = 0.1,
    ) -> Tuple[np.ndarray, float]:
        """
        Simulate signal propagation from emitter to receiver.

        Args:
            emitter: Simulated emitter
            receiver: Simulated receiver
            noise_level: Noise level (sigma)

        Returns:
            Tuple of (I/Q samples, time_of_arrival)
        """

        # Calculate distance
        distance = np.linalg.norm(
            np.array(emitter.position) - np.array(receiver.position)
        )

        # Time of arrival
        toa = distance / self.speed_of_light

        # Path loss (free space)
        wavelength = self.speed_of_light / emitter.frequency
        path_loss = (wavelength / (4 * np.pi * distance)) ** 2

        # Received power
        rx_power = emitter.power * path_loss

        # Generate simulated signal (FM modulation)
        num_samples = 1024
        t = np.arange(num_samples) / receiver.sample_rate

        # Carrier
        carrier_phase = 2 * np.pi * emitter.frequency * t

        # Modulation (simple sinusoid for FM)
        modulation_freq = 1000  # Hz
        modulation = np.sin(2 * np.pi * modulation_freq * t)

        # FM signal
        frequency_deviation = 75e3  # Hz
        phase = carrier_phase + (frequency_deviation / modulation_freq) * modulation

        # Complex signal
        signal = np.sqrt(rx_power) * np.exp(1j * phase)

        # Add noise
        noise = noise_level * (
            np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        )

        signal += noise

        return signal, toa

    def calculate_true_tdoas(
        self,
        emitter: SimulatedEmitter,
    ) -> dict:
        """
        Calculate ground truth TDOAs.

        Args:
            emitter: Emitter to calculate TDOAs for

        Returns:
            Dictionary mapping (rx1_id, rx2_id) to TDOA
        """

        # Calculate time of arrival for each receiver
        toas = {}
        for receiver in self.receivers:
            distance = np.linalg.norm(
                np.array(emitter.position) - np.array(receiver.position)
            )
            toa = distance / self.speed_of_light
            toas[receiver.receiver_id] = toa

        # Calculate TDOAs (relative to first receiver)
        ref_id = self.receivers[0].receiver_id
        ref_toa = toas[ref_id]

        tdoas = {}
        for receiver in self.receivers[1:]:
            tdoa = toas[receiver.receiver_id] - ref_toa
            tdoas[(ref_id, receiver.receiver_id)] = tdoa

        return tdoas

    def run_single_iteration(self, emitter_idx: int = 0) -> dict:
        """
        Run a single iteration of the demo.

        Args:
            emitter_idx: Which emitter to process

        Returns:
            Dictionary with results
        """

        emitter = self.emitters[emitter_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing emitter: {emitter.emitter_id}")
        logger.info(f"True position: {emitter.position}")

        # Simulate signal reception at all receivers
        signals = {}
        for receiver in self.receivers:
            signal, toa = self.simulate_signal_propagation(emitter, receiver)
            signals[receiver.receiver_id] = signal

        # Calculate TDOAs using GCC-PHAT
        from ..core.tdoa.gcc_phat import calculate_tdoa

        ref_id = self.receivers[0].receiver_id
        ref_signal = signals[ref_id]

        tdoa_measurements = []

        for receiver in self.receivers[1:]:
            signal = signals[receiver.receiver_id]

            # Calculate TDOA
            tdoa, confidence = calculate_tdoa(
                ref_signal,
                signal,
                self.receivers[0].sample_rate,
                method="gcc-phat",
            )

            # Create measurement
            from ..core.tdoa.multilateration import TDOAMeasurement

            measurement = TDOAMeasurement(
                receiver1_pos=self.receivers[0].position,
                receiver2_pos=receiver.position,
                tdoa=tdoa,
                confidence=confidence,
            )
            tdoa_measurements.append(measurement)

            logger.info(
                f"  TDOA {ref_id} <-> {receiver.receiver_id}: "
                f"{tdoa*1e6:.2f}μs (conf={confidence:.3f})"
            )

        # Perform multilateration
        from ..core.tdoa.multilateration import multilaterate_taylor_series

        try:
            estimated_pos, residual = multilaterate_taylor_series(tdoa_measurements)

            logger.info(f"Estimated position: {estimated_pos}")
            logger.info(f"Residual error: {residual:.2f}m")

            # Calculate actual error
            true_pos = np.array(emitter.position)
            est_pos = np.array(estimated_pos)
            error = np.linalg.norm(true_pos - est_pos)

            logger.info(f"Actual error: {error:.2f}m")

            result = {
                "timestamp": datetime.now().isoformat(),
                "emitter_id": emitter.emitter_id,
                "true_position": emitter.position,
                "estimated_position": estimated_pos,
                "error_meters": error,
                "residual_error": residual,
                "num_measurements": len(tdoa_measurements),
            }

            return result

        except Exception as e:
            logger.error(f"Multilateration failed: {e}")
            return None

    def run(
        self,
        duration: float = None,
        update_rate: float = 1.0,
    ):
        """
        Run the demo system.

        Args:
            duration: Duration in seconds (None = infinite)
            update_rate: Update rate in Hz
        """

        logger.info(f"\n{'='*60}")
        logger.info("Starting Zelda Demo System")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        iteration = 0

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Process each emitter
                for i in range(len(self.emitters)):
                    result = self.run_single_iteration(i)

                    if result:
                        logger.info(f"✓ Iteration {iteration} completed")
                    else:
                        logger.warning(f"✗ Iteration {iteration} failed")

                # Update emitter positions (add motion)
                for emitter in self.emitters:
                    new_pos = (
                        emitter.position[0] + emitter.velocity[0] / update_rate,
                        emitter.position[1] + emitter.velocity[1] / update_rate,
                        emitter.position[2] + emitter.velocity[2] / update_rate,
                    )

                    # Bounce off boundaries
                    if new_pos[0] < 0 or new_pos[0] > self.area_size:
                        emitter.velocity = (-emitter.velocity[0], emitter.velocity[1], emitter.velocity[2])
                    if new_pos[1] < 0 or new_pos[1] > self.area_size:
                        emitter.velocity = (emitter.velocity[0], -emitter.velocity[1], emitter.velocity[2])

                    emitter.position = new_pos

                iteration += 1

                # Sleep for update rate
                time.sleep(1.0 / update_rate)

        except KeyboardInterrupt:
            logger.info("\nDemo stopped by user")

        logger.info(f"\nDemo completed: {iteration} iterations")


if __name__ == "__main__":
    # Run demo
    demo = DemoSystem(num_receivers=4, num_emitters=1)
    demo.run(duration=10, update_rate=1.0)
