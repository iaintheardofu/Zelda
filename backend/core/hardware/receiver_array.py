"""
Receiver Array Management for TDOA systems
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import threading

from .sdr_base import SDRReceiver, IQSample


class ReceiverArray:
    """
    Manages an array of SDR receivers for TDOA processing.

    Responsibilities:
    - Coordinate multiple receivers
    - Synchronize sample acquisition
    - Handle time alignment
    - Manage phase coherence
    """

    def __init__(self, name: str = "Array"):
        self.name = name
        self.receivers: Dict[str, SDRReceiver] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._streaming = False
        self._lock = threading.Lock()

    def add_receiver(self, receiver: SDRReceiver) -> None:
        """Add a receiver to the array"""
        with self._lock:
            self.receivers[receiver.receiver_id] = receiver
            logger.info(
                f"Added receiver {receiver.receiver_id} to array {self.name} "
                f"({len(self.receivers)} total)"
            )

    def remove_receiver(self, receiver_id: str) -> None:
        """Remove a receiver from the array"""
        with self._lock:
            if receiver_id in self.receivers:
                receiver = self.receivers[receiver_id]
                if receiver.is_streaming:
                    receiver.stop_stream()
                receiver.disconnect()
                del self.receivers[receiver_id]
                logger.info(f"Removed receiver {receiver_id}")

    def get_receiver(self, receiver_id: str) -> Optional[SDRReceiver]:
        """Get a receiver by ID"""
        return self.receivers.get(receiver_id)

    @property
    def num_receivers(self) -> int:
        """Number of receivers in array"""
        return len(self.receivers)

    @property
    def receiver_ids(self) -> List[str]:
        """List of receiver IDs"""
        return list(self.receivers.keys())

    def connect_all(self) -> bool:
        """Connect all receivers"""
        logger.info(f"Connecting {self.num_receivers} receivers...")

        success = True
        for receiver_id, receiver in self.receivers.items():
            if not receiver.connect():
                logger.error(f"Failed to connect receiver {receiver_id}")
                success = False

        if success:
            logger.info("All receivers connected successfully")
        return success

    def disconnect_all(self) -> None:
        """Disconnect all receivers"""
        logger.info("Disconnecting all receivers...")

        for receiver in self.receivers.values():
            try:
                receiver.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {receiver.receiver_id}: {e}")

    def start_streaming(self, num_threads: Optional[int] = None) -> None:
        """Start streaming on all receivers"""
        if self._streaming:
            logger.warning("Already streaming")
            return

        if num_threads is None:
            num_threads = min(self.num_receivers, 8)

        self._executor = ThreadPoolExecutor(max_workers=num_threads)

        # Start all streams
        for receiver in self.receivers.values():
            receiver.start_stream()

        self._streaming = True
        logger.info(f"Started streaming on {self.num_receivers} receivers")

    def stop_streaming(self) -> None:
        """Stop streaming on all receivers"""
        if not self._streaming:
            return

        for receiver in self.receivers.values():
            try:
                receiver.stop_stream()
            except Exception as e:
                logger.error(f"Error stopping stream {receiver.receiver_id}: {e}")

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._streaming = False
        logger.info("Stopped streaming on all receivers")

    def read_synchronized_samples(
        self, num_samples: int
    ) -> Dict[str, IQSample]:
        """
        Read samples from all receivers in a synchronized manner.

        This attempts to read samples at the same time across all receivers.
        For true time synchronization, receivers should be GPS-disciplined
        or use a common clock source.

        Args:
            num_samples: Number of samples to read from each receiver

        Returns:
            Dictionary mapping receiver_id to IQSample
        """
        if not self._streaming:
            raise RuntimeError("Not streaming. Call start_streaming() first.")

        samples = {}

        # Read from all receivers concurrently
        def read_from_receiver(receiver: SDRReceiver) -> Tuple[str, IQSample]:
            try:
                sample = receiver.read_samples(num_samples)
                return (receiver.receiver_id, sample)
            except Exception as e:
                logger.error(f"Error reading from {receiver.receiver_id}: {e}")
                return (receiver.receiver_id, None)

        if self._executor:
            futures = [
                self._executor.submit(read_from_receiver, receiver)
                for receiver in self.receivers.values()
            ]

            for future in futures:
                receiver_id, sample = future.result()
                if sample is not None:
                    samples[receiver_id] = sample
        else:
            # Fallback to sequential reading
            for receiver in self.receivers.values():
                receiver_id, sample = read_from_receiver(receiver)
                if sample is not None:
                    samples[receiver_id] = sample

        return samples

    def set_frequency_all(self, freq: float) -> None:
        """Set the same frequency on all receivers"""
        logger.info(f"Setting frequency to {freq/1e6:.3f} MHz on all receivers")
        for receiver in self.receivers.values():
            try:
                receiver.set_frequency(freq)
            except Exception as e:
                logger.error(f"Error setting frequency on {receiver.receiver_id}: {e}")

    def set_sample_rate_all(self, rate: float) -> None:
        """Set the same sample rate on all receivers"""
        logger.info(f"Setting sample rate to {rate/1e6:.3f} Msps on all receivers")
        for receiver in self.receivers.values():
            try:
                receiver.set_sample_rate(rate)
            except Exception as e:
                logger.error(f"Error setting sample rate on {receiver.receiver_id}: {e}")

    def set_gain_all(self, gain: float) -> None:
        """Set the same gain on all receivers"""
        logger.info(f"Setting gain to {gain} dB on all receivers")
        for receiver in self.receivers.values():
            try:
                receiver.set_gain(gain)
            except Exception as e:
                logger.error(f"Error setting gain on {receiver.receiver_id}: {e}")

    def get_receiver_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Get positions of all receivers.

        Returns:
            Dictionary mapping receiver_id to (lat, lon, alt)
        """
        positions = {}
        for receiver_id, receiver in self.receivers.items():
            pos = receiver.position
            if pos:
                positions[receiver_id] = pos
        return positions

    def calibrate_phase_offsets(
        self, reference_signal: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calibrate phase offsets between receivers.

        This is critical for TDOA accuracy. In practice, this would:
        1. Use a known reference signal (e.g., from a noise source)
        2. Measure phase differences between receivers
        3. Store calibration offsets

        Args:
            reference_signal: Known reference signal for calibration

        Returns:
            Dictionary mapping receiver_id to phase offset in radians
        """
        offsets = {}

        # TODO: Implement proper phase calibration
        # For now, just return configured offsets
        for receiver_id, receiver in self.receivers.items():
            offsets[receiver_id] = receiver.config.phase_offset

        logger.info(f"Phase calibration: {offsets}")
        return offsets

    def validate_geometry(self) -> bool:
        """
        Validate that receiver geometry is suitable for TDOA.

        For good TDOA performance:
        - Need at least 3 receivers (4+ is better)
        - Receivers shouldn't be colinear
        - Good geometric dilution of precision (GDOP)

        Returns:
            True if geometry is valid
        """
        positions = self.get_receiver_positions()

        if len(positions) < 3:
            logger.error(f"Need at least 3 receivers, have {len(positions)}")
            return False

        # Check for valid positions
        for receiver_id, pos in positions.items():
            if pos is None:
                logger.error(f"Receiver {receiver_id} has no position")
                return False

        # TODO: Check for colinearity and compute GDOP

        logger.info(f"Receiver geometry validated ({len(positions)} receivers)")
        return True

    def __enter__(self):
        """Context manager entry"""
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_streaming()
        self.disconnect_all()

    def __repr__(self) -> str:
        return (
            f"ReceiverArray(name='{self.name}', "
            f"receivers={self.num_receivers}, "
            f"streaming={self._streaming})"
        )
