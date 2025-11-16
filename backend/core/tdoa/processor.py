"""
TDOA Processor - High-level interface for TDOA geolocation
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from ..hardware import ReceiverArray, IQSample
from .gcc_phat import calculate_tdoa, batch_tdoa
from .multilateration import (
    TDOAMeasurement,
    multilaterate_taylor_series,
    multilaterate_least_squares,
    multilaterate_genetic,
    compute_gdop,
)


@dataclass
class GeolocationResult:
    """Result of a TDOA geolocation"""

    # Position estimate
    latitude: float
    longitude: float
    altitude: float

    # Quality metrics
    residual_error: float  # meters
    gdop: float
    confidence: float  # 0-1

    # Metadata
    timestamp: datetime
    num_receivers: int
    num_measurements: int
    algorithm: str

    # Raw data
    tdoa_measurements: List[TDOAMeasurement]

    def __repr__(self) -> str:
        return (
            f"GeolocationResult("
            f"pos=({self.latitude:.6f}, {self.longitude:.6f}, {self.altitude:.1f}m), "
            f"error={self.residual_error:.1f}m, "
            f"GDOP={self.gdop:.2f}, "
            f"conf={self.confidence:.3f})"
        )


class TDOAProcessor:
    """
    High-level TDOA processor that coordinates:
    1. Signal acquisition from receiver array
    2. TDOA calculation between receiver pairs
    3. Multilateration to estimate source position
    4. Quality metrics and validation
    """

    def __init__(
        self,
        receiver_array: ReceiverArray,
        tdoa_method: str = "gcc-phat",
        multilateration_method: str = "taylor",
        reference_receiver_id: Optional[str] = None,
    ):
        """
        Initialize TDOA processor.

        Args:
            receiver_array: Array of SDR receivers
            tdoa_method: TDOA calculation method ('gcc-phat', 'xcorr', 'adaptive')
            multilateration_method: Multilateration algorithm ('taylor', 'least_squares', 'genetic')
            reference_receiver_id: ID of reference receiver (first if None)
        """

        self.receiver_array = receiver_array
        self.tdoa_method = tdoa_method
        self.multilateration_method = multilateration_method

        # Select reference receiver
        if reference_receiver_id is None:
            self.reference_receiver_id = receiver_array.receiver_ids[0]
        else:
            self.reference_receiver_id = reference_receiver_id

        logger.info(
            f"TDOAProcessor initialized: {receiver_array.num_receivers} receivers, "
            f"TDOA method='{tdoa_method}', multilateration='{multilateration_method}'"
        )

    def process_samples(
        self,
        samples: Dict[str, IQSample],
        max_tau: Optional[float] = None,
    ) -> Optional[GeolocationResult]:
        """
        Process a set of synchronized samples to produce a geolocation.

        Args:
            samples: Dictionary mapping receiver_id to IQSample
            max_tau: Maximum expected TDOA (seconds)

        Returns:
            GeolocationResult or None if processing failed
        """

        # Validate we have enough receivers
        if len(samples) < 3:
            logger.error(f"Need at least 3 receivers, got {len(samples)}")
            return None

        # Get reference signal
        if self.reference_receiver_id not in samples:
            logger.error(f"Reference receiver {self.reference_receiver_id} not in samples")
            return None

        ref_sample = samples[self.reference_receiver_id]
        ref_position = self.receiver_array.get_receiver(self.reference_receiver_id).position

        if ref_position is None:
            logger.error(f"Reference receiver has no position")
            return None

        # Calculate TDOA for each receiver pair
        measurements = []

        for receiver_id, sample in samples.items():
            if receiver_id == self.reference_receiver_id:
                continue

            receiver = self.receiver_array.get_receiver(receiver_id)
            if receiver is None or receiver.position is None:
                logger.warning(f"Receiver {receiver_id} has no position, skipping")
                continue

            # Calculate TDOA
            tdoa, confidence = calculate_tdoa(
                ref_sample.data,
                sample.data,
                ref_sample.sample_rate,
                method=self.tdoa_method,
                max_tau=max_tau,
            )

            # Create measurement
            measurement = TDOAMeasurement(
                receiver1_pos=ref_position,
                receiver2_pos=receiver.position,
                tdoa=tdoa,
                confidence=confidence,
            )
            measurements.append(measurement)

            logger.debug(
                f"TDOA: {self.reference_receiver_id} <-> {receiver_id}: "
                f"{tdoa*1e6:.2f}μs (conf={confidence:.3f})"
            )

        if len(measurements) < 2:
            logger.error("Not enough valid TDOA measurements")
            return None

        # Perform multilateration
        try:
            position, residual = self._multilaterate(measurements)
        except Exception as e:
            logger.error(f"Multilateration failed: {e}")
            return None

        # Calculate GDOP
        receiver_positions = [
            self.receiver_array.get_receiver(rid).position
            for rid in samples.keys()
            if self.receiver_array.get_receiver(rid).position is not None
        ]
        gdop = compute_gdop(receiver_positions, position)

        # Overall confidence (geometric mean of measurement confidences)
        confidences = [m.confidence for m in measurements]
        overall_confidence = np.exp(np.mean(np.log(np.array(confidences) + 1e-10)))

        # Create result
        result = GeolocationResult(
            latitude=position[0],
            longitude=position[1],
            altitude=position[2],
            residual_error=residual,
            gdop=gdop,
            confidence=overall_confidence,
            timestamp=datetime.now(),
            num_receivers=len(samples),
            num_measurements=len(measurements),
            algorithm=f"{self.tdoa_method}+{self.multilateration_method}",
            tdoa_measurements=measurements,
        )

        logger.info(f"Geolocation: {result}")

        return result

    def _multilaterate(
        self,
        measurements: List[TDOAMeasurement]
    ) -> Tuple[Tuple[float, float, float], float]:
        """
        Perform multilateration using configured method.

        Args:
            measurements: List of TDOA measurements

        Returns:
            Tuple of (position, residual_error)
        """

        if self.multilateration_method == "taylor":
            return multilaterate_taylor_series(measurements)

        elif self.multilateration_method == "least_squares":
            return multilaterate_least_squares(measurements)

        elif self.multilateration_method == "genetic":
            return multilaterate_genetic(measurements)

        else:
            raise ValueError(f"Unknown multilateration method: {self.multilateration_method}")

    def process_stream(
        self,
        num_samples: int = 16384,
        max_tau: Optional[float] = None,
    ) -> GeolocationResult:
        """
        Process a single block from the streaming receiver array.

        Args:
            num_samples: Number of samples to read
            max_tau: Maximum expected TDOA

        Returns:
            GeolocationResult
        """

        # Read synchronized samples from all receivers
        samples = self.receiver_array.read_synchronized_samples(num_samples)

        # Process them
        return self.process_samples(samples, max_tau=max_tau)

    def continuous_processing(
        self,
        num_samples: int = 16384,
        max_tau: Optional[float] = None,
        callback = None,
    ):
        """
        Continuously process samples and yield geolocation results.

        Args:
            num_samples: Number of samples per processing block
            max_tau: Maximum expected TDOA
            callback: Optional callback function called with each result

        Yields:
            GeolocationResult objects
        """

        logger.info("Starting continuous TDOA processing...")

        while self.receiver_array._streaming:
            try:
                result = self.process_stream(num_samples, max_tau)

                if result is not None:
                    if callback:
                        callback(result)
                    yield result

            except KeyboardInterrupt:
                logger.info("Continuous processing interrupted")
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                continue

    def validate_geometry(self) -> bool:
        """Validate receiver geometry for TDOA"""
        return self.receiver_array.validate_geometry()

    def get_expected_max_tdoa(self) -> float:
        """
        Calculate maximum expected TDOA based on receiver geometry.

        Returns:
            Maximum TDOA in seconds
        """

        positions = list(self.receiver_array.get_receiver_positions().values())

        if len(positions) < 2:
            return 0.0

        # Find maximum distance between receivers
        max_distance = 0.0
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                max_distance = max(max_distance, dist)

        # Convert to time
        # Speed of light in m/s
        c = 299792458.0
        max_tdoa = max_distance / c

        logger.debug(f"Maximum receiver baseline: {max_distance:.1f}m, max TDOA: {max_tdoa*1e6:.2f}μs")

        return max_tdoa
