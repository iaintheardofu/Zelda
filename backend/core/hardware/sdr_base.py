"""
Base classes for SDR hardware abstraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class ReceiverConfig:
    """Configuration for an SDR receiver"""

    # Hardware identification
    driver: str  # e.g., "rtlsdr", "uhd", "krakensdr"
    serial: Optional[str] = None

    # RF parameters
    center_freq: float = 100e6  # Hz
    sample_rate: float = 2.4e6  # Samples/second
    bandwidth: Optional[float] = None  # Hz (if different from sample_rate)
    gain: float = 20.0  # dB

    # Position (for TDOA)
    latitude: Optional[float] = None  # degrees
    longitude: Optional[float] = None  # degrees
    altitude: Optional[float] = 0.0  # meters

    # Timing
    use_gps: bool = False  # GPS-disciplined timing
    pps_source: Optional[str] = None  # PPS source for synchronization

    # Buffering
    buffer_size: int = 16384  # samples
    num_buffers: int = 16

    # Calibration
    phase_offset: float = 0.0  # radians
    time_offset: float = 0.0  # seconds

    def __post_init__(self):
        """Validate configuration"""
        if self.bandwidth is None:
            self.bandwidth = self.sample_rate

        if self.latitude is not None and self.longitude is not None:
            assert -90 <= self.latitude <= 90, "Invalid latitude"
            assert -180 <= self.longitude <= 180, "Invalid longitude"


@dataclass
class IQSample:
    """A block of I/Q samples with metadata"""

    data: np.ndarray  # Complex samples (I + jQ)
    timestamp: datetime  # Reception time
    center_freq: float  # Hz
    sample_rate: float  # Samples/second
    receiver_id: str  # Which receiver captured this
    sequence_num: int = 0  # Sample sequence number

    @property
    def duration(self) -> float:
        """Duration of this sample block in seconds"""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of samples in this block"""
        return len(self.data)


class SDRReceiver(ABC):
    """
    Abstract base class for all SDR receivers.

    This provides a unified interface for:
    - RTL-SDR dongles
    - USRP devices
    - KrakenSDR arrays
    - Any SoapySDR-compatible hardware
    """

    def __init__(self, config: ReceiverConfig, receiver_id: Optional[str] = None):
        self.config = config
        self.receiver_id = receiver_id or f"{config.driver}_{id(self)}"
        self._is_streaming = False
        self._sample_count = 0

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the SDR hardware.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the SDR hardware"""
        pass

    @abstractmethod
    def start_stream(self) -> None:
        """Start streaming I/Q samples"""
        pass

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop streaming I/Q samples"""
        pass

    @abstractmethod
    def read_samples(self, num_samples: int) -> IQSample:
        """
        Read a block of I/Q samples.

        Args:
            num_samples: Number of samples to read

        Returns:
            IQSample object containing the data
        """
        pass

    @abstractmethod
    def set_frequency(self, freq: float) -> None:
        """Set center frequency in Hz"""
        pass

    @abstractmethod
    def set_sample_rate(self, rate: float) -> None:
        """Set sample rate in samples/second"""
        pass

    @abstractmethod
    def set_gain(self, gain: float) -> None:
        """Set gain in dB"""
        pass

    @abstractmethod
    def get_frequency(self) -> float:
        """Get current center frequency in Hz"""
        pass

    @abstractmethod
    def get_sample_rate(self) -> float:
        """Get current sample rate"""
        pass

    @abstractmethod
    def get_gain(self) -> float:
        """Get current gain in dB"""
        pass

    @property
    def is_streaming(self) -> bool:
        """Check if receiver is currently streaming"""
        return self._is_streaming

    @property
    def position(self) -> Optional[Tuple[float, float, float]]:
        """Get receiver position as (lat, lon, alt)"""
        if self.config.latitude is not None and self.config.longitude is not None:
            return (
                self.config.latitude,
                self.config.longitude,
                self.config.altitude or 0.0
            )
        return None

    def calibrate_phase(self, reference_signal: np.ndarray) -> float:
        """
        Calibrate phase offset using a reference signal.

        Args:
            reference_signal: Known reference signal

        Returns:
            Measured phase offset in radians
        """
        # This would be implemented by subclasses or as a utility function
        # For now, return the configured offset
        return self.config.phase_offset

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id='{self.receiver_id}', "
            f"driver='{self.config.driver}', "
            f"freq={self.config.center_freq/1e6:.2f}MHz, "
            f"sr={self.config.sample_rate/1e6:.2f}Msps)"
        )
