"""
SoapySDR implementation of the SDR receiver interface
"""

from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime
from loguru import logger

try:
    import SoapySDR
    SOAPY_AVAILABLE = True
except ImportError:
    SOAPY_AVAILABLE = False
    logger.warning("SoapySDR not available - hardware support disabled")

from .sdr_base import SDRReceiver, ReceiverConfig, IQSample


class SoapySDRReceiver(SDRReceiver):
    """
    SoapySDR implementation supporting multiple SDR hardware platforms:
    - RTL-SDR
    - USRP (via UHD)
    - HackRF
    - BladeRF
    - LimeSDR
    - PlutoSDR
    - And any other SoapySDR-compatible device
    """

    def __init__(self, config: ReceiverConfig, receiver_id: Optional[str] = None):
        super().__init__(config, receiver_id)

        if not SOAPY_AVAILABLE:
            raise RuntimeError("SoapySDR is not installed. Install with: pip install SoapySDR")

        self.device: Optional[SoapySDR.Device] = None
        self.stream: Optional[SoapySDR.Stream] = None
        self._buffer = None

    def connect(self) -> bool:
        """Connect to SDR device via SoapySDR"""
        try:
            # Build device arguments
            args = {"driver": self.config.driver}
            if self.config.serial:
                args["serial"] = self.config.serial

            logger.info(f"Connecting to {self.config.driver} with args: {args}")

            # Create device
            self.device = SoapySDR.Device(args)

            # Configure device
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.config.sample_rate)
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.config.center_freq)

            # Set gain (try automatic first, then manual)
            try:
                self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)
                self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.config.gain)
            except Exception as e:
                logger.warning(f"Could not set gain: {e}")

            # Set bandwidth if supported
            if self.config.bandwidth:
                try:
                    self.device.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, self.config.bandwidth)
                except Exception as e:
                    logger.warning(f"Could not set bandwidth: {e}")

            # Log actual settings
            actual_rate = self.device.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
            actual_freq = self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
            logger.info(f"Connected: {actual_freq/1e6:.3f} MHz @ {actual_rate/1e6:.3f} Msps")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to device: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from SDR device"""
        if self._is_streaming:
            self.stop_stream()

        if self.device:
            logger.info(f"Disconnecting from {self.receiver_id}")
            self.device = None

    def start_stream(self) -> None:
        """Start streaming I/Q samples"""
        if not self.device:
            raise RuntimeError("Device not connected")

        if self._is_streaming:
            logger.warning("Stream already active")
            return

        # Create stream
        self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)

        # Activate stream
        self.device.activateStream(self.stream)

        # Allocate buffer
        self._buffer = np.zeros(self.config.buffer_size, dtype=np.complex64)

        self._is_streaming = True
        self._sample_count = 0
        logger.info(f"Stream started for {self.receiver_id}")

    def stop_stream(self) -> None:
        """Stop streaming I/Q samples"""
        if not self._is_streaming:
            return

        if self.device and self.stream:
            self.device.deactivateStream(self.stream)
            self.device.closeStream(self.stream)
            self.stream = None

        self._is_streaming = False
        logger.info(f"Stream stopped for {self.receiver_id}")

    def read_samples(self, num_samples: int) -> IQSample:
        """Read I/Q samples from the device"""
        if not self._is_streaming:
            raise RuntimeError("Stream not active. Call start_stream() first.")

        if not self.device or not self.stream:
            raise RuntimeError("Device or stream not initialized")

        # Allocate buffer if needed
        if self._buffer is None or len(self._buffer) < num_samples:
            self._buffer = np.zeros(num_samples, dtype=np.complex64)

        # Read samples
        sr = self.device.readStream(self.stream, [self._buffer], num_samples)

        if sr.ret < 0:
            raise RuntimeError(f"Stream error: {sr.ret}")

        # Extract actual samples read
        samples_read = sr.ret
        data = self._buffer[:samples_read].copy()

        # Create IQSample object
        sample = IQSample(
            data=data,
            timestamp=datetime.now(),
            center_freq=self.get_frequency(),
            sample_rate=self.get_sample_rate(),
            receiver_id=self.receiver_id,
            sequence_num=self._sample_count
        )

        self._sample_count += samples_read

        return sample

    def set_frequency(self, freq: float) -> None:
        """Set center frequency"""
        if not self.device:
            raise RuntimeError("Device not connected")

        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        self.config.center_freq = freq
        logger.debug(f"{self.receiver_id}: Frequency set to {freq/1e6:.3f} MHz")

    def set_sample_rate(self, rate: float) -> None:
        """Set sample rate"""
        if not self.device:
            raise RuntimeError("Device not connected")

        self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, rate)
        self.config.sample_rate = rate
        logger.debug(f"{self.receiver_id}: Sample rate set to {rate/1e6:.3f} Msps")

    def set_gain(self, gain: float) -> None:
        """Set gain"""
        if not self.device:
            raise RuntimeError("Device not connected")

        try:
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self.config.gain = gain
            logger.debug(f"{self.receiver_id}: Gain set to {gain} dB")
        except Exception as e:
            logger.warning(f"Failed to set gain: {e}")

    def get_frequency(self) -> float:
        """Get current frequency"""
        if not self.device:
            return self.config.center_freq
        return self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)

    def get_sample_rate(self) -> float:
        """Get current sample rate"""
        if not self.device:
            return self.config.sample_rate
        return self.device.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)

    def get_gain(self) -> float:
        """Get current gain"""
        if not self.device:
            return self.config.gain
        try:
            return self.device.getGain(SoapySDR.SOAPY_SDR_RX, 0)
        except:
            return self.config.gain

    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        if not self.device:
            return {}

        info = {
            "driver": self.config.driver,
            "hardware": self.device.getHardwareKey(),
            "receiver_id": self.receiver_id,
            "frequency": self.get_frequency(),
            "sample_rate": self.get_sample_rate(),
            "gain": self.get_gain(),
        }

        # Try to get additional info
        try:
            info["hardware_info"] = self.device.getHardwareInfo()
        except:
            pass

        return info

    @staticmethod
    def enumerate_devices() -> list:
        """Enumerate all available SoapySDR devices"""
        if not SOAPY_AVAILABLE:
            return []

        try:
            devices = SoapySDR.Device.enumerate()
            logger.info(f"Found {len(devices)} SoapySDR device(s)")
            for i, dev in enumerate(devices):
                logger.info(f"  [{i}] {dev}")
            return devices
        except Exception as e:
            logger.error(f"Failed to enumerate devices: {e}")
            return []
