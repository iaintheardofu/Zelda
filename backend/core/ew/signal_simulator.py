"""
ZELDA Defensive EW Suite - Signal Simulator (Educational/Testing Only)

Generates synthetic RF signals for testing detection and mitigation algorithms.
NO TRANSMISSION - Software simulation only for educational purposes.

Legal Use: Algorithm testing, training, education (no RF transmission)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# DISCLAIMER
DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════════╗
║  ZELDA SIGNAL SIMULATOR - EDUCATIONAL USE ONLY                       ║
║                                                                      ║
║  This module generates SOFTWARE SIMULATIONS of RF signals.           ║
║  NO HARDWARE TRANSMISSION OCCURS.                                    ║
║                                                                      ║
║  Legal Use Cases:                                                    ║
║   ✓ Algorithm testing and development                                ║
║   ✓ Educational demonstrations                                       ║
║   ✓ Training and certification                                       ║
║   ✓ Performance benchmarking                                         ║
║                                                                      ║
║  WARNING: Actual RF transmission requires FCC authorization.         ║
║           This tool does NOT transmit - it generates I/Q samples.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


class SignalType(Enum):
    """Types of signals that can be simulated"""
    CLEAN_TONE = "clean_tone"
    QPSK = "qpsk"
    QAM16 = "qam16"
    OFDM = "ofdm"
    GPS_L1 = "gps_l1"
    WIFI = "wifi"
    LTE = "lte"


class JammingSimulationType(Enum):
    """Types of jamming that can be simulated"""
    NONE = "none"
    BARRAGE = "barrage"  # Wideband noise
    SPOT = "spot"  # Narrowband tone
    SWEPT = "swept"  # Frequency sweeping
    PULSE = "pulse"  # On/off pulsed
    CHIRP = "chirp"  # Linear FM sweep


class RFSignalSimulator:
    """
    Simulate RF signals for testing and education.

    IMPORTANT: This class generates I/Q samples in software only.
    No hardware transmission occurs. For testing algorithms only.
    """

    def __init__(self, sample_rate: float = 40e6):
        """
        Initialize RF signal simulator.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        logger.info(f"RFSignalSimulator initialized ({sample_rate/1e6:.1f} MHz sample rate)")
        logger.info("⚠ SOFTWARE SIMULATION ONLY - NO TRANSMISSION")

    def generate_clean_signal(
        self,
        signal_type: SignalType,
        duration_sec: float,
        carrier_freq: float,
        amplitude: float = 1.0,
        snr_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate clean RF signal (no jamming).

        Args:
            signal_type: Type of signal to generate
            duration_sec: Duration in seconds
            carrier_freq: Carrier frequency in Hz
            amplitude: Signal amplitude
            snr_db: Optional SNR (adds noise if specified)

        Returns:
            Complex I/Q samples
        """
        num_samples = int(duration_sec * self.sample_rate)
        t = np.arange(num_samples) / self.sample_rate

        # Generate baseband signal
        if signal_type == SignalType.CLEAN_TONE:
            baseband = np.ones(num_samples, dtype=complex)

        elif signal_type == SignalType.QPSK:
            baseband = self._generate_qpsk(num_samples)

        elif signal_type == SignalType.QAM16:
            baseband = self._generate_qam16(num_samples)

        elif signal_type == SignalType.OFDM:
            baseband = self._generate_ofdm(num_samples)

        elif signal_type == SignalType.GPS_L1:
            baseband = self._generate_gps_like(num_samples)

        else:
            # Default to simple modulated signal
            baseband = self._generate_qpsk(num_samples)

        # Upconvert to carrier
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        signal = amplitude * baseband * carrier

        # Add noise if SNR specified
        if snr_db is not None:
            signal = self._add_noise(signal, snr_db)

        return signal

    def generate_jammed_signal(
        self,
        signal_type: SignalType,
        duration_sec: float,
        carrier_freq: float,
        jamming_type: JammingSimulationType,
        jammer_power_db: float = 10.0,
        signal_amplitude: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signal with jamming interference.

        Args:
            signal_type: Type of desired signal
            duration_sec: Duration in seconds
            carrier_freq: Carrier frequency in Hz
            jamming_type: Type of jamming to add
            jammer_power_db: Jammer power relative to signal (dB)
            signal_amplitude: Signal amplitude

        Returns:
            (jammed_signal, clean_signal) tuple
        """
        # Generate clean signal
        clean_signal = self.generate_clean_signal(
            signal_type, duration_sec, carrier_freq, signal_amplitude
        )

        # Generate jamming signal
        jammer = self._generate_jamming(
            jamming_type, len(clean_signal), carrier_freq, jammer_power_db
        )

        # Combine
        jammed_signal = clean_signal + jammer

        return jammed_signal, clean_signal

    def _generate_qpsk(self, num_samples: int) -> np.ndarray:
        """Generate QPSK modulated baseband signal"""
        # Symbol rate (arbitrary)
        symbols_per_sample = 10

        num_symbols = num_samples // symbols_per_sample

        # Generate random QPSK symbols
        symbols = np.random.randint(0, 4, num_symbols)
        qpsk_constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        baseband_symbols = qpsk_constellation[symbols]

        # Upsample and pulse shape
        baseband = np.repeat(baseband_symbols, symbols_per_sample)

        # Pad if necessary
        if len(baseband) < num_samples:
            baseband = np.pad(baseband, (0, num_samples - len(baseband)))

        return baseband[:num_samples]

    def _generate_qam16(self, num_samples: int) -> np.ndarray:
        """Generate 16-QAM modulated baseband signal"""
        symbols_per_sample = 10
        num_symbols = num_samples // symbols_per_sample

        # 16-QAM constellation
        qam16_constellation = np.array([
            -3-3j, -3-1j, -3+1j, -3+3j,
            -1-3j, -1-1j, -1+1j, -1+3j,
            +1-3j, +1-1j, +1+1j, +1+3j,
            +3-3j, +3-1j, +3+1j, +3+3j
        ]) / np.sqrt(10)

        symbols = np.random.randint(0, 16, num_symbols)
        baseband_symbols = qam16_constellation[symbols]

        baseband = np.repeat(baseband_symbols, symbols_per_sample)

        if len(baseband) < num_samples:
            baseband = np.pad(baseband, (0, num_samples - len(baseband)))

        return baseband[:num_samples]

    def _generate_ofdm(self, num_samples: int) -> np.ndarray:
        """Generate OFDM-like baseband signal"""
        # Simplified OFDM simulation
        num_subcarriers = 64
        num_symbols = num_samples // num_subcarriers

        # Generate random QAM symbols for each subcarrier
        data = np.random.randn(num_symbols, num_subcarriers) + \
               1j * np.random.randn(num_symbols, num_subcarriers)

        # IFFT to create OFDM symbols
        ofdm_symbols = np.fft.ifft(data, axis=1)

        # Flatten and take required samples
        baseband = ofdm_symbols.flatten()

        if len(baseband) < num_samples:
            baseband = np.pad(baseband, (0, num_samples - len(baseband)))

        return baseband[:num_samples]

    def _generate_gps_like(self, num_samples: int) -> np.ndarray:
        """Generate GPS-like spread spectrum signal"""
        # Simplified GPS simulation (C/A code like)
        chip_rate = 1.023e6  # GPS C/A chip rate
        chips_per_sample = int(self.sample_rate / chip_rate)

        num_chips = num_samples // chips_per_sample

        # Generate pseudo-random chip sequence
        chips = np.random.choice([-1, 1], num_chips)

        # Upsample
        baseband = np.repeat(chips, chips_per_sample)

        if len(baseband) < num_samples:
            baseband = np.pad(baseband, (0, num_samples - len(baseband)))

        return baseband[:num_samples].astype(complex)

    def _generate_jamming(
        self,
        jamming_type: JammingSimulationType,
        num_samples: int,
        carrier_freq: float,
        power_db: float
    ) -> np.ndarray:
        """
        Generate jamming signal.

        Args:
            jamming_type: Type of jamming
            num_samples: Number of samples
            carrier_freq: Carrier frequency
            power_db: Jammer power relative to signal

        Returns:
            Complex jamming signal
        """
        t = np.arange(num_samples) / self.sample_rate

        # Convert power from dB
        jammer_amplitude = 10 ** (power_db / 20)

        if jamming_type == JammingSimulationType.NONE:
            return np.zeros(num_samples, dtype=complex)

        elif jamming_type == JammingSimulationType.BARRAGE:
            # Wideband Gaussian noise
            noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
            return jammer_amplitude * noise

        elif jamming_type == JammingSimulationType.SPOT:
            # Narrowband CW tone (offset from carrier)
            jammer_freq = carrier_freq + 2e6  # 2 MHz offset
            carrier = np.exp(1j * 2 * np.pi * jammer_freq * t)
            return jammer_amplitude * carrier

        elif jamming_type == JammingSimulationType.SWEPT:
            # Frequency sweep
            sweep_rate = 10e6 / (num_samples / self.sample_rate)  # 10 MHz per duration
            instantaneous_freq = carrier_freq + sweep_rate * t
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
            return jammer_amplitude * np.exp(1j * phase)

        elif jamming_type == JammingSimulationType.PULSE:
            # Pulsed noise (30% duty cycle)
            pulse_mask = (np.random.rand(num_samples) < 0.3).astype(float)
            noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
            return jammer_amplitude * noise * pulse_mask

        elif jamming_type == JammingSimulationType.CHIRP:
            # Linear FM chirp
            chirp_bandwidth = 20e6
            chirp_rate = chirp_bandwidth / (num_samples / self.sample_rate)
            instantaneous_freq = carrier_freq + chirp_rate * t
            phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
            return jammer_amplitude * np.exp(1j * phase)

        else:
            return np.zeros(num_samples, dtype=complex)

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Add Gaussian noise to achieve specified SNR.

        Args:
            signal: Clean signal
            snr_db: Desired SNR in dB

        Returns:
            Noisy signal
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate complex Gaussian noise
        noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) / np.sqrt(2)
        noise *= np.sqrt(noise_power)

        return signal + noise

    def generate_test_suite(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate comprehensive test suite of all signal/jamming combinations.

        Returns:
            Dictionary of test scenarios
        """
        logger.info("Generating comprehensive test suite...")

        test_suite = {}
        duration = 0.0001  # 100 microseconds
        carrier_freq = 1e9  # 1 GHz

        # Clean signals
        for signal_type in [SignalType.CLEAN_TONE, SignalType.QPSK, SignalType.OFDM]:
            clean = self.generate_clean_signal(
                signal_type, duration, carrier_freq, snr_db=20
            )
            test_suite[f"clean_{signal_type.value}"] = {
                'signal': clean,
                'label': 'clean'
            }

        # Jammed signals
        for jamming_type in [JammingSimulationType.BARRAGE, JammingSimulationType.SPOT,
                             JammingSimulationType.PULSE, JammingSimulationType.SWEPT]:
            jammed, clean = self.generate_jammed_signal(
                SignalType.QPSK, duration, carrier_freq, jamming_type, jammer_power_db=10
            )
            test_suite[f"jammed_{jamming_type.value}"] = {
                'signal': jammed,
                'clean': clean,
                'label': jamming_type.value
            }

        logger.info(f"Generated {len(test_suite)} test scenarios")
        return test_suite


# Example usage
if __name__ == "__main__":
    print(DISCLAIMER)

    print("\n" + "=" * 60)
    print("ZELDA Signal Simulator - Educational Testing")
    print("=" * 60)

    # Create simulator
    simulator = RFSignalSimulator(sample_rate=40e6)

    # Test 1: Clean signals
    print("\n--- Test 1: Clean Signal Generation ---")
    for signal_type in [SignalType.CLEAN_TONE, SignalType.QPSK, SignalType.QAM16]:
        signal = simulator.generate_clean_signal(
            signal_type,
            duration_sec=0.0001,
            carrier_freq=1e9,
            snr_db=30
        )
        print(f"  {signal_type.value}: {len(signal)} samples, "
              f"power = {10*np.log10(np.mean(np.abs(signal)**2)):.2f} dB")

    # Test 2: Jammed signals
    print("\n--- Test 2: Jammed Signal Generation ---")
    for jamming_type in [JammingSimulationType.BARRAGE, JammingSimulationType.SPOT,
                         JammingSimulationType.PULSE]:
        jammed, clean = simulator.generate_jammed_signal(
            SignalType.QPSK,
            duration_sec=0.0001,
            carrier_freq=1e9,
            jamming_type=jamming_type,
            jammer_power_db=10
        )
        jammer_only = jammed - clean
        jsr = 10 * np.log10(np.mean(np.abs(jammer_only)**2) / np.mean(np.abs(clean)**2))
        print(f"  {jamming_type.value}: JSR = {jsr:.2f} dB")

    # Test 3: Comprehensive test suite
    print("\n--- Test 3: Comprehensive Test Suite ---")
    test_suite = simulator.generate_test_suite()
    print(f"  Generated {len(test_suite)} test scenarios")
    print("  Scenarios:")
    for name in sorted(test_suite.keys()):
        print(f"    - {name}")

    print("\n" + "=" * 60)
    print("✓ Signal simulator operational")
    print("✓ SOFTWARE SIMULATION ONLY - NO RF TRANSMISSION")
    print("✓ For testing and educational purposes only")
    print("=" * 60)
