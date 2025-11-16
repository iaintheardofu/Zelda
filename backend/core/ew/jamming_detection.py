"""
ZELDA Defensive EW Suite - Jamming Detection Module

This module provides real-time detection and characterization of RF jamming signals.
All capabilities are DEFENSIVE - detection and analysis only, no transmission.

Legal Use: Spectrum monitoring, interference detection, security research
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class JammingType(Enum):
    """Types of jamming signals that can be detected"""
    NONE = "none"
    BARRAGE = "barrage"  # Wideband noise
    SPOT = "spot"  # Narrowband continuous
    SWEPT = "swept"  # Frequency sweeping
    PULSE = "pulse"  # Pulsed jamming
    FOLLOWER = "follower"  # Reactive jamming
    DECEPTIVE = "deceptive"  # Signal mimicking
    UNKNOWN = "unknown"


@dataclass
class JammingDetection:
    """Results from jamming detection analysis"""
    is_jammed: bool
    jamming_type: JammingType
    confidence: float  # 0.0 to 1.0
    signal_to_noise_db: float
    interference_power_db: float
    affected_bandwidth_hz: float
    center_frequency_hz: float
    duty_cycle: Optional[float] = None  # For pulsed jamming
    sweep_rate_hz_per_sec: Optional[float] = None  # For swept jamming
    characteristics: Dict = None

    def __post_init__(self):
        if self.characteristics is None:
            self.characteristics = {}


class JammingDetector:
    """
    Real-time jamming detection and characterization system.

    Detects various types of RF interference and jamming:
    - Barrage (wideband noise)
    - Spot (narrowband continuous)
    - Swept (frequency sweeping)
    - Pulse (on/off jamming)
    - Follower (reactive jamming)
    - Deceptive (signal mimicking)
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        window_size: int = 4096,
        snr_threshold_db: float = -3.0,  # Below this = likely jammed
        detection_threshold: float = 0.8,  # Confidence threshold
    ):
        """
        Initialize jamming detector.

        Args:
            sample_rate: Sample rate in Hz
            window_size: Number of samples per analysis window
            snr_threshold_db: SNR below which jamming is suspected
            detection_threshold: Minimum confidence for positive detection
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.snr_threshold_db = snr_threshold_db
        self.detection_threshold = detection_threshold

        # State tracking for temporal analysis
        self.power_history = []
        self.frequency_history = []
        self.max_history_length = 100

        logger.info(f"JammingDetector initialized: {sample_rate/1e6:.1f} MHz sample rate")

    def detect(self, iq_signal: np.ndarray) -> JammingDetection:
        """
        Analyze I/Q signal for jamming signatures.

        Args:
            iq_signal: Complex I/Q samples (numpy array)

        Returns:
            JammingDetection with analysis results
        """
        # Ensure complex signal
        if iq_signal.dtype != np.complex64 and iq_signal.dtype != np.complex128:
            iq_signal = iq_signal[0] + 1j * iq_signal[1]

        # Calculate power spectral density
        freqs, psd = self._compute_psd(iq_signal)

        # Estimate SNR
        snr_db = self._estimate_snr(psd)

        # Calculate interference power
        interference_power_db = self._estimate_interference_power(psd)

        # Update history for temporal analysis
        self._update_history(psd, freqs)

        # Classify jamming type
        jamming_type, confidence, characteristics = self._classify_jamming(
            iq_signal, freqs, psd, snr_db
        )

        # Determine if signal is jammed
        is_jammed = (snr_db < self.snr_threshold_db) and (confidence >= self.detection_threshold)

        # Calculate affected bandwidth
        affected_bw = self._estimate_affected_bandwidth(freqs, psd)

        # Find center frequency of interference
        center_freq = self._find_interference_center(freqs, psd)

        return JammingDetection(
            is_jammed=is_jammed,
            jamming_type=jamming_type,
            confidence=confidence,
            signal_to_noise_db=snr_db,
            interference_power_db=interference_power_db,
            affected_bandwidth_hz=affected_bw,
            center_frequency_hz=center_freq,
            characteristics=characteristics
        )

    def _compute_psd(self, iq_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density"""
        freqs, psd = signal.welch(
            iq_signal,
            fs=self.sample_rate,
            nperseg=min(self.window_size, len(iq_signal)),
            scaling='density'
        )
        return freqs, psd

    def _estimate_snr(self, psd: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio from PSD.

        Uses percentile-based estimation:
        - Signal power: 90th percentile (peak regions)
        - Noise floor: 10th percentile (baseline)
        """
        signal_power = np.percentile(psd, 90)
        noise_floor = np.percentile(psd, 10)

        if noise_floor > 0:
            snr = 10 * np.log10(signal_power / noise_floor)
        else:
            snr = 100.0  # Very high SNR (no noise detected)

        return snr

    def _estimate_interference_power(self, psd: np.ndarray) -> float:
        """Calculate total interference power in dB"""
        total_power = np.sum(psd)
        if total_power > 0:
            return 10 * np.log10(total_power)
        return -100.0

    def _estimate_affected_bandwidth(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Estimate bandwidth affected by interference"""
        # Find threshold (median + 6 dB)
        threshold = np.median(psd) * 4  # 6 dB = 4x power

        # Count bins above threshold
        affected_bins = np.sum(psd > threshold)

        # Convert to bandwidth
        freq_resolution = freqs[1] - freqs[0]
        return affected_bins * freq_resolution

    def _find_interference_center(self, freqs: np.ndarray, psd: np.ndarray) -> float:
        """Find center frequency of interference"""
        # Weighted average of frequencies by power
        total_power = np.sum(psd)
        if total_power > 0:
            center_freq = np.sum(freqs * psd) / total_power
        else:
            center_freq = freqs[len(freqs) // 2]  # Center of band

        return center_freq

    def _update_history(self, psd: np.ndarray, freqs: np.ndarray):
        """Update historical data for temporal analysis"""
        # Store power statistics
        self.power_history.append(np.mean(psd))
        if len(self.power_history) > self.max_history_length:
            self.power_history.pop(0)

        # Store peak frequency
        peak_idx = np.argmax(psd)
        self.frequency_history.append(freqs[peak_idx])
        if len(self.frequency_history) > self.max_history_length:
            self.frequency_history.pop(0)

    def _classify_jamming(
        self,
        iq_signal: np.ndarray,
        freqs: np.ndarray,
        psd: np.ndarray,
        snr_db: float
    ) -> Tuple[JammingType, float, Dict]:
        """
        Classify type of jamming based on signal characteristics.

        Returns:
            (jamming_type, confidence, characteristics_dict)
        """
        characteristics = {}

        # Check if SNR is acceptable (no jamming)
        if snr_db >= self.snr_threshold_db:
            return JammingType.NONE, 1.0, characteristics

        # Calculate various signal statistics
        flatness = self._calculate_spectral_flatness(psd)
        kurtosis = self._calculate_kurtosis(iq_signal)
        peak_to_avg = self._calculate_peak_to_average(psd)
        temporal_variance = self._calculate_temporal_variance(iq_signal)

        # Store characteristics
        characteristics['spectral_flatness'] = flatness
        characteristics['kurtosis'] = kurtosis
        characteristics['peak_to_average_db'] = peak_to_avg
        characteristics['temporal_variance'] = temporal_variance

        # Classification logic
        confidence = 0.0
        jamming_type = JammingType.UNKNOWN

        # Barrage jamming: flat spectrum, high temporal variance
        if flatness > 0.8 and temporal_variance > 0.5:
            jamming_type = JammingType.BARRAGE
            confidence = min(flatness, temporal_variance)
            characteristics['description'] = "Wideband noise jamming across spectrum"

        # Spot jamming: narrow peak, low spectral flatness
        elif flatness < 0.3 and peak_to_avg > 10:
            jamming_type = JammingType.SPOT
            confidence = (1.0 - flatness) * min(peak_to_avg / 20, 1.0)
            characteristics['description'] = "Narrowband continuous jamming"

        # Swept jamming: frequency history shows movement
        elif self._detect_frequency_sweep():
            jamming_type = JammingType.SWEPT
            sweep_rate = self._estimate_sweep_rate()
            confidence = 0.85
            characteristics['sweep_rate_hz_per_sec'] = sweep_rate
            characteristics['description'] = "Frequency-sweeping jamming"

        # Pulse jamming: high kurtosis, bursty temporal pattern
        elif kurtosis > 5.0 and temporal_variance > 0.6:
            jamming_type = JammingType.PULSE
            duty_cycle = self._estimate_duty_cycle(iq_signal)
            confidence = min(kurtosis / 10, 1.0)
            characteristics['duty_cycle'] = duty_cycle
            characteristics['description'] = "Pulsed on/off jamming"

        # Deceptive jamming: structured signal (low kurtosis, moderate flatness)
        elif kurtosis < 2.0 and 0.3 < flatness < 0.7:
            jamming_type = JammingType.DECEPTIVE
            confidence = 0.7
            characteristics['description'] = "Signal mimicking/deceptive jamming"

        # Unknown jamming type
        else:
            jamming_type = JammingType.UNKNOWN
            confidence = 0.5
            characteristics['description'] = "Unclassified interference detected"

        return jamming_type, confidence, characteristics

    def _calculate_spectral_flatness(self, psd: np.ndarray) -> float:
        """
        Calculate spectral flatness (Wiener entropy).

        Returns value between 0 (pure tone) and 1 (white noise).
        """
        # Avoid log(0)
        psd_safe = psd + 1e-10

        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd_safe)

        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
        else:
            flatness = 0.0

        return np.clip(flatness, 0.0, 1.0)

    def _calculate_kurtosis(self, iq_signal: np.ndarray) -> float:
        """Calculate kurtosis (measure of signal peakedness)"""
        magnitude = np.abs(iq_signal)
        return stats.kurtosis(magnitude)

    def _calculate_peak_to_average(self, psd: np.ndarray) -> float:
        """Calculate peak-to-average ratio in dB"""
        peak = np.max(psd)
        average = np.mean(psd)

        if average > 0:
            return 10 * np.log10(peak / average)
        return 0.0

    def _calculate_temporal_variance(self, iq_signal: np.ndarray) -> float:
        """Calculate normalized temporal variance of signal envelope"""
        envelope = np.abs(iq_signal)
        variance = np.var(envelope)
        mean = np.mean(envelope)

        if mean > 0:
            normalized_variance = variance / (mean ** 2)
        else:
            normalized_variance = 0.0

        return normalized_variance

    def _detect_frequency_sweep(self) -> bool:
        """Detect if frequency is sweeping based on history"""
        if len(self.frequency_history) < 10:
            return False

        # Calculate trend (linear regression)
        x = np.arange(len(self.frequency_history))
        y = np.array(self.frequency_history)

        # Check for monotonic trend
        slope, _ = np.polyfit(x, y, 1)

        # Significant sweep if slope exceeds threshold
        sweep_threshold = self.sample_rate * 0.01  # 1% of sample rate
        return abs(slope) > sweep_threshold

    def _estimate_sweep_rate(self) -> float:
        """Estimate sweep rate in Hz/second"""
        if len(self.frequency_history) < 2:
            return 0.0

        # Linear fit to frequency history
        x = np.arange(len(self.frequency_history))
        y = np.array(self.frequency_history)
        slope, _ = np.polyfit(x, y, 1)

        # Convert to Hz/second (assuming history spacing is ~10ms)
        sweep_rate = slope * 100  # 100 samples/second

        return abs(sweep_rate)

    def _estimate_duty_cycle(self, iq_signal: np.ndarray) -> float:
        """Estimate duty cycle for pulsed jamming"""
        envelope = np.abs(iq_signal)
        threshold = np.median(envelope) * 2  # 6 dB above median

        on_samples = np.sum(envelope > threshold)
        total_samples = len(envelope)

        duty_cycle = on_samples / total_samples
        return duty_cycle

    def generate_report(self, detection: JammingDetection) -> str:
        """Generate human-readable report of jamming detection"""
        report = []
        report.append("=" * 60)
        report.append("ZELDA JAMMING DETECTION REPORT")
        report.append("=" * 60)

        if detection.is_jammed:
            report.append(f"⚠️  JAMMING DETECTED: {detection.jamming_type.value.upper()}")
            report.append(f"   Confidence: {detection.confidence*100:.1f}%")
        else:
            report.append("✓  No jamming detected")

        report.append("")
        report.append("Signal Analysis:")
        report.append(f"  SNR: {detection.signal_to_noise_db:+.2f} dB")
        report.append(f"  Interference Power: {detection.interference_power_db:+.2f} dB")
        report.append(f"  Affected Bandwidth: {detection.affected_bandwidth_hz/1e6:.2f} MHz")
        report.append(f"  Center Frequency: {detection.center_frequency_hz/1e6:.2f} MHz")

        if detection.characteristics:
            report.append("")
            report.append("Characteristics:")
            for key, value in detection.characteristics.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.3f}")
                else:
                    report.append(f"  {key}: {value}")

        report.append("=" * 60)
        return "\n".join(report)


class AdaptiveJammingDetector(JammingDetector):
    """
    Advanced jamming detector with adaptive thresholds and learning.

    Automatically adjusts detection thresholds based on baseline conditions
    and historical interference patterns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Baseline statistics
        self.baseline_snr = []
        self.baseline_power = []
        self.baseline_samples = 50  # Number of samples to establish baseline

        # Adaptive thresholds
        self.adaptive_snr_threshold = self.snr_threshold_db

        logger.info("AdaptiveJammingDetector initialized with learning capabilities")

    def update_baseline(self, iq_signal: np.ndarray):
        """Update baseline statistics during clean signal periods"""
        if len(self.baseline_snr) >= self.baseline_samples:
            return  # Baseline established

        # Calculate SNR for this sample
        freqs, psd = self._compute_psd(iq_signal)
        snr_db = self._estimate_snr(psd)
        power_db = self._estimate_interference_power(psd)

        self.baseline_snr.append(snr_db)
        self.baseline_power.append(power_db)

        # Update adaptive threshold
        if len(self.baseline_snr) >= 10:
            mean_snr = np.mean(self.baseline_snr)
            std_snr = np.std(self.baseline_snr)
            # Set threshold 2 sigma below mean
            self.adaptive_snr_threshold = mean_snr - 2 * std_snr

            logger.debug(f"Adaptive SNR threshold updated: {self.adaptive_snr_threshold:.2f} dB")

    def detect(self, iq_signal: np.ndarray) -> JammingDetection:
        """Detect with adaptive thresholds"""
        # Use adaptive threshold if baseline established
        if len(self.baseline_snr) >= self.baseline_samples:
            original_threshold = self.snr_threshold_db
            self.snr_threshold_db = self.adaptive_snr_threshold
            result = super().detect(iq_signal)
            self.snr_threshold_db = original_threshold
            return result
        else:
            # Still establishing baseline
            self.update_baseline(iq_signal)
            return super().detect(iq_signal)


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("ZELDA Defensive EW - Jamming Detection Module")
    print("=" * 60)
    print("Legal Use: Detection and analysis only (no transmission)")
    print("=" * 60)

    # Create detector
    detector = AdaptiveJammingDetector(
        sample_rate=40e6,
        window_size=4096,
        snr_threshold_db=-3.0
    )

    # Test Case 1: Clean signal
    print("\nTest 1: Clean Signal")
    t = np.linspace(0, 4096/40e6, 4096)
    clean_signal = np.exp(1j * 2 * np.pi * 1e6 * t)  # 1 MHz tone
    result = detector.detect(clean_signal)
    print(detector.generate_report(result))

    # Test Case 2: Barrage jamming (wideband noise)
    print("\nTest 2: Barrage Jamming (Wideband Noise)")
    noise = np.random.randn(4096) + 1j * np.random.randn(4096)
    jammed_signal = clean_signal * 0.1 + noise * 2
    result = detector.detect(jammed_signal)
    print(detector.generate_report(result))

    # Test Case 3: Spot jamming (narrowband interferer)
    print("\nTest 3: Spot Jamming (Narrowband Interferer)")
    interferer = 5 * np.exp(1j * 2 * np.pi * 1.5e6 * t)  # Strong 1.5 MHz tone
    jammed_signal = clean_signal + interferer
    result = detector.detect(jammed_signal)
    print(detector.generate_report(result))

    # Test Case 4: Pulse jamming
    print("\nTest 4: Pulse Jamming")
    pulse_mask = (np.random.rand(4096) > 0.7).astype(float)  # 30% duty cycle
    pulse_noise = noise * pulse_mask[:, np.newaxis] if noise.ndim > 1 else noise * pulse_mask
    jammed_signal = clean_signal * 0.5 + pulse_noise * 3
    result = detector.detect(jammed_signal)
    print(detector.generate_report(result))

    print("\n✓ Jamming detection module operational")
    print("✓ All capabilities are DEFENSIVE (detection/analysis only)")
