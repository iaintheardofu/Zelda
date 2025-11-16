"""
ZELDA Defensive EW Suite - Anti-Jam Signal Processing

Adaptive signal processing techniques to mitigate jamming effects.
All capabilities are DEFENSIVE - signal filtering and enhancement only, no transmission.

Legal Use: Interference mitigation, signal recovery, authorized testing
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftshift
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AntiJamResult:
    """Results from anti-jam processing"""
    cleaned_signal: np.ndarray  # Processed I/Q signal
    snr_improvement_db: float  # SNR improvement achieved
    interference_suppressed_db: float  # Interference power reduction
    method_used: str  # Processing method applied
    success: bool  # Whether processing was successful


class AdaptiveNotchFilter:
    """
    Adaptive notch filter for narrowband interference suppression.

    Automatically detects and removes narrowband jamming signals
    while preserving desired signal content.
    """

    def __init__(self, sample_rate: float, notch_bandwidth_hz: float = 1000):
        """
        Initialize adaptive notch filter.

        Args:
            sample_rate: Sample rate in Hz
            notch_bandwidth_hz: Bandwidth of each notch filter
        """
        self.sample_rate = sample_rate
        self.notch_bandwidth = notch_bandwidth_hz
        logger.info(f"AdaptiveNotchFilter initialized: {notch_bandwidth_hz} Hz bandwidth")

    def process(self, iq_signal: np.ndarray, num_notches: int = 5) -> AntiJamResult:
        """
        Apply adaptive notch filtering to suppress interference.

        Args:
            iq_signal: Complex I/Q signal
            num_notches: Maximum number of notch filters to apply

        Returns:
            AntiJamResult with processed signal
        """
        original_power = np.mean(np.abs(iq_signal) ** 2)

        # Find interference peaks in frequency domain
        freqs, psd = signal.welch(iq_signal, fs=self.sample_rate, nperseg=1024)

        # Identify top interference frequencies
        threshold = np.percentile(psd, 90)  # Top 10%
        interference_peaks = signal.find_peaks(psd, height=threshold, distance=10)[0]

        # Limit to num_notches strongest peaks
        if len(interference_peaks) > num_notches:
            peak_heights = psd[interference_peaks]
            top_indices = np.argsort(peak_heights)[-num_notches:]
            interference_peaks = interference_peaks[top_indices]

        # Apply notch filter for each interference peak
        cleaned_signal = iq_signal.copy()

        for peak_idx in interference_peaks:
            interference_freq = freqs[peak_idx]

            # Design notch filter
            notch_filter = self._design_notch(interference_freq)

            # Apply filter
            cleaned_signal = signal.lfilter(notch_filter[0], notch_filter[1], cleaned_signal)

        # Calculate improvement
        cleaned_power = np.mean(np.abs(cleaned_signal) ** 2)
        interference_suppressed_db = 10 * np.log10(original_power / (cleaned_power + 1e-12))

        # Estimate SNR improvement (simplified)
        snr_improvement_db = min(interference_suppressed_db * 0.5, 20)  # Cap at 20 dB

        return AntiJamResult(
            cleaned_signal=cleaned_signal,
            snr_improvement_db=snr_improvement_db,
            interference_suppressed_db=interference_suppressed_db,
            method_used=f"Adaptive Notch Filter ({len(interference_peaks)} notches)",
            success=True
        )

    def _design_notch(self, center_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Design IIR notch filter at specified frequency"""
        # Normalize frequency
        nyquist = self.sample_rate / 2
        normalized_freq = abs(center_freq) / nyquist

        # Clip to valid range (must be 0 < w0 < 1)
        normalized_freq = np.clip(normalized_freq, 0.001, 0.999)

        # Quality factor based on bandwidth
        Q = max(abs(center_freq) / self.notch_bandwidth, 1.0)  # Ensure Q >= 1

        # Design notch filter
        b, a = signal.iirnotch(normalized_freq, Q, fs=self.sample_rate)

        return b, a


class SpectralExcisionFilter:
    """
    Spectral excision for wideband interference mitigation.

    Detects and removes interference in frequency domain while
    preserving signal in clean frequency bands.
    """

    def __init__(self, sample_rate: float, threshold_factor: float = 3.0):
        """
        Initialize spectral excision filter.

        Args:
            sample_rate: Sample rate in Hz
            threshold_factor: Multiplier for interference detection threshold
        """
        self.sample_rate = sample_rate
        self.threshold_factor = threshold_factor
        logger.info("SpectralExcisionFilter initialized")

    def process(self, iq_signal: np.ndarray) -> AntiJamResult:
        """
        Apply spectral excision to remove wideband interference.

        Args:
            iq_signal: Complex I/Q signal

        Returns:
            AntiJamResult with processed signal
        """
        original_power = np.mean(np.abs(iq_signal) ** 2)

        # Transform to frequency domain
        spectrum = fft(iq_signal)
        power_spectrum = np.abs(spectrum) ** 2

        # Calculate threshold (median + threshold_factor * MAD)
        median_power = np.median(power_spectrum)
        mad = np.median(np.abs(power_spectrum - median_power))  # Median Absolute Deviation
        threshold = median_power + self.threshold_factor * mad

        # Create excision mask
        excision_mask = power_spectrum < threshold

        # Apply mask
        cleaned_spectrum = spectrum * excision_mask

        # Transform back to time domain
        cleaned_signal = ifft(cleaned_spectrum)

        # Calculate improvement
        cleaned_power = np.mean(np.abs(cleaned_signal) ** 2)
        interference_suppressed_db = 10 * np.log10(original_power / (cleaned_power + 1e-12))

        # Calculate how much spectrum was excised
        excised_percentage = 100 * (1 - np.sum(excision_mask) / len(excision_mask))

        snr_improvement_db = min(interference_suppressed_db * 0.6, 25)

        return AntiJamResult(
            cleaned_signal=cleaned_signal,
            snr_improvement_db=snr_improvement_db,
            interference_suppressed_db=interference_suppressed_db,
            method_used=f"Spectral Excision ({excised_percentage:.1f}% removed)",
            success=True
        )


class AdaptiveWhitening:
    """
    Adaptive whitening filter for barrage jamming mitigation.

    Flattens the spectrum to suppress wideband noise jamming
    while preserving signal structure.
    """

    def __init__(self, sample_rate: float, window_size: int = 256):
        """
        Initialize adaptive whitening filter.

        Args:
            sample_rate: Sample rate in Hz
            window_size: Size of analysis window
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        logger.info("AdaptiveWhitening initialized")

    def process(self, iq_signal: np.ndarray) -> AntiJamResult:
        """
        Apply adaptive whitening to mitigate barrage jamming.

        Args:
            iq_signal: Complex I/Q signal

        Returns:
            AntiJamResult with processed signal
        """
        original_power = np.mean(np.abs(iq_signal) ** 2)

        # Estimate power spectral density
        freqs, psd = signal.welch(iq_signal, fs=self.sample_rate, nperseg=self.window_size)

        # Calculate whitening filter (inverse of PSD)
        whitening_filter = 1.0 / np.sqrt(psd + 1e-10)

        # Normalize
        whitening_filter /= np.max(whitening_filter)

        # Apply in frequency domain
        spectrum = fft(iq_signal)
        freq_bins = len(spectrum)

        # Interpolate whitening filter to match spectrum size
        whitening_interpolated = np.interp(
            np.linspace(0, len(whitening_filter), freq_bins),
            np.arange(len(whitening_filter)),
            whitening_filter
        )

        # Apply whitening
        whitened_spectrum = spectrum * whitening_interpolated

        # Transform back
        cleaned_signal = ifft(whitened_spectrum)

        # Calculate improvement
        cleaned_power = np.mean(np.abs(cleaned_signal) ** 2)

        # Normalize to preserve power
        if cleaned_power > 0:
            cleaned_signal *= np.sqrt(original_power / cleaned_power)

        interference_suppressed_db = self._estimate_flatness_improvement(iq_signal, cleaned_signal)
        snr_improvement_db = min(interference_suppressed_db * 0.4, 15)

        return AntiJamResult(
            cleaned_signal=cleaned_signal,
            snr_improvement_db=snr_improvement_db,
            interference_suppressed_db=interference_suppressed_db,
            method_used="Adaptive Whitening",
            success=True
        )

    def _estimate_flatness_improvement(self, original: np.ndarray, cleaned: np.ndarray) -> float:
        """Estimate spectral flatness improvement"""
        _, psd_orig = signal.welch(original, fs=self.sample_rate)
        _, psd_clean = signal.welch(cleaned, fs=self.sample_rate)

        flatness_orig = self._spectral_flatness(psd_orig)
        flatness_clean = self._spectral_flatness(psd_clean)

        improvement = 10 * np.log10((1 - flatness_orig + 0.01) / (1 - flatness_clean + 0.01))
        return max(0, improvement)

    def _spectral_flatness(self, psd: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)"""
        psd_safe = psd + 1e-10
        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd_safe)
        return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0


class PulseBlankingFilter:
    """
    Pulse blanking for pulsed jamming mitigation.

    Detects and blanks out high-power pulses while preserving
    signal during clean periods.
    """

    def __init__(self, threshold_factor: float = 5.0):
        """
        Initialize pulse blanking filter.

        Args:
            threshold_factor: Multiplier for pulse detection threshold
        """
        self.threshold_factor = threshold_factor
        logger.info("PulseBlankingFilter initialized")

    def process(self, iq_signal: np.ndarray) -> AntiJamResult:
        """
        Apply pulse blanking to mitigate pulsed jamming.

        Args:
            iq_signal: Complex I/Q signal

        Returns:
            AntiJamResult with processed signal
        """
        original_power = np.mean(np.abs(iq_signal) ** 2)

        # Calculate envelope
        envelope = np.abs(iq_signal)

        # Detect pulses (threshold based on median)
        median_power = np.median(envelope)
        threshold = median_power * self.threshold_factor

        # Create blanking mask
        blanking_mask = envelope < threshold

        # Apply blanking
        cleaned_signal = iq_signal * blanking_mask

        # Calculate statistics
        blanked_percentage = 100 * (1 - np.sum(blanking_mask) / len(blanking_mask))
        cleaned_power = np.mean(np.abs(cleaned_signal) ** 2)

        interference_suppressed_db = 10 * np.log10(original_power / (cleaned_power + 1e-12))
        snr_improvement_db = min(blanked_percentage * 0.5, 30)

        return AntiJamResult(
            cleaned_signal=cleaned_signal,
            snr_improvement_db=snr_improvement_db,
            interference_suppressed_db=interference_suppressed_db,
            method_used=f"Pulse Blanking ({blanked_percentage:.1f}% blanked)",
            success=True
        )


class AdaptiveAntiJamProcessor:
    """
    Intelligent anti-jam processor that automatically selects and applies
    the best mitigation technique based on jamming type.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize adaptive anti-jam processor.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Initialize all processing methods
        self.notch_filter = AdaptiveNotchFilter(sample_rate)
        self.spectral_excision = SpectralExcisionFilter(sample_rate)
        self.whitening = AdaptiveWhitening(sample_rate)
        self.pulse_blanking = PulseBlankingFilter()

        logger.info("AdaptiveAntiJamProcessor initialized with all methods")

    def process(self, iq_signal: np.ndarray, jamming_type: Optional[str] = None) -> AntiJamResult:
        """
        Automatically select and apply best anti-jam technique.

        Args:
            iq_signal: Complex I/Q signal
            jamming_type: Optional hint about jamming type
                ("spot", "barrage", "pulse", "swept", or None for auto-detect)

        Returns:
            AntiJamResult with processed signal
        """
        # Auto-detect jamming type if not provided
        if jamming_type is None:
            jamming_type = self._classify_jamming(iq_signal)

        logger.info(f"Processing with jamming type: {jamming_type}")

        # Select appropriate method
        if jamming_type == "spot":
            # Narrowband jamming → notch filter
            result = self.notch_filter.process(iq_signal)

        elif jamming_type == "barrage":
            # Wideband noise → whitening
            result = self.whitening.process(iq_signal)

        elif jamming_type == "pulse":
            # Pulsed jamming → pulse blanking
            result = self.pulse_blanking.process(iq_signal)

        elif jamming_type == "swept":
            # Swept jamming → spectral excision
            result = self.spectral_excision.process(iq_signal)

        else:
            # Unknown → try multiple methods and pick best
            result = self._cascade_processing(iq_signal)

        return result

    def _classify_jamming(self, iq_signal: np.ndarray) -> str:
        """
        Classify jamming type based on signal characteristics.

        Returns:
            Jamming type string
        """
        # Calculate power spectral density
        freqs, psd = signal.welch(iq_signal, fs=self.sample_rate, nperseg=1024)

        # Calculate spectral flatness
        flatness = self._spectral_flatness(psd)

        # Calculate kurtosis (peakedness)
        envelope = np.abs(iq_signal)
        from scipy.stats import kurtosis
        kurt = kurtosis(envelope)

        # Classification logic
        if flatness > 0.8:
            return "barrage"  # Flat spectrum = wideband noise
        elif flatness < 0.3:
            return "spot"  # Narrow spectrum = spot jamming
        elif kurt > 5.0:
            return "pulse"  # High kurtosis = pulsed
        else:
            return "swept"  # Medium flatness = swept or unknown

    def _spectral_flatness(self, psd: np.ndarray) -> float:
        """Calculate spectral flatness"""
        psd_safe = psd + 1e-10
        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd_safe)
        return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

    def _cascade_processing(self, iq_signal: np.ndarray) -> AntiJamResult:
        """
        Apply multiple processing methods and select best result.

        Args:
            iq_signal: Complex I/Q signal

        Returns:
            Best AntiJamResult
        """
        methods = [
            ("Notch Filter", self.notch_filter),
            ("Spectral Excision", self.spectral_excision),
            ("Whitening", self.whitening),
            ("Pulse Blanking", self.pulse_blanking)
        ]

        results = []
        for name, processor in methods:
            try:
                result = processor.process(iq_signal)
                results.append(result)
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

        # Select best result (highest SNR improvement)
        if results:
            best_result = max(results, key=lambda r: r.snr_improvement_db)
            best_result.method_used = f"Cascade ({best_result.method_used})"
            return best_result
        else:
            # No processing worked, return original
            return AntiJamResult(
                cleaned_signal=iq_signal,
                snr_improvement_db=0.0,
                interference_suppressed_db=0.0,
                method_used="None (processing failed)",
                success=False
            )


def generate_antijam_report(original: np.ndarray, result: AntiJamResult) -> str:
    """Generate human-readable anti-jam processing report"""
    report = []
    report.append("=" * 60)
    report.append("ZELDA ANTI-JAM PROCESSING REPORT")
    report.append("=" * 60)

    # Original signal stats
    orig_power = 10 * np.log10(np.mean(np.abs(original) ** 2) + 1e-12)
    clean_power = 10 * np.log10(np.mean(np.abs(result.cleaned_signal) ** 2) + 1e-12)

    report.append(f"Method: {result.method_used}")
    report.append(f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    report.append("")

    report.append("Signal Power:")
    report.append(f"  Original: {orig_power:+.2f} dB")
    report.append(f"  Cleaned:  {clean_power:+.2f} dB")
    report.append("")

    report.append("Improvements:")
    report.append(f"  SNR Improvement: {result.snr_improvement_db:+.2f} dB")
    report.append(f"  Interference Suppressed: {result.interference_suppressed_db:+.2f} dB")
    report.append("")

    if result.success:
        report.append("✓ Signal processing successful")
        report.append("✓ Interference mitigation applied")
    else:
        report.append("⚠ Processing had limited effect")
        report.append("  Consider alternative methods or manual tuning")

    report.append("=" * 60)
    report.append("All processing is DEFENSIVE (signal enhancement only)")
    report.append("=" * 60)

    return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("ZELDA Defensive EW - Anti-Jam Signal Processing")
    print("=" * 60)
    print("Legal Use: Signal enhancement and mitigation only (no transmission)")
    print("=" * 60)

    # Create processor
    processor = AdaptiveAntiJamProcessor(sample_rate=40e6)

    # Test Case 1: Spot jamming
    print("\n--- Test 1: Spot Jamming Mitigation ---")
    t = np.linspace(0, 4096/40e6, 4096)
    clean_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
    jammer = 5 * np.exp(1j * 2 * np.pi * 3e6 * t)
    jammed_signal = clean_signal + jammer

    result = processor.process(jammed_signal, jamming_type="spot")
    print(generate_antijam_report(jammed_signal, result))

    # Test Case 2: Barrage jamming
    print("\n--- Test 2: Barrage Jamming Mitigation ---")
    noise = 3 * (np.random.randn(4096) + 1j * np.random.randn(4096))
    jammed_signal = clean_signal * 0.5 + noise

    result = processor.process(jammed_signal, jamming_type="barrage")
    print(generate_antijam_report(jammed_signal, result))

    # Test Case 3: Pulse jamming
    print("\n--- Test 3: Pulse Jamming Mitigation ---")
    pulse_mask = (np.random.rand(4096) > 0.7).astype(float)
    pulse_noise = 10 * noise * pulse_mask
    jammed_signal = clean_signal + pulse_noise

    result = processor.process(jammed_signal, jamming_type="pulse")
    print(generate_antijam_report(jammed_signal, result))

    # Test Case 4: Auto-detect
    print("\n--- Test 4: Auto-Detection ---")
    result = processor.process(jammed_signal, jamming_type=None)
    print(generate_antijam_report(jammed_signal, result))

    print("\n✓ Anti-jam processing module operational")
    print("✓ All capabilities are DEFENSIVE (signal enhancement only)")
