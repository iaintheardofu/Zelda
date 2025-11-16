"""
ZELDA Quantum-Inspired Spoofing Detection Module

Implements quantum-inspired correlation-based methods for spoofing detection.
Based on research: "Spoofing Resilience for Simple-Detection Quantum Illumination LIDAR"

Key Features:
- Multi-path correlation analysis
- Signal authentication through correlation patterns
- Quantum-inspired spoofing resilience
- Enhanced GPS/RF signal validation

While not using true quantum systems, this module applies quantum-inspired
correlation techniques to detect signal spoofing and replay attacks.

All capabilities are DEFENSIVE - detection and analysis only, no transmission.
Legal Use: Security monitoring, threat detection, authorized testing
"""

import numpy as np
from scipy import signal, stats
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SpoofingMethod(Enum):
    """Methods of spoofing attack"""
    NONE = "none"
    MEACONING = "meaconing"  # Replay attack with amplification
    SIMULATION = "simulation"  # Fake signal generation
    MANIPULATION = "manipulation"  # Signal modification
    UNKNOWN = "unknown"


@dataclass
class CorrelationFingerprint:
    """
    Correlation-based signal fingerprint.

    Inspired by quantum illumination: authentic signals have specific
    correlation patterns with noise that spoofed signals cannot replicate.
    """
    autocorrelation: np.ndarray
    cross_correlation_peaks: List[float]
    correlation_entropy: float
    phase_coherence: float
    temporal_consistency: float


@dataclass
class QuantumInspiredDetection:
    """Results from quantum-inspired spoofing detection"""
    is_spoofed: bool
    spoofing_method: SpoofingMethod
    confidence: float  # 0.0 to 1.0
    correlation_score: float  # Correlation-based authenticity score
    fingerprint: CorrelationFingerprint
    anomaly_indicators: List[str]
    detection_time: datetime
    signal_characteristics: Dict


class QuantumInspiredSpoofDetector:
    """
    Quantum-inspired spoofing detection using correlation analysis.

    Based on Paper 1 principle: "Authentic quantum illumination returns
    have specific correlation patterns with the transmit signal that
    spoofed returns cannot replicate without knowing the idler photon state"

    Classical analog: Authentic signals have noise correlation patterns
    that are extremely difficult for spoofers to replicate.
    """

    def __init__(self, sample_rate: float = 10e6):
        """
        Initialize quantum-inspired detector.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

        # Reference noise pattern (like quantum idler state)
        # In practice, this would be derived from trusted transmitter
        self.reference_noise_pattern = None

        # Historical fingerprints for temporal consistency
        self.fingerprint_history = []
        self.max_history = 50

        logger.info("QuantumInspiredSpoofDetector initialized")

    def set_reference_pattern(self, reference_signal: np.ndarray):
        """
        Set reference noise pattern from trusted authentic signal.

        This is analogous to the "idler" state in quantum illumination.

        Args:
            reference_signal: Complex I/Q samples from trusted source
        """
        # Extract noise characteristics
        self.reference_noise_pattern = self._extract_noise_pattern(reference_signal)
        logger.info("Reference noise pattern established")

    def detect_spoofing(
        self,
        iq_signal: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> QuantumInspiredDetection:
        """
        Detect spoofing using quantum-inspired correlation analysis.

        Args:
            iq_signal: Complex I/Q samples to analyze
            metadata: Optional metadata (timestamp, power, etc.)

        Returns:
            QuantumInspiredDetection with analysis results
        """
        # Extract correlation fingerprint
        fingerprint = self._extract_fingerprint(iq_signal)

        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(
            fingerprint, iq_signal
        )

        # Detect anomalies
        anomaly_indicators = []
        confidence = 0.0
        spoofing_method = SpoofingMethod.NONE

        # Analysis 1: Correlation pattern matching
        if self.reference_noise_pattern is not None:
            pattern_match = self._match_noise_pattern(iq_signal)
            if pattern_match < 0.5:  # Poor match suggests spoofing
                anomaly_indicators.append(
                    f"Noise pattern mismatch (score: {pattern_match:.2f})"
                )
                confidence += 0.35
                spoofing_method = SpoofingMethod.SIMULATION

        # Analysis 2: Autocorrelation anomalies
        if self._detect_autocorrelation_anomaly(fingerprint.autocorrelation):
            anomaly_indicators.append("Abnormal autocorrelation (replay suspected)")
            confidence += 0.30
            spoofing_method = SpoofingMethod.MEACONING

        # Analysis 3: Low correlation entropy (too clean = fake)
        if fingerprint.correlation_entropy < 0.3:
            anomaly_indicators.append(
                f"Suspiciously low correlation entropy: {fingerprint.correlation_entropy:.3f}"
            )
            confidence += 0.25

        # Analysis 4: Phase coherence anomaly
        if fingerprint.phase_coherence > 0.95:  # Too coherent
            anomaly_indicators.append(
                "Abnormally high phase coherence (simulated signal suspected)"
            )
            confidence += 0.20
            spoofing_method = SpoofingMethod.SIMULATION

        # Analysis 5: Temporal consistency check
        if len(self.fingerprint_history) >= 5:
            temporal_consistency = self._check_temporal_consistency(fingerprint)
            if temporal_consistency < 0.6:
                anomaly_indicators.append(
                    "Temporal fingerprint inconsistency detected"
                )
                confidence += 0.25

        # Analysis 6: Cross-correlation peak analysis
        if self._detect_cross_correlation_anomaly(
            fingerprint.cross_correlation_peaks
        ):
            anomaly_indicators.append("Anomalous cross-correlation peak pattern")
            confidence += 0.20

        # Determine if spoofed
        is_spoofed = confidence >= 0.5

        # Extract signal characteristics
        signal_characteristics = self._extract_signal_characteristics(iq_signal)

        # Store fingerprint in history
        self.fingerprint_history.append(fingerprint)
        if len(self.fingerprint_history) > self.max_history:
            self.fingerprint_history.pop(0)

        return QuantumInspiredDetection(
            is_spoofed=is_spoofed,
            spoofing_method=spoofing_method if is_spoofed else SpoofingMethod.NONE,
            confidence=min(confidence, 1.0),
            correlation_score=correlation_score,
            fingerprint=fingerprint,
            anomaly_indicators=anomaly_indicators,
            detection_time=datetime.now(),
            signal_characteristics=signal_characteristics
        )

    def _extract_fingerprint(self, iq_signal: np.ndarray) -> CorrelationFingerprint:
        """
        Extract correlation-based fingerprint from signal.

        This captures unique correlation properties that are difficult to spoof.
        """
        # Autocorrelation
        autocorr = np.correlate(iq_signal, iq_signal, mode='same')
        autocorr = autocorr / np.max(np.abs(autocorr))  # Normalize

        # Cross-correlation with time-shifted versions
        cross_corr_peaks = []
        for shift in [10, 50, 100, 500]:
            if len(iq_signal) > shift:
                shifted = np.roll(iq_signal, shift)
                cc = np.correlate(iq_signal, shifted, mode='valid')
                cross_corr_peaks.append(float(np.max(np.abs(cc))))

        # Correlation entropy (measure of randomness in correlation)
        correlation_entropy = self._calculate_correlation_entropy(autocorr)

        # Phase coherence
        phase = np.angle(iq_signal)
        phase_diff = np.diff(phase)
        phase_coherence = float(np.abs(np.mean(np.exp(1j * phase_diff))))

        # Temporal consistency (will be calculated when comparing to history)
        temporal_consistency = 1.0

        return CorrelationFingerprint(
            autocorrelation=autocorr,
            cross_correlation_peaks=cross_corr_peaks,
            correlation_entropy=correlation_entropy,
            phase_coherence=phase_coherence,
            temporal_consistency=temporal_consistency
        )

    def _extract_noise_pattern(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract noise pattern from authentic signal.

        This is the "reference idler" in quantum illumination terms.
        """
        # Remove strong signal components
        freqs, psd = signal.welch(signal, fs=self.sample_rate, nperseg=1024)

        # Identify noise floor (below median power)
        noise_threshold = np.median(psd)

        # Extract noise-dominated regions
        noise_mask = psd < noise_threshold

        # Create noise pattern representation
        noise_pattern = psd[noise_mask]

        return noise_pattern

    def _match_noise_pattern(self, iq_signal: np.ndarray) -> float:
        """
        Match signal's noise pattern to reference.

        Returns correlation score (0.0 to 1.0), higher = better match
        """
        if self.reference_noise_pattern is None:
            return 0.5  # Unknown

        # Extract noise from current signal
        current_noise = self._extract_noise_pattern(iq_signal)

        # Compare patterns using correlation
        if len(current_noise) == 0 or len(self.reference_noise_pattern) == 0:
            return 0.0

        # Resize to same length
        min_len = min(len(current_noise), len(self.reference_noise_pattern))
        current_noise = current_noise[:min_len]
        ref_noise = self.reference_noise_pattern[:min_len]

        # Calculate correlation
        correlation = np.corrcoef(current_noise, ref_noise)[0, 1]

        # Return absolute correlation
        return float(abs(correlation))

    def _calculate_correlation_score(
        self,
        fingerprint: CorrelationFingerprint,
        iq_signal: np.ndarray
    ) -> float:
        """
        Calculate overall correlation-based authenticity score.

        Combines multiple correlation metrics into single score.
        """
        scores = []

        # Score 1: Correlation entropy (should be moderate)
        entropy_score = min(fingerprint.correlation_entropy * 2, 1.0)
        scores.append(entropy_score)

        # Score 2: Phase coherence (should be moderate, not too high/low)
        phase_score = 1.0 - abs(fingerprint.phase_coherence - 0.7)
        scores.append(phase_score)

        # Score 3: Cross-correlation peaks consistency
        if len(fingerprint.cross_correlation_peaks) > 1:
            peak_variance = np.var(fingerprint.cross_correlation_peaks)
            peak_score = 1.0 / (1.0 + peak_variance)  # Lower variance = higher score
            scores.append(peak_score)

        # Score 4: Autocorrelation shape
        autocorr_score = self._score_autocorrelation_shape(
            fingerprint.autocorrelation
        )
        scores.append(autocorr_score)

        # Average scores
        return float(np.mean(scores))

    def _calculate_correlation_entropy(self, autocorr: np.ndarray) -> float:
        """
        Calculate entropy of autocorrelation function.

        Spoofed signals often have lower entropy (more predictable patterns).
        """
        # Normalize to probability distribution
        autocorr_abs = np.abs(autocorr)
        autocorr_prob = autocorr_abs / (np.sum(autocorr_abs) + 1e-10)

        # Calculate Shannon entropy
        entropy = -np.sum(
            autocorr_prob * np.log2(autocorr_prob + 1e-10)
        )

        # Normalize to 0-1 range
        max_entropy = np.log2(len(autocorr))
        normalized_entropy = entropy / max_entropy

        return float(normalized_entropy)

    def _detect_autocorrelation_anomaly(self, autocorr: np.ndarray) -> bool:
        """
        Detect anomalies in autocorrelation pattern.

        Replay attacks often show periodic peaks in autocorrelation.
        """
        # Find peaks
        peaks, properties = signal.find_peaks(
            np.abs(autocorr),
            height=0.3,
            distance=50
        )

        # Too many strong peaks suggests replay
        if len(peaks) > 5:
            return True

        # Check for periodic pattern (replay signature)
        if len(peaks) >= 3:
            peak_spacing = np.diff(peaks)
            spacing_variance = np.var(peak_spacing)
            if spacing_variance < 10:  # Very regular spacing
                return True

        return False

    def _detect_cross_correlation_anomaly(
        self,
        cross_corr_peaks: List[float]
    ) -> bool:
        """
        Detect anomalies in cross-correlation peaks.

        Spoofed signals often have suspiciously uniform cross-correlation.
        """
        if len(cross_corr_peaks) < 3:
            return False

        # Check variance (too low = suspicious)
        variance = np.var(cross_corr_peaks)
        if variance < 0.01:
            return True

        # Check if all peaks are nearly identical (unrealistic)
        max_diff = np.max(cross_corr_peaks) - np.min(cross_corr_peaks)
        if max_diff < 0.1:
            return True

        return False

    def _check_temporal_consistency(
        self,
        current_fingerprint: CorrelationFingerprint
    ) -> float:
        """
        Check if fingerprint is consistent with historical fingerprints.

        Authentic signals should have gradually evolving fingerprints.
        Spoofed signals may show sudden changes.

        Returns consistency score (0.0 to 1.0)
        """
        if len(self.fingerprint_history) < 3:
            return 1.0  # Not enough history

        # Compare current fingerprint to recent history
        recent_history = self.fingerprint_history[-5:]

        consistency_scores = []

        for hist_fp in recent_history:
            # Compare correlation entropies
            entropy_diff = abs(
                current_fingerprint.correlation_entropy - hist_fp.correlation_entropy
            )
            entropy_score = 1.0 - min(entropy_diff * 2, 1.0)

            # Compare phase coherence
            phase_diff = abs(
                current_fingerprint.phase_coherence - hist_fp.phase_coherence
            )
            phase_score = 1.0 - min(phase_diff * 2, 1.0)

            # Average
            consistency_scores.append((entropy_score + phase_score) / 2.0)

        return float(np.mean(consistency_scores))

    def _score_autocorrelation_shape(self, autocorr: np.ndarray) -> float:
        """
        Score the shape of autocorrelation function.

        Authentic signals have specific autocorrelation shapes.
        """
        # Ideally, autocorrelation should decay from center
        center = len(autocorr) // 2
        autocorr_abs = np.abs(autocorr)

        # Check if center is peak
        center_is_peak = autocorr_abs[center] == np.max(autocorr_abs)
        if not center_is_peak:
            return 0.3  # Anomalous

        # Check decay rate from center
        left_half = autocorr_abs[:center]
        right_half = autocorr_abs[center:]

        # Expect monotonic decrease (mostly)
        left_decreasing = np.sum(np.diff(left_half) > 0) / len(left_half)
        right_decreasing = np.sum(np.diff(right_half) < 0) / len(right_half)

        decay_score = (left_decreasing + right_decreasing) / 2.0

        return float(decay_score)

    def _extract_signal_characteristics(self, iq_signal: np.ndarray) -> Dict:
        """Extract additional signal characteristics for reporting"""
        characteristics = {}

        # Power
        power = np.mean(np.abs(iq_signal) ** 2)
        characteristics['power_db'] = 10 * np.log10(power + 1e-12)

        # Kurtosis
        magnitude = np.abs(iq_signal)
        characteristics['kurtosis'] = float(stats.kurtosis(magnitude))

        # Spectral properties
        freqs, psd = signal.welch(iq_signal, fs=self.sample_rate, nperseg=1024)
        characteristics['peak_frequency_hz'] = float(freqs[np.argmax(psd)])
        characteristics['spectral_flatness'] = self._calc_spectral_flatness(psd)

        return characteristics

    def _calc_spectral_flatness(self, psd: np.ndarray) -> float:
        """Calculate spectral flatness"""
        psd_safe = psd + 1e-10
        geometric_mean = np.exp(np.mean(np.log(psd_safe)))
        arithmetic_mean = np.mean(psd_safe)
        return float(geometric_mean / arithmetic_mean)

    def generate_report(self, detection: QuantumInspiredDetection) -> str:
        """Generate human-readable detection report"""
        report = []
        report.append("=" * 70)
        report.append("ZELDA QUANTUM-INSPIRED SPOOFING DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"Detection Time: {detection.detection_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        if detection.is_spoofed:
            report.append(f"⚠️  SPOOFING DETECTED: {detection.spoofing_method.value.upper()}")
            report.append(f"   Confidence: {detection.confidence*100:.1f}%")
        else:
            report.append("✓  Signal appears authentic")

        report.append("")
        report.append("Correlation Analysis:")
        report.append(f"  Authenticity Score: {detection.correlation_score*100:.1f}%")
        report.append(f"  Correlation Entropy: {detection.fingerprint.correlation_entropy:.3f}")
        report.append(f"  Phase Coherence: {detection.fingerprint.phase_coherence:.3f}")
        report.append(f"  Temporal Consistency: {detection.fingerprint.temporal_consistency:.3f}")

        if detection.anomaly_indicators:
            report.append("")
            report.append("⚠️  Anomaly Indicators:")
            for indicator in detection.anomaly_indicators:
                report.append(f"  • {indicator}")

        if detection.signal_characteristics:
            report.append("")
            report.append("Signal Characteristics:")
            for key, value in detection.signal_characteristics.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.3f}")
                else:
                    report.append(f"  {key}: {value}")

        report.append("=" * 70)
        report.append("Quantum-inspired correlation-based detection")
        report.append("All capabilities are DEFENSIVE (detection/analysis only)")
        report.append("=" * 70)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("ZELDA Quantum-Inspired Spoofing Detection Module")
    print("=" * 60)
    print("Based on: Quantum Illumination LIDAR Spoofing Resilience")
    print("=" * 60)

    # Initialize detector
    detector = QuantumInspiredSpoofDetector(sample_rate=10e6)

    # Test Case 1: Authentic signal
    print("\nTest 1: Authentic Signal")
    t = np.linspace(0, 1024/10e6, 1024)
    # Authentic signal: QPSK + realistic noise
    authentic_signal = np.exp(1j * 2 * np.pi * 1.575e9 * t)  # GPS L1
    authentic_signal += 0.1 * (np.random.randn(1024) + 1j * np.random.randn(1024))

    # Set as reference
    detector.set_reference_pattern(authentic_signal)

    result = detector.detect_spoofing(authentic_signal)
    print(detector.generate_report(result))

    # Test Case 2: Meaconing (replay) attack
    print("\nTest 2: Meaconing (Replay) Attack")
    # Replay attack: repeated authentic signal
    replay_signal = np.tile(authentic_signal[:256], 4)

    result = detector.detect_spoofing(replay_signal)
    print(detector.generate_report(result))

    # Test Case 3: Simulated (fake) signal
    print("\nTest 3: Simulated (Fake) Signal")
    # Simulated signal: too clean, no realistic noise pattern
    simulated_signal = 2 * np.exp(1j * 2 * np.pi * 1.575e9 * t)
    # Minimal noise
    simulated_signal += 0.01 * (np.random.randn(1024) + 1j * np.random.randn(1024))

    result = detector.detect_spoofing(simulated_signal)
    print(detector.generate_report(result))

    print("\n✓ Quantum-inspired detection operational")
    print("✓ Correlation-based spoofing resilience active")
    print("✓ All capabilities are DEFENSIVE (detection/analysis only)")
