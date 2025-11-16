"""
Advanced Signal Detection Algorithms

Implements state-of-the-art detection methods for weak, hidden, and low-probability-of-intercept (LPI) signals.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CyclostationaryDetector:
    """
    Cyclostationary Feature Detector

    Detects signals with periodic statistical properties (e.g., modulated signals).
    Highly effective for hidden signals and low SNR environments.

    Theory: Most communication signals exhibit cyclostationarity due to:
    - Carrier modulation
    - Symbol timing
    - Coding patterns

    Can detect signals below the noise floor that energy detection would miss.
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        nfft: int = 2048,
        alpha_resolution: int = 256,
        detection_threshold: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.alpha_resolution = alpha_resolution
        self.detection_threshold = detection_threshold

    def spectral_correlation_function(
        self,
        signal_data: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Compute Spectral Correlation Function (SCF)

        SCF reveals hidden periodicities in the signal spectrum.

        Args:
            signal_data: Complex baseband signal
            alpha: Cyclic frequency to test

        Returns:
            Spectral correlation density
        """
        # Time-smoothing method for SCF calculation
        # More efficient than frequency-smoothing for real-time processing

        N = len(signal_data)
        L = self.nfft

        # Down-shift by alpha/2
        t = np.arange(N) / self.sample_rate
        shifted_pos = signal_data * np.exp(1j * np.pi * alpha * t)
        shifted_neg = signal_data * np.exp(-1j * np.pi * alpha * t)

        # Compute FFTs
        X_pos = np.fft.fft(shifted_pos[:L])
        X_neg = np.fft.fft(shifted_neg[:L])

        # Spectral correlation
        S_alpha = X_pos * np.conj(X_neg)

        return S_alpha

    def cyclic_domain_profile(
        self,
        signal_data: np.ndarray,
        frequency: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Cyclic Domain Profile (alpha profile)

        Shows cyclic frequencies present in the signal.
        Peaks indicate strong cyclostationary features.

        Args:
            signal_data: Complex baseband signal
            frequency: Specific frequency to analyze (None = full band)

        Returns:
            (alpha_values, cyclic_power)
        """
        # Alpha range: 0 to sample_rate
        alpha_max = self.sample_rate / 2
        alpha_values = np.linspace(0, alpha_max, self.alpha_resolution)

        cyclic_power = np.zeros(len(alpha_values))

        for i, alpha in enumerate(alpha_values):
            S_alpha = self.spectral_correlation_function(signal_data, alpha)

            if frequency is None:
                # Integrate over all frequencies
                cyclic_power[i] = np.sum(np.abs(S_alpha))
            else:
                # Power at specific frequency
                freq_idx = int(frequency / self.sample_rate * self.nfft)
                cyclic_power[i] = np.abs(S_alpha[freq_idx])

        # Normalize
        cyclic_power /= np.max(cyclic_power) if np.max(cyclic_power) > 0 else 1.0

        return alpha_values, cyclic_power

    def detect(
        self,
        signal_data: np.ndarray,
        return_features: bool = False
    ) -> Dict:
        """
        Detect signals using cyclostationary features

        Args:
            signal_data: Complex baseband IQ data
            return_features: Return detailed feature information

        Returns:
            Detection result with confidence and features
        """
        # Compute cyclic domain profile
        alpha_values, cyclic_power = self.cyclic_domain_profile(signal_data)

        # Find peaks in alpha profile (indicates cyclostationary signals)
        peaks, properties = signal.find_peaks(
            cyclic_power,
            height=self.detection_threshold,
            prominence=0.2
        )

        detected = len(peaks) > 0
        confidence = float(np.max(cyclic_power)) if detected else 0.0

        result = {
            'detected': detected,
            'confidence': confidence,
            'num_cyclic_features': len(peaks),
            'detector_type': 'cyclostationary',
            'snr_estimate': self._estimate_snr(signal_data),
        }

        if return_features and detected:
            result['cyclic_frequencies'] = alpha_values[peaks].tolist()
            result['feature_strengths'] = cyclic_power[peaks].tolist()
            result['symbol_rate_estimate'] = self._estimate_symbol_rate(
                alpha_values[peaks]
            )

        return result

    def _estimate_snr(self, signal_data: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio"""
        # Use noise floor estimation
        psd = np.abs(np.fft.fft(signal_data[:self.nfft]))**2
        noise_floor = np.percentile(psd, 25)  # Lower quartile as noise estimate
        signal_power = np.mean(psd)

        snr_linear = (signal_power - noise_floor) / noise_floor
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

        return float(snr_db)

    def _estimate_symbol_rate(self, cyclic_freqs: np.ndarray) -> Optional[float]:
        """Estimate symbol rate from cyclic frequencies"""
        if len(cyclic_freqs) < 2:
            return None

        # Symbol rate often appears as fundamental cyclic frequency
        # or as difference between carrier-related peaks
        return float(np.min(cyclic_freqs[cyclic_freqs > 0]))


class EnergyDetector:
    """
    Advanced Energy Detector with CFAR (Constant False Alarm Rate)

    Detects signals by measuring energy in frequency bins.
    Enhanced with adaptive thresholding for varying noise environments.
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        nfft: int = 2048,
        pfa: float = 1e-3,  # Probability of false alarm
        integration_time: float = 0.001,  # 1ms
    ):
        self.sample_rate = sample_rate
        self.nfft = nfft
        self.pfa = pfa
        self.integration_time = integration_time
        self.noise_floor = None

    def estimate_noise_floor(
        self,
        signal_data: np.ndarray,
        percentile: float = 25
    ) -> float:
        """
        Estimate noise floor using robust statistics

        Uses lower percentile of power spectral density to avoid
        bias from strong signals.
        """
        psd = np.abs(np.fft.fft(signal_data[:self.nfft]))**2
        noise_estimate = np.percentile(psd, percentile)

        self.noise_floor = noise_estimate
        return float(noise_estimate)

    def cfar_threshold(
        self,
        reference_cells: np.ndarray,
        guard_cells: int = 2
    ) -> float:
        """
        Compute CFAR (Constant False Alarm Rate) threshold

        Adaptive threshold based on local noise statistics.
        Maintains constant false alarm rate across varying noise levels.

        Args:
            reference_cells: Power values from neighboring cells
            guard_cells: Number of cells to exclude around test cell

        Returns:
            Detection threshold
        """
        # Cell-Averaging CFAR (CA-CFAR)
        # Exclude guard cells
        if len(reference_cells) > 2 * guard_cells:
            ref_power = np.concatenate([
                reference_cells[:len(reference_cells)//2 - guard_cells],
                reference_cells[len(reference_cells)//2 + guard_cells:]
            ])
        else:
            ref_power = reference_cells

        # Average noise power
        noise_avg = np.mean(ref_power)

        # Threshold based on desired Pfa
        # For exponential distribution (common in radar/comms)
        threshold_factor = -np.log(self.pfa)
        threshold = noise_avg * threshold_factor

        return threshold

    def detect(
        self,
        signal_data: np.ndarray,
        return_spectrum: bool = False
    ) -> Dict:
        """
        Detect signals using energy detection with CFAR

        Args:
            signal_data: Complex baseband signal
            return_spectrum: Return power spectrum

        Returns:
            Detection results with frequencies and confidence
        """
        # Compute power spectral density
        num_integrations = int(self.integration_time * self.sample_rate / self.nfft)
        num_integrations = max(1, num_integrations)

        psd_integrated = np.zeros(self.nfft)
        for i in range(num_integrations):
            start_idx = i * self.nfft
            end_idx = start_idx + self.nfft
            if end_idx > len(signal_data):
                break

            segment = signal_data[start_idx:end_idx]
            psd = np.abs(np.fft.fft(segment))**2
            psd_integrated += psd

        psd_integrated /= num_integrations

        # Estimate noise floor if not set
        if self.noise_floor is None:
            self.estimate_noise_floor(signal_data)

        # Apply CFAR detection
        detections = []
        window_size = 32  # Reference window size
        guard_size = 4

        for i in range(window_size, len(psd_integrated) - window_size):
            # Get reference cells
            ref_cells = np.concatenate([
                psd_integrated[i - window_size:i - guard_size],
                psd_integrated[i + guard_size:i + window_size]
            ])

            threshold = self.cfar_threshold(ref_cells, guard_cells=guard_size)

            # Test cell
            if psd_integrated[i] > threshold:
                freq = (i / self.nfft - 0.5) * self.sample_rate
                power = psd_integrated[i]
                snr = 10 * np.log10(power / self.noise_floor)

                detections.append({
                    'frequency': freq,
                    'power': power,
                    'snr': snr,
                    'bin_index': i
                })

        result = {
            'detected': len(detections) > 0,
            'num_detections': len(detections),
            'detector_type': 'energy_cfar',
            'noise_floor': float(self.noise_floor),
            'detections': detections[:100]  # Limit to top 100
        }

        if return_spectrum:
            frequencies = np.fft.fftfreq(self.nfft, 1/self.sample_rate)
            result['frequencies'] = frequencies.tolist()
            result['psd'] = psd_integrated.tolist()

        return result


class BlindDetector:
    """
    Blind Signal Detector using Eigenvalue-based Methods

    Detects signals without prior knowledge of signal characteristics.
    Uses random matrix theory and eigenvalue decomposition.

    Highly effective for:
    - Unknown signal types
    - Multiple simultaneous signals
    - Low SNR environments
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        smoothing_factor: int = 10,
        detection_threshold: float = 2.0
    ):
        self.sample_rate = sample_rate
        self.smoothing_factor = smoothing_factor
        self.detection_threshold = detection_threshold

    def covariance_matrix(
        self,
        signal_data: np.ndarray,
        M: int = 20
    ) -> np.ndarray:
        """
        Compute sample covariance matrix

        Args:
            signal_data: Complex signal samples
            M: Dimension of covariance matrix (smoothing factor)

        Returns:
            M x M covariance matrix
        """
        N = len(signal_data) - M + 1

        # Form data matrix
        X = np.zeros((M, N), dtype=complex)
        for i in range(M):
            X[i, :] = signal_data[i:i+N]

        # Sample covariance
        R = (X @ X.conj().T) / N

        return R

    def eigenvalue_spectrum(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalue spectrum of covariance matrix

        Returns:
            (eigenvalues, eigenvectors) sorted descending
        """
        R = self.covariance_matrix(signal_data, M=self.smoothing_factor)

        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def detect(
        self,
        signal_data: np.ndarray,
        return_features: bool = False
    ) -> Dict:
        """
        Blind signal detection using eigenvalue analysis

        Uses Maximum Eigenvalue (ME) and Energy with Minimum Eigenvalue (EME) tests.

        Args:
            signal_data: Complex baseband signal
            return_features: Return eigenvalue spectrum

        Returns:
            Detection result with confidence
        """
        eigenvalues, eigenvectors = self.eigenvalue_spectrum(signal_data)

        # Maximum Eigenvalue test
        lambda_max = eigenvalues[0]
        lambda_min = eigenvalues[-1]

        # ME test statistic
        T_ME = lambda_max / lambda_min

        # EME test statistic
        trace = np.sum(eigenvalues)
        T_EME = trace / (self.smoothing_factor * lambda_min)

        # Detection decision
        detected_ME = T_ME > self.detection_threshold
        detected_EME = T_EME > self.detection_threshold

        detected = detected_ME or detected_EME

        # Estimate number of signals (non-noise eigenvalues)
        # Use information theoretic criteria (AIC/MDL)
        num_signals = self._estimate_num_signals(eigenvalues)

        result = {
            'detected': detected,
            'detector_type': 'blind_eigenvalue',
            'test_statistic_ME': float(T_ME),
            'test_statistic_EME': float(T_EME),
            'num_signals_estimate': num_signals,
            'confidence': float(min(T_ME, T_EME) / self.detection_threshold)
        }

        if return_features:
            result['eigenvalues'] = eigenvalues.real.tolist()
            result['condition_number'] = float(lambda_max / lambda_min)

        return result

    def _estimate_num_signals(self, eigenvalues: np.ndarray) -> int:
        """
        Estimate number of signals using MDL (Minimum Description Length)

        Based on information theoretic model selection.
        """
        M = len(eigenvalues)

        mdl_values = []
        for k in range(M):
            # Arithmetic mean of noise eigenvalues
            if k < M:
                noise_eigs = eigenvalues[k:]
                arith_mean = np.mean(noise_eigs)
                geom_mean = np.exp(np.mean(np.log(noise_eigs)))

                # MDL criterion
                N = 1000  # Sample size estimate
                mdl = -N * (M - k) * np.log(geom_mean / arith_mean) + \
                      0.5 * k * (2*M - k) * np.log(N)
                mdl_values.append(mdl)
            else:
                mdl_values.append(np.inf)

        # Number of signals is k that minimizes MDL
        num_signals = int(np.argmin(mdl_values))

        return num_signals


class MultiAlgorithmFusion:
    """
    Fuses results from multiple detection algorithms

    Combines cyclostationary, energy, and blind detection for:
    - Improved detection probability
    - Reduced false alarms
    - Robustness across signal types
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        fusion_strategy: str = 'majority_vote'  # or 'weighted', 'all_agree'
    ):
        self.sample_rate = sample_rate
        self.fusion_strategy = fusion_strategy

        # Initialize detectors
        self.cyclo_detector = CyclostationaryDetector(sample_rate=sample_rate)
        self.energy_detector = EnergyDetector(sample_rate=sample_rate)
        self.blind_detector = BlindDetector(sample_rate=sample_rate)

    def detect(
        self,
        signal_data: np.ndarray,
        return_individual: bool = False
    ) -> Dict:
        """
        Run all detectors and fuse results

        Args:
            signal_data: Complex baseband signal
            return_individual: Return individual detector results

        Returns:
            Fused detection result
        """
        # Run all detectors
        cyclo_result = self.cyclo_detector.detect(signal_data, return_features=True)
        energy_result = self.energy_detector.detect(signal_data, return_spectrum=False)
        blind_result = self.blind_detector.detect(signal_data, return_features=True)

        # Fusion logic
        detections = [
            cyclo_result['detected'],
            energy_result['detected'],
            blind_result['detected']
        ]

        confidences = [
            cyclo_result.get('confidence', 0),
            1.0 if energy_result['detected'] else 0.0,
            blind_result.get('confidence', 0)
        ]

        if self.fusion_strategy == 'majority_vote':
            fused_detection = sum(detections) >= 2
        elif self.fusion_strategy == 'weighted':
            # Weighted by confidence
            weights = [0.4, 0.3, 0.3]  # Cyclo gets higher weight
            fused_confidence = sum(c * w for c, w in zip(confidences, weights))
            fused_detection = fused_confidence > 0.5
        elif self.fusion_strategy == 'all_agree':
            fused_detection = all(detections)
        else:
            # OR fusion (any detector)
            fused_detection = any(detections)

        result = {
            'detected': fused_detection,
            'fusion_strategy': self.fusion_strategy,
            'num_agreeing': sum(detections),
            'overall_confidence': float(np.mean(confidences)),
            'detector_votes': {
                'cyclostationary': cyclo_result['detected'],
                'energy': energy_result['detected'],
                'blind': blind_result['detected']
            }
        }

        if return_individual:
            result['individual_results'] = {
                'cyclostationary': cyclo_result,
                'energy': energy_result,
                'blind': blind_result
            }

        logger.info(f"Multi-algorithm fusion: {sum(detections)}/3 detectors agree, "
                   f"final decision: {fused_detection}")

        return result
