"""
Interference Detection using ML
"""

import numpy as np
from typing import List, Tuple
from loguru import logger
from .feature_extraction import extract_iq_features, estimate_snr


class InterferenceDetector:
    """
    Detect and classify interference in RF signals.

    Types of interference:
    - Narrowband interference
    - Wideband/broadband jamming
    - Impulsive noise
    - Multipath
    """

    def __init__(
        self,
        snr_threshold: float = 10.0,
        narrowband_threshold: float = 0.3,
        impulsive_threshold: float = 5.0,
    ):
        """
        Initialize interference detector.

        Args:
            snr_threshold: Minimum SNR (dB) for clean signal
            narrowband_threshold: Fraction of bandwidth for narrowband detection
            impulsive_threshold: Threshold for impulsive noise detection (sigma)
        """

        self.snr_threshold = snr_threshold
        self.narrowband_threshold = narrowband_threshold
        self.impulsive_threshold = impulsive_threshold

    def detect(self, iq_data: np.ndarray, sample_rate: float) -> dict:
        """
        Detect interference in I/Q signal.

        Args:
            iq_data: Complex I/Q samples
            sample_rate: Sample rate in Hz

        Returns:
            Dictionary with interference analysis
        """

        result = {
            'has_interference': False,
            'interference_types': [],
            'quality_score': 1.0,  # 0-1, higher is better
            'snr_db': 0.0,
        }

        # Estimate SNR
        snr = estimate_snr(iq_data)
        result['snr_db'] = snr

        if snr < self.snr_threshold:
            result['has_interference'] = True
            result['interference_types'].append('low_snr')
            result['quality_score'] *= (snr / self.snr_threshold)

        # Extract features
        features = extract_iq_features(iq_data, sample_rate)

        # Check for narrowband interference
        if self._is_narrowband_interference(iq_data, features):
            result['has_interference'] = True
            result['interference_types'].append('narrowband')
            result['quality_score'] *= 0.7

        # Check for impulsive noise
        if self._is_impulsive_noise(iq_data, features):
            result['has_interference'] = True
            result['interference_types'].append('impulsive')
            result['quality_score'] *= 0.5

        # Check for DC offset
        if abs(features['i_mean']) > 0.1 or abs(features['q_mean']) > 0.1:
            result['has_interference'] = True
            result['interference_types'].append('dc_offset')
            result['quality_score'] *= 0.9

        # Check I/Q imbalance
        if features['iq_ratio'] < 0.8 or features['iq_ratio'] > 1.2:
            result['has_interference'] = True
            result['interference_types'].append('iq_imbalance')
            result['quality_score'] *= 0.8

        if result['has_interference']:
            logger.debug(
                f"Interference detected: {result['interference_types']}, "
                f"quality={result['quality_score']:.2f}"
            )

        return result

    def _is_narrowband_interference(
        self,
        iq_data: np.ndarray,
        features: dict
    ) -> bool:
        """Detect narrowband interference (e.g., CW tone)"""

        # Check if bandwidth is very narrow compared to sample rate
        bandwidth_ratio = features.get('bandwidth_3db', 0) / features.get('sample_rate', 1)

        if bandwidth_ratio < self.narrowband_threshold:
            return True

        # Check for high spectral peak
        if features.get('spectral_max', 0) - features.get('spectral_mean', 0) > 20:
            return True

        return False

    def _is_impulsive_noise(
        self,
        iq_data: np.ndarray,
        features: dict
    ) -> bool:
        """Detect impulsive noise"""

        amplitude = np.abs(iq_data)
        mean = np.mean(amplitude)
        std = np.std(amplitude)

        # Check for outliers
        threshold = mean + self.impulsive_threshold * std
        num_impulses = np.sum(amplitude > threshold)

        impulse_rate = num_impulses / len(amplitude)

        # If more than 1% of samples are impulses, flag it
        if impulse_rate > 0.01:
            return True

        return False

    def clean_signal(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        method: str = 'notch'
    ) -> np.ndarray:
        """
        Attempt to clean interference from signal.

        Args:
            iq_data: Complex I/Q samples
            sample_rate: Sample rate in Hz
            method: Cleaning method ('notch', 'median', 'adaptive')

        Returns:
            Cleaned I/Q data
        """

        detection = self.detect(iq_data, sample_rate)

        if not detection['has_interference']:
            return iq_data

        cleaned = iq_data.copy()

        # Remove DC offset
        if 'dc_offset' in detection['interference_types']:
            cleaned = cleaned - np.mean(cleaned)

        # Remove impulsive noise (median filter)
        if 'impulsive' in detection['interference_types']:
            cleaned = self._remove_impulses(cleaned)

        # Narrowband notch filter
        if 'narrowband' in detection['interference_types'] and method == 'notch':
            cleaned = self._notch_filter(cleaned, sample_rate)

        logger.debug("Signal cleaned")

        return cleaned

    def _remove_impulses(self, iq_data: np.ndarray) -> np.ndarray:
        """Remove impulsive noise using median filter"""

        from scipy.signal import medfilt

        # Separate I and Q
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)

        # Apply median filter
        i_clean = medfilt(i_data, kernel_size=5)
        q_clean = medfilt(q_data, kernel_size=5)

        return i_clean + 1j * q_clean

    def _notch_filter(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply notch filter to remove narrowband interference"""

        from scipy.signal import iirnotch, filtfilt

        # Find dominant frequency
        fft = np.fft.fft(iq_data)
        freqs = np.fft.fftfreq(len(iq_data), 1/sample_rate)

        # Find peak
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        if peak_freq == 0:
            return iq_data

        # Design notch filter
        notch_freq = abs(peak_freq)
        quality_factor = 30  # Higher Q = narrower notch

        b, a = iirnotch(notch_freq, quality_factor, sample_rate)

        # Apply filter
        cleaned = filtfilt(b, a, iq_data)

        logger.debug(f"Notch filter applied at {notch_freq/1e3:.1f} kHz")

        return cleaned
