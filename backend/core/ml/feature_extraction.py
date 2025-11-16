"""
Feature extraction from I/Q signals for ML
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, Tuple, Optional


def extract_iq_features(iq_data: np.ndarray, sample_rate: float) -> Dict[str, float]:
    """
    Extract statistical and spectral features from I/Q data.

    These features can be used for:
    - Signal classification
    - Quality assessment
    - Anomaly detection

    Args:
        iq_data: Complex I/Q samples
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary of features
    """

    features = {}

    # Amplitude features
    amplitude = np.abs(iq_data)
    features['mean_amplitude'] = np.mean(amplitude)
    features['std_amplitude'] = np.std(amplitude)
    features['max_amplitude'] = np.max(amplitude)
    features['min_amplitude'] = np.min(amplitude)

    # Phase features
    phase = np.angle(iq_data)
    phase_diff = np.diff(phase)
    # Unwrap phase for better statistics
    phase_unwrapped = np.unwrap(phase)
    features['mean_phase'] = np.mean(phase_unwrapped)
    features['std_phase'] = np.std(phase_diff)

    # Frequency features
    instantaneous_freq = np.diff(phase_unwrapped) * sample_rate / (2 * np.pi)
    features['mean_freq_offset'] = np.mean(instantaneous_freq)
    features['std_freq_offset'] = np.std(instantaneous_freq)

    # Power features
    power = np.abs(iq_data) ** 2
    features['mean_power'] = np.mean(power)
    features['peak_power'] = np.max(power)
    features['papr'] = np.max(power) / (np.mean(power) + 1e-10)  # Peak-to-average ratio

    # I/Q balance
    i_data = np.real(iq_data)
    q_data = np.imag(iq_data)
    features['i_mean'] = np.mean(i_data)
    features['q_mean'] = np.mean(q_data)
    features['iq_ratio'] = np.std(i_data) / (np.std(q_data) + 1e-10)

    # Spectral features
    fft = np.fft.fft(iq_data)
    psd = np.abs(fft) ** 2
    psd_db = 10 * np.log10(psd + 1e-10)

    features['spectral_mean'] = np.mean(psd_db)
    features['spectral_std'] = np.std(psd_db)
    features['spectral_max'] = np.max(psd_db)

    # Bandwidth estimation (3dB bandwidth)
    peak_power_db = np.max(psd_db)
    threshold_db = peak_power_db - 3
    above_threshold = psd_db > threshold_db
    bandwidth_bins = np.sum(above_threshold)
    features['bandwidth_3db'] = (bandwidth_bins / len(psd)) * sample_rate

    # Kurtosis (measures peakiness of distribution)
    features['kurtosis'] = _kurtosis(amplitude)

    # Zero-crossing rate
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(i_data)) != 0) / len(i_data)

    return features


def extract_spectrogram(
    iq_data: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram of I/Q signal.

    Args:
        iq_data: Complex I/Q samples
        sample_rate: Sample rate in Hz
        nperseg: Length of each segment
        noverlap: Number of overlapping samples

    Returns:
        Tuple of (frequencies, times, spectrogram_magnitude_dB)
    """

    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = sp_signal.spectrogram(
        iq_data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )

    # Convert to dB
    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)

    # Shift zero frequency to center
    f = np.fft.fftshift(f)
    Sxx_db = np.fft.fftshift(Sxx_db, axes=0)

    return f, t, Sxx_db


def compute_constellation(iq_data: np.ndarray, bins: int = 100) -> np.ndarray:
    """
    Compute constellation diagram (2D histogram of I/Q points).

    Args:
        iq_data: Complex I/Q samples
        bins: Number of bins for 2D histogram

    Returns:
        2D histogram of I/Q constellation
    """

    i_data = np.real(iq_data)
    q_data = np.imag(iq_data)

    # Normalize to [-1, 1]
    max_val = max(np.max(np.abs(i_data)), np.max(np.abs(q_data)))
    if max_val > 0:
        i_data = i_data / max_val
        q_data = q_data / max_val

    # Create 2D histogram
    hist, _, _ = np.histogram2d(
        i_data, q_data,
        bins=bins,
        range=[[-1, 1], [-1, 1]]
    )

    return hist


def detect_signal_presence(
    iq_data: np.ndarray,
    noise_floor: Optional[float] = None,
    threshold_db: float = 10.0
) -> bool:
    """
    Detect if a signal is present above noise floor.

    Args:
        iq_data: Complex I/Q samples
        noise_floor: Noise floor power (if known)
        threshold_db: Detection threshold above noise in dB

    Returns:
        True if signal detected
    """

    power = np.abs(iq_data) ** 2
    mean_power = np.mean(power)

    if noise_floor is None:
        # Estimate noise floor from lowest 10% of samples
        sorted_power = np.sort(power)
        noise_floor = np.mean(sorted_power[:len(sorted_power) // 10])

    # Calculate SNR
    snr_linear = mean_power / (noise_floor + 1e-10)
    snr_db = 10 * np.log10(snr_linear)

    return snr_db > threshold_db


def estimate_snr(iq_data: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio.

    Uses M2M4 method (second and fourth moments).

    Args:
        iq_data: Complex I/Q samples

    Returns:
        Estimated SNR in dB
    """

    # M2M4 estimator
    amplitude = np.abs(iq_data)

    m2 = np.mean(amplitude ** 2)
    m4 = np.mean(amplitude ** 4)

    # Avoid division by zero
    if m2 == 0 or m4 == 0:
        return -np.inf

    # SNR estimation
    snr_linear = np.sqrt(2 * m2**2 - m4) / (m2 - np.sqrt(2 * m2**2 - m4))

    # Clip to reasonable range
    snr_linear = np.clip(snr_linear, 1e-10, 1e10)

    snr_db = 10 * np.log10(snr_linear)

    return snr_db


def _kurtosis(x: np.ndarray) -> float:
    """Calculate kurtosis of a distribution"""
    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return 0.0

    normalized = (x - mean) / std
    kurt = np.mean(normalized ** 4) - 3

    return kurt
