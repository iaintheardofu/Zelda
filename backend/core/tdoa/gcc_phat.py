"""
GCC-PHAT (Generalized Cross-Correlation with Phase Transform)

This is the workhorse algorithm for TDOA calculation. GCC-PHAT is robust
to multipath and reverberation, making it ideal for real-world RF environments.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from loguru import logger


def gcc_phat(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sample_rate: float,
    max_tau: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate Time Difference of Arrival using GCC-PHAT algorithm.

    The GCC-PHAT method is preferred for TDOA estimation because:
    1. Robust to reverberation and multipath
    2. Emphasizes phase information over magnitude
    3. Works well in low SNR conditions
    4. Computationally efficient via FFT

    Args:
        sig1: Signal from first receiver (complex I/Q)
        sig2: Signal from second receiver (complex I/Q)
        sample_rate: Sample rate in Hz
        max_tau: Maximum expected time difference in seconds (for validation)

    Returns:
        Tuple of (tdoa_seconds, confidence):
            - tdoa_seconds: Time difference of arrival in seconds (positive means sig2 arrived later)
            - confidence: Confidence metric (0-1, higher is better)

    References:
        Knapp, C., & Carter, G. (1976). The generalized correlation method for
        estimation of time delay. IEEE Trans. ASSP, 24(4), 320-327.
    """

    # Ensure signals are same length
    n = min(len(sig1), len(sig2))
    sig1 = sig1[:n]
    sig2 = sig2[:n]

    # Compute FFT of both signals
    fft1 = np.fft.fft(sig1)
    fft2 = np.fft.fft(sig2)

    # Cross-power spectrum
    cross_spectrum = fft1 * np.conj(fft2)

    # GCC-PHAT weighting: normalize by magnitude
    # This is the "Phase Transform" part - we're keeping only phase information
    magnitude = np.abs(cross_spectrum)
    magnitude[magnitude < 1e-10] = 1e-10  # Avoid division by zero

    gcc_phat_spectrum = cross_spectrum / magnitude

    # Inverse FFT to get cross-correlation in time domain
    correlation = np.fft.ifft(gcc_phat_spectrum)

    # Take real part and shift zero-lag to center
    correlation = np.real(correlation)
    correlation = np.fft.fftshift(correlation)

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_value = np.abs(correlation[peak_idx])

    # Convert peak index to time delay
    center_idx = len(correlation) // 2
    lag_samples = peak_idx - center_idx
    tdoa_seconds = lag_samples / sample_rate

    # Calculate confidence metric
    # Higher peak relative to mean indicates better confidence
    mean_corr = np.mean(np.abs(correlation))
    confidence = min(peak_value / (mean_corr + 1e-10), 1.0)

    # Validate against max_tau if provided
    if max_tau is not None and abs(tdoa_seconds) > max_tau:
        logger.warning(
            f"TDOA {tdoa_seconds*1e6:.2f}us exceeds max_tau {max_tau*1e6:.2f}us, "
            "confidence set to 0"
        )
        confidence = 0.0

    return tdoa_seconds, confidence


def calculate_tdoa(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sample_rate: float,
    method: str = "gcc-phat",
    **kwargs
) -> Tuple[float, float]:
    """
    Calculate TDOA using specified method.

    Args:
        sig1: Signal from first receiver
        sig2: Signal from second receiver
        sample_rate: Sample rate in Hz
        method: TDOA method to use ('gcc-phat', 'xcorr', 'adaptive')
        **kwargs: Additional arguments for specific methods

    Returns:
        Tuple of (tdoa_seconds, confidence)
    """

    if method == "gcc-phat":
        return gcc_phat(sig1, sig2, sample_rate, **kwargs)

    elif method == "xcorr":
        # Standard cross-correlation (simpler but less robust)
        return _simple_xcorr(sig1, sig2, sample_rate, **kwargs)

    elif method == "adaptive":
        # Adaptive method that tries multiple approaches
        return _adaptive_tdoa(sig1, sig2, sample_rate, **kwargs)

    else:
        raise ValueError(f"Unknown TDOA method: {method}")


def _simple_xcorr(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sample_rate: float,
    max_tau: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Simple cross-correlation for TDOA.

    Less robust than GCC-PHAT but faster and simpler.
    Good for high SNR scenarios.
    """

    # Ensure same length
    n = min(len(sig1), len(sig2))
    sig1 = sig1[:n]
    sig2 = sig2[:n]

    # Compute cross-correlation via FFT
    correlation = signal.correlate(sig1, sig2, mode='same', method='fft')

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    peak_value = np.abs(correlation[peak_idx])

    # Convert to time delay
    center_idx = len(correlation) // 2
    lag_samples = peak_idx - center_idx
    tdoa_seconds = lag_samples / sample_rate

    # Confidence
    mean_corr = np.mean(np.abs(correlation))
    confidence = min(peak_value / (mean_corr + 1e-10), 1.0)

    if max_tau is not None and abs(tdoa_seconds) > max_tau:
        confidence = 0.0

    return tdoa_seconds, confidence


def _adaptive_tdoa(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sample_rate: float,
    **kwargs
) -> Tuple[float, float]:
    """
    Adaptive TDOA that tries multiple methods and picks the best.

    Strategy:
    1. Try GCC-PHAT first (most robust)
    2. If confidence is low, try standard cross-correlation
    3. Return result with highest confidence
    """

    # Try GCC-PHAT
    tdoa_phat, conf_phat = gcc_phat(sig1, sig2, sample_rate, **kwargs)

    # If confidence is good, return it
    if conf_phat > 0.7:
        return tdoa_phat, conf_phat

    # Otherwise try standard correlation
    tdoa_xcorr, conf_xcorr = _simple_xcorr(sig1, sig2, sample_rate, **kwargs)

    # Return whichever has higher confidence
    if conf_xcorr > conf_phat:
        logger.debug("Adaptive TDOA: xcorr selected")
        return tdoa_xcorr, conf_xcorr
    else:
        logger.debug("Adaptive TDOA: gcc-phat selected")
        return tdoa_phat, conf_phat


def refine_tdoa_subsample(
    sig1: np.ndarray,
    sig2: np.ndarray,
    sample_rate: float,
    initial_tdoa: float,
    window_samples: int = 64,
) -> Tuple[float, float]:
    """
    Refine TDOA estimate to sub-sample precision.

    Uses parabolic interpolation around the correlation peak to achieve
    sub-sample timing resolution.

    Args:
        sig1: Signal from first receiver
        sig2: Signal from second receiver
        sample_rate: Sample rate in Hz
        initial_tdoa: Initial TDOA estimate in seconds
        window_samples: Number of samples around peak to use

    Returns:
        Tuple of (refined_tdoa_seconds, confidence)
    """

    # Shift sig2 by initial TDOA estimate
    shift_samples = int(initial_tdoa * sample_rate)

    # Extract windows around expected alignment
    n = min(len(sig1), len(sig2))
    center = n // 2

    start1 = max(0, center - window_samples // 2)
    end1 = min(n, center + window_samples // 2)

    start2 = max(0, center + shift_samples - window_samples // 2)
    end2 = min(n, center + shift_samples + window_samples // 2)

    # Ensure same length
    length = min(end1 - start1, end2 - start2)
    window1 = sig1[start1:start1 + length]
    window2 = sig2[start2:start2 + length]

    # Compute high-resolution cross-correlation
    corr = signal.correlate(window1, window2, mode='same')

    # Find peak and neighboring samples
    peak_idx = np.argmax(np.abs(corr))

    if 0 < peak_idx < len(corr) - 1:
        # Parabolic interpolation for sub-sample peak
        y1 = np.abs(corr[peak_idx - 1])
        y2 = np.abs(corr[peak_idx])
        y3 = np.abs(corr[peak_idx + 1])

        # Parabola vertex
        delta = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3 + 1e-10)
        refined_lag = (peak_idx - len(corr) // 2) + delta
    else:
        refined_lag = peak_idx - len(corr) // 2

    # Convert to time
    refined_tdoa = initial_tdoa + (refined_lag / sample_rate)

    # Confidence from peak sharpness
    peak_value = np.abs(corr[peak_idx])
    mean_corr = np.mean(np.abs(corr))
    confidence = min(peak_value / (mean_corr + 1e-10), 1.0)

    return refined_tdoa, confidence


def batch_tdoa(
    signals: list,
    sample_rate: float,
    reference_idx: int = 0,
    method: str = "gcc-phat",
) -> np.ndarray:
    """
    Calculate TDOA for multiple signal pairs against a reference.

    Args:
        signals: List of signal arrays
        sample_rate: Sample rate in Hz
        reference_idx: Index of reference signal
        method: TDOA calculation method

    Returns:
        Array of TDOA values (N-1 values, excluding reference)
        and confidence values
    """

    n_signals = len(signals)
    tdoas = []
    confidences = []

    ref_signal = signals[reference_idx]

    for i, sig in enumerate(signals):
        if i == reference_idx:
            continue

        tdoa, conf = calculate_tdoa(ref_signal, sig, sample_rate, method=method)
        tdoas.append(tdoa)
        confidences.append(conf)

    return np.array(tdoas), np.array(confidences)
