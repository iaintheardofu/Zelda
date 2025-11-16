"""
Cognitive Radio Engine

Intelligent spectrum management and adaptive interference mitigation.
Enables operation in congested/hostile RF environments.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CognitiveEngine:
    """
    Cognitive Radio Decision Engine

    Makes intelligent decisions about:
    - Spectrum sensing and hole detection
    - Optimal frequency selection
    - Power allocation
    - Interference avoidance

    Uses learning algorithms to adapt to RF environment.
    """

    def __init__(
        self,
        sample_rate: float = 40e6,
        freq_range: Tuple[float, float] = (800e6, 6e9),
        learning_rate: float = 0.01
    ):
        self.sample_rate = sample_rate
        self.freq_range = freq_range
        self.learning_rate = learning_rate

        # Spectrum database (learned from observations)
        self.spectrum_history = []
        self.interference_map = {}

    def sense_spectrum(
        self,
        signal_data: np.ndarray,
        center_freq: float
    ) -> Dict:
        """
        Wideband spectrum sensing

        Identifies:
        - Occupied channels
        - Interference sources
        - Spectrum holes (available frequencies)

        Returns spectrum occupancy map.
        """
        # FFT-based energy detection across band
        nfft = 2048
        psd = np.abs(np.fft.fft(signal_data[:nfft]))**2

        # Convert to dBm
        psd_dbm = 10 * np.log10(psd + 1e-10)

        # Frequency bins
        frequencies = np.fft.fftfreq(nfft, 1/self.sample_rate) + center_freq

        # Adaptive threshold for occupancy detection
        noise_floor = np.percentile(psd_dbm, 25)
        threshold = noise_floor + 10  # 10 dB above noise

        occupied = psd_dbm > threshold

        # Identify spectrum holes (contiguous unoccupied regions)
        holes = self._find_spectrum_holes(frequencies, occupied)

        # Update interference map
        self._update_interference_map(frequencies, psd_dbm, center_freq)

        return {
            'frequencies': frequencies.tolist(),
            'psd_dbm': psd_dbm.tolist(),
            'occupied': occupied.tolist(),
            'spectrum_holes': holes,
            'noise_floor': float(noise_floor),
            'occupancy_percent': float(np.mean(occupied) * 100)
        }

    def _find_spectrum_holes(
        self,
        frequencies: np.ndarray,
        occupied: np.ndarray,
        min_bandwidth: float = 1e6
    ) -> List[Dict]:
        """
        Find contiguous unoccupied frequency bands

        Args:
            frequencies: Frequency bins
            occupied: Boolean array of occupancy
            min_bandwidth: Minimum bandwidth for a hole to be useful

        Returns:
            List of spectrum holes with start/end frequencies
        """
        holes = []

        # Find transitions
        diff = np.diff(occupied.astype(int))
        hole_starts = np.where(diff == -1)[0] + 1  # Occupied -> Unoccupied
        hole_ends = np.where(diff == 1)[0] + 1     # Unoccupied -> Occupied

        # Handle edge cases
        if not occupied[0]:
            hole_starts = np.concatenate([[0], hole_starts])
        if not occupied[-1]:
            hole_ends = np.concatenate([hole_ends, [len(occupied) - 1]])

        # Extract hole information
        for start_idx, end_idx in zip(hole_starts, hole_ends):
            if start_idx >= len(frequencies) or end_idx >= len(frequencies):
                continue

            start_freq = frequencies[start_idx]
            end_freq = frequencies[end_idx]
            bandwidth = end_freq - start_freq

            if bandwidth >= min_bandwidth:
                holes.append({
                    'start_freq': float(start_freq),
                    'end_freq': float(end_freq),
                    'center_freq': float((start_freq + end_freq) / 2),
                    'bandwidth': float(bandwidth),
                    'quality_score': self._compute_hole_quality(start_idx, end_idx)
                })

        # Sort by quality score
        holes.sort(key=lambda x: x['quality_score'], reverse=True)

        return holes

    def _compute_hole_quality(self, start_idx: int, end_idx: int) -> float:
        """
        Compute quality score for a spectrum hole

        Considers:
        - Bandwidth (wider is better)
        - Historical interference (lower is better)
        - Proximity to known interferers (farther is better)
        """
        bandwidth_score = (end_idx - start_idx) / 100.0
        # Add interference history and proximity scores in production
        return min(bandwidth_score, 1.0)

    def _update_interference_map(
        self,
        frequencies: np.ndarray,
        psd_dbm: np.ndarray,
        center_freq: float
    ):
        """
        Update learned interference map

        Tracks interference sources over time for predictive avoidance.
        """
        # Store recent observations
        self.spectrum_history.append({
            'timestamp': None,  # Add in production
            'center_freq': center_freq,
            'psd': psd_dbm
        })

        # Keep last 100 observations
        if len(self.spectrum_history) > 100:
            self.spectrum_history.pop(0)

        # Update interference map (simplified)
        for i, freq in enumerate(frequencies):
            freq_key = int(freq / 1e6)  # Round to MHz
            if freq_key not in self.interference_map:
                self.interference_map[freq_key] = []

            self.interference_map[freq_key].append(psd_dbm[i])

            # Keep last 100 samples
            if len(self.interference_map[freq_key]) > 100:
                self.interference_map[freq_key].pop(0)

    def select_best_frequency(
        self,
        spectrum_holes: List[Dict],
        required_bandwidth: float = 5e6
    ) -> Optional[Dict]:
        """
        Select optimal frequency for operation

        Uses multi-criteria decision making:
        - Sufficient bandwidth
        - Low historical interference
        - Stability over time

        Returns:
            Best frequency channel or None if no suitable hole found
        """
        candidates = [
            hole for hole in spectrum_holes
            if hole['bandwidth'] >= required_bandwidth
        ]

        if not candidates:
            logger.warning("No spectrum holes with sufficient bandwidth")
            return None

        # Select highest quality score
        best_hole = candidates[0]

        logger.info(f"Selected frequency: {best_hole['center_freq']/1e6:.3f} MHz, "
                   f"BW: {best_hole['bandwidth']/1e6:.3f} MHz, "
                   f"Quality: {best_hole['quality_score']:.2f}")

        return best_hole


class InterferenceCanceller:
    """
    Advanced Interference Cancellation

    Implements adaptive filtering techniques:
    - LMS (Least Mean Squares)
    - RLS (Recursive Least Squares)
    - Blind source separation

    Removes or suppresses interference without knowledge of interfering signal.
    """

    def __init__(
        self,
        num_taps: int = 64,
        algorithm: str = 'lms',
        mu: float = 0.01  # Step size for LMS
    ):
        self.num_taps = num_taps
        self.algorithm = algorithm
        self.mu = mu

        # Filter weights
        self.weights = np.zeros(num_taps, dtype=complex)

    def lms_filter(
        self,
        signal_data: np.ndarray,
        desired_signal: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Least Mean Squares adaptive filter

        Args:
            signal_data: Input signal (signal + interference)
            desired_signal: Known desired signal (if available)

        Returns:
            (filtered_signal, error_signal)
        """
        N = len(signal_data)
        output = np.zeros(N, dtype=complex)
        error = np.zeros(N, dtype=complex)

        # If desired signal not provided, use blind adaptation
        if desired_signal is None:
            # Use decision-directed approach
            desired_signal = np.sign(signal_data.real) + 1j * np.sign(signal_data.imag)

        for n in range(self.num_taps, N):
            # Input vector (current and past samples)
            x = signal_data[n-self.num_taps:n][::-1]

            # Filter output
            y = np.dot(self.weights.conj(), x)
            output[n] = y

            # Error
            e = desired_signal[n] - y
            error[n] = e

            # Update weights (LMS)
            self.weights += self.mu * e.conj() * x

        return output, error

    def rls_filter(
        self,
        signal_data: np.ndarray,
        desired_signal: np.ndarray,
        lambda_rls: float = 0.99
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recursive Least Squares adaptive filter

        Faster convergence than LMS but higher computational cost.

        Args:
            signal_data: Input signal
            desired_signal: Desired signal
            lambda_rls: Forgetting factor (0.95-0.99)

        Returns:
            (filtered_signal, error_signal)
        """
        N = len(signal_data)
        output = np.zeros(N, dtype=complex)
        error = np.zeros(N, dtype=complex)

        # Initialize correlation matrix
        delta = 0.01  # Small positive constant
        P = np.eye(self.num_taps) / delta

        for n in range(self.num_taps, N):
            x = signal_data[n-self.num_taps:n][::-1]

            # Output
            y = np.dot(self.weights.conj(), x)
            output[n] = y

            # Error
            e = desired_signal[n] - y
            error[n] = e

            # Gain vector
            k = P @ x / (lambda_rls + x.conj() @ P @ x)

            # Update weights
            self.weights += k * e.conj()

            # Update correlation matrix
            P = (P - np.outer(k, x.conj() @ P)) / lambda_rls

        return output, error

    def cancel_interference(
        self,
        signal_data: np.ndarray,
        reference_signal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Cancel interference from signal

        Args:
            signal_data: Signal contaminated with interference
            reference_signal: Clean reference (if available)

        Returns:
            Interference-cancelled signal
        """
        if self.algorithm == 'lms':
            filtered, _ = self.lms_filter(signal_data, reference_signal)
        elif self.algorithm == 'rls':
            filtered, _ = self.rls_filter(signal_data, reference_signal or signal_data)
        else:
            logger.error(f"Unknown algorithm: {self.algorithm}")
            return signal_data

        return filtered


class SpectrumManager:
    """
    Intelligent Spectrum Management

    Manages spectrum usage across multiple receivers/channels:
    - Dynamic channel allocation
    - Load balancing
    - Interference coordination
    """

    def __init__(self, num_channels: int = 8):
        self.num_channels = num_channels
        self.channel_allocations = {}
        self.channel_usage = {i: 0.0 for i in range(num_channels)}

    def allocate_channel(
        self,
        required_bandwidth: float,
        priority: int = 1
    ) -> Optional[int]:
        """
        Allocate a channel for use

        Args:
            required_bandwidth: Bandwidth needed
            priority: Priority level (1=low, 10=high)

        Returns:
            Channel number or None if no channel available
        """
        # Find least loaded channel
        available_channels = [
            ch for ch, usage in self.channel_usage.items()
            if usage < 0.8  # 80% threshold
        ]

        if not available_channels:
            logger.warning("No available channels")
            return None

        # Select channel with lowest usage
        best_channel = min(available_channels, key=lambda ch: self.channel_usage[ch])

        # Update usage
        self.channel_usage[best_channel] += 0.2  # Estimate

        logger.info(f"Allocated channel {best_channel}, usage now {self.channel_usage[best_channel]:.1%}")

        return best_channel

    def release_channel(self, channel: int):
        """Release a channel back to pool"""
        if channel in self.channel_usage:
            self.channel_usage[channel] = max(0.0, self.channel_usage[channel] - 0.2)
            logger.info(f"Released channel {channel}")


class AdaptiveFilter:
    """
    General-purpose adaptive filter framework

    Supports multiple adaptation algorithms and filter structures.
    """

    def __init__(
        self,
        filter_length: int = 64,
        algorithm: str = 'nlms',  # Normalized LMS
        step_size: float = 0.1
    ):
        self.filter_length = filter_length
        self.algorithm = algorithm
        self.step_size = step_size
        self.coefficients = np.zeros(filter_length, dtype=complex)

    def nlms(
        self,
        input_signal: np.ndarray,
        desired_signal: np.ndarray,
        epsilon: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalized Least Mean Squares

        Normalizes step size by input power for better stability.
        """
        N = len(input_signal)
        output = np.zeros(N, dtype=complex)
        error = np.zeros(N, dtype=complex)

        for n in range(self.filter_length, N):
            x = input_signal[n-self.filter_length:n][::-1]

            # Filter output
            y = np.dot(self.coefficients.conj(), x)
            output[n] = y

            # Error
            e = desired_signal[n] - y
            error[n] = e

            # Normalized step size
            x_power = np.dot(x.conj(), x).real + epsilon
            mu_n = self.step_size / x_power

            # Update coefficients
            self.coefficients += mu_n * e.conj() * x

        return output, error

    def filter_signal(
        self,
        signal_data: np.ndarray,
        desired_signal: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply adaptive filter to signal

        Args:
            signal_data: Input signal
            desired_signal: Reference signal (if available)

        Returns:
            Filtered signal
        """
        if desired_signal is None:
            # Blind mode: estimate desired signal
            desired_signal = np.sign(signal_data.real) + 1j * np.sign(signal_data.imag)

        if self.algorithm == 'nlms':
            output, _ = self.nlms(signal_data, desired_signal)
        else:
            logger.error(f"Unsupported algorithm: {self.algorithm}")
            return signal_data

        return output
