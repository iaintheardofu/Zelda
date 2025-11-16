"""
Live Signal Detection System
Real-time signal detection with visualization
"""

import torch
import numpy as np
from pathlib import Path
import time
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse

from backend.core.ml.advanced_detector import create_model


class LiveDetector:
    """Live signal detection with real-time visualization"""

    def __init__(
        self,
        model_path: str,
        model_type: str = "ultra",
        device: str = "cuda",
        window_size: int = 4096,
        history_size: int = 1000,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.window_size = window_size

        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model = create_model(model_type=model_type, input_length=window_size).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

        # Detection history
        self.detection_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)

        # Statistics
        self.total_processed = 0
        self.total_detections = 0
        self.start_time = time.time()

    @torch.no_grad()
    def detect(self, iq_data: np.ndarray) -> dict:
        """
        Perform detection on I/Q data

        Args:
            iq_data: Complex I/Q samples (numpy array)

        Returns:
            Detection result dictionary
        """

        # Ensure correct length
        if len(iq_data) > self.window_size:
            # Take center portion
            start = (len(iq_data) - self.window_size) // 2
            iq_data = iq_data[start:start + self.window_size]
        elif len(iq_data) < self.window_size:
            # Zero-pad
            padding = self.window_size - len(iq_data)
            iq_data = np.pad(iq_data, (padding // 2, padding - padding // 2))

        # Preprocess
        i_channel = np.real(iq_data)
        q_channel = np.imag(iq_data)

        # Normalize
        max_val = max(np.abs(i_channel).max(), np.abs(q_channel).max()) + 1e-10
        i_channel = i_channel / max_val
        q_channel = q_channel / max_val

        # Create tensor
        iq_tensor = np.stack([i_channel, q_channel], axis=0)
        iq_tensor = torch.from_numpy(iq_tensor).float().unsqueeze(0)
        iq_tensor = iq_tensor.to(self.device)

        # Inference
        start_time = time.time()
        output, strength = self.model(iq_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Get prediction
        confidence = torch.sigmoid(output).item()
        detected = confidence > 0.5
        strength_val = strength.item()

        # Update history
        current_time = time.time() - self.start_time
        self.detection_history.append(1 if detected else 0)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(current_time)

        self.total_processed += 1
        if detected:
            self.total_detections += 1

        return {
            'detected': detected,
            'confidence': confidence,
            'strength': strength_val,
            'inference_time_ms': inference_time,
            'timestamp': current_time,
        }

    def get_statistics(self) -> dict:
        """Get detection statistics"""

        if self.total_processed == 0:
            return {
                'total_processed': 0,
                'total_detections': 0,
                'detection_rate': 0.0,
                'avg_confidence': 0.0,
                'runtime_seconds': 0.0,
            }

        return {
            'total_processed': self.total_processed,
            'total_detections': self.total_detections,
            'detection_rate': self.total_detections / self.total_processed,
            'avg_confidence': np.mean(list(self.confidence_history)),
            'runtime_seconds': time.time() - self.start_time,
            'throughput_samples_per_sec': self.total_processed / (time.time() - self.start_time),
        }

    def run_on_dataset(
        self,
        data_dir: str,
        difficulty: str = "easy",
        num_files: int = 10,
        visualize: bool = True,
    ):
        """Run live detection on dataset files"""

        from backend.datasets.zelda_loader import ZeldaDataset

        # Load dataset
        dataset = ZeldaDataset(
            data_dir=data_dir,
            difficulty=difficulty,
            window_size=self.window_size,
        )

        logger.info(f"Processing {min(num_files, len(dataset))} samples from {difficulty} dataset...")

        # Setup visualization if requested
        if visualize:
            plt.ion()
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        detection_count = 0

        for idx in range(min(num_files * 100, len(dataset))):
            iq_tensor, label, metadata = dataset[idx]

            # Convert tensor back to numpy complex
            i_vals = iq_tensor[0].numpy()
            q_vals = iq_tensor[1].numpy()
            iq_data = i_vals + 1j * q_vals

            # Detect
            result = self.detect(iq_data)

            if result['detected']:
                detection_count += 1

                logger.info(
                    f"[{idx}] DETECTION! Confidence: {result['confidence']:.3f}, "
                    f"Strength: {result['strength']:.3f}, "
                    f"Inference: {result['inference_time_ms']:.2f}ms"
                )

            # Update visualization every 10 samples
            if visualize and idx % 10 == 0:
                self._update_visualization(axes, iq_data, result)

            # Small delay for visualization
            if visualize:
                plt.pause(0.01)

        # Final statistics
        stats = self.get_statistics()
        logger.info("\n" + "=" * 50)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total Processed:    {stats['total_processed']}")
        logger.info(f"Total Detections:   {stats['total_detections']}")
        logger.info(f"Detection Rate:     {stats['detection_rate']:.2%}")
        logger.info(f"Avg Confidence:     {stats['avg_confidence']:.3f}")
        logger.info(f"Runtime:            {stats['runtime_seconds']:.1f}s")
        logger.info(f"Throughput:         {stats['throughput_samples_per_sec']:.1f} samples/sec")
        logger.info("=" * 50)

        if visualize:
            plt.ioff()
            plt.show()

    def _update_visualization(self, axes, iq_data, result):
        """Update real-time visualization"""

        for ax in axes:
            ax.clear()

        # 1. I/Q time series
        t = np.arange(len(iq_data)) / 40e6  # Assuming 40 MHz sample rate
        axes[0].plot(t * 1e6, np.real(iq_data), label='I', alpha=0.7)
        axes[0].plot(t * 1e6, np.imag(iq_data), label='Q', alpha=0.7)
        axes[0].set_xlabel('Time (μs)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f"I/Q Data - {'SIGNAL DETECTED' if result['detected'] else 'No Signal'}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Spectrum
        fft = np.fft.fftshift(np.fft.fft(iq_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/40e6))
        psd = 10 * np.log10(np.abs(fft)**2 + 1e-10)

        axes[1].plot(freqs / 1e6, psd)
        axes[1].set_xlabel('Frequency (MHz)')
        axes[1].set_ylabel('Power (dB)')
        axes[1].set_title(f"Spectrum - Confidence: {result['confidence']:.3f}")
        axes[1].grid(True, alpha=0.3)

        # 3. Detection history
        if len(self.timestamp_history) > 0:
            axes[2].plot(list(self.timestamp_history), list(self.confidence_history), 'b-', alpha=0.7)
            axes[2].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
            axes[2].fill_between(
                list(self.timestamp_history),
                list(self.confidence_history),
                0.5,
                where=np.array(list(self.confidence_history)) > 0.5,
                alpha=0.3,
                color='green',
                label='Detected'
            )
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Detection Confidence')
            axes[2].set_title('Detection History')
            axes[2].set_ylim([0, 1])
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Live Signal Detection")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model-type", type=str, default="ultra", choices=["ultra", "transformer", "ensemble"])
    parser.add_argument("--data-dir", type=str, default="./data/datasets")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--num-files", type=int, default=10, help="Number of files to process")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    detector = LiveDetector(
        model_path=args.model_path,
        model_type=args.model_type,
    )

    detector.run_on_dataset(
        data_dir=args.data_dir,
        difficulty=args.difficulty,
        num_files=args.num_files,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
