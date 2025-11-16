"""
RadioML 2018 Benchmark for Signal Classification

Tests Zelda's ML signal classifier against the industry-standard
RadioML 2018.01A dataset.

Target Performance (State-of-the-Art):
- Overall Accuracy: >62% (all SNRs)
- High SNR (>18dB): >90%
- Medium SNR (6-18dB): >75%
- Low SNR (<6dB): >40%
"""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import time
from dataclasses import dataclass, asdict
import json

from ..datasets.radioml_loader import RadioMLLoader
from ..core.ml.signal_classifier import SignalClassifier, ModulationType


@dataclass
class ClassificationMetrics:
    """Metrics for signal classification"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    per_snr_accuracy: Dict[int, float]
    per_class_accuracy: Dict[str, float]
    inference_time_ms: float


class RadioMLBenchmark:
    """
    Comprehensive benchmark for signal classification on RadioML.
    """

    def __init__(
        self,
        dataset_path: str,
        model_path: Optional[str] = None,
    ):
        """
        Initialize benchmark.

        Args:
            dataset_path: Path to RadioML dataset
            model_path: Path to trained model weights (None for untrained)
        """

        self.loader = RadioMLLoader(dataset_path)
        self.classifier = SignalClassifier(
            model_path=model_path,
            signal_length=1024,
        )

        logger.info("RadioML Benchmark initialized")

    def run_full_benchmark(
        self,
        test_size: float = 0.2,
        save_results: bool = True,
    ) -> ClassificationMetrics:
        """
        Run complete benchmark on RadioML dataset.

        Args:
            test_size: Fraction of data for testing
            save_results: Save results to file

        Returns:
            ClassificationMetrics
        """

        logger.info("="*60)
        logger.info("RadioML 2018 Benchmark - Starting")
        logger.info("="*60)

        # Load and split data
        logger.info("Loading dataset...")
        (X_train, y_train, snr_train), (X_test, y_test, snr_test) = \
            self.loader.get_train_test_split(test_size=test_size)

        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")

        # Run inference on test set
        logger.info("\nRunning inference...")

        predictions = []
        inference_times = []

        batch_size = 128

        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]

            start_time = time.time()
            results = self.classifier.batch_classify(batch)
            inference_times.append((time.time() - start_time) / len(batch))

            predictions.extend([
                list(ModulationType).index(r.modulation)
                for r in results
            ])

            if (i // batch_size) % 10 == 0:
                logger.info(f"  Processed {i}/{len(X_test)} samples...")

        predictions = np.array(predictions)

        # Calculate metrics
        logger.info("\nCalculating metrics...")

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        metrics = ClassificationMetrics(
            accuracy=accuracy_score(y_test, predictions),
            precision=precision_score(y_test, predictions, average='weighted', zero_division=0),
            recall=recall_score(y_test, predictions, average='weighted', zero_division=0),
            f1_score=f1_score(y_test, predictions, average='weighted', zero_division=0),
            confusion_matrix=confusion_matrix(y_test, predictions),
            per_snr_accuracy=self._calculate_per_snr_accuracy(y_test, predictions, snr_test),
            per_class_accuracy=self._calculate_per_class_accuracy(y_test, predictions),
            inference_time_ms=np.mean(inference_times) * 1000,
        )

        # Print results
        self._print_results(metrics)

        # Save if requested
        if save_results:
            self._save_results(metrics)

        return metrics

    def _calculate_per_snr_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        snrs: np.ndarray
    ) -> Dict[int, float]:
        """Calculate accuracy for each SNR level"""

        from sklearn.metrics import accuracy_score

        per_snr = {}

        for snr in sorted(set(snrs)):
            mask = snrs == snr
            if np.any(mask):
                acc = accuracy_score(y_true[mask], y_pred[mask])
                per_snr[int(snr)] = float(acc)

        return per_snr

    def _calculate_per_class_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate accuracy for each modulation class"""

        from sklearn.metrics import accuracy_score

        per_class = {}

        for i, mod in enumerate(RadioMLLoader.MODULATIONS):
            mask = y_true == i
            if np.any(mask):
                acc = accuracy_score(y_true[mask], y_pred[mask])
                per_class[mod] = float(acc)

        return per_class

    def _print_results(self, metrics: ClassificationMetrics):
        """Print benchmark results"""

        logger.info("\n" + "="*60)
        logger.info("RESULTS")
        logger.info("="*60)

        logger.info(f"\nOverall Performance:")
        logger.info(f"  Accuracy:  {metrics.accuracy*100:.2f}%")
        logger.info(f"  Precision: {metrics.precision*100:.2f}%")
        logger.info(f"  Recall:    {metrics.recall*100:.2f}%")
        logger.info(f"  F1 Score:  {metrics.f1_score*100:.2f}%")
        logger.info(f"  Inference: {metrics.inference_time_ms:.2f}ms per sample")

        logger.info(f"\nPerformance by SNR:")
        for snr, acc in sorted(metrics.per_snr_accuracy.items()):
            logger.info(f"  SNR {snr:+3d}dB: {acc*100:.2f}%")

        logger.info(f"\nTop 5 Classes (by accuracy):")
        sorted_classes = sorted(
            metrics.per_class_accuracy.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for mod, acc in sorted_classes[:5]:
            logger.info(f"  {mod:12s}: {acc*100:.2f}%")

        logger.info(f"\nBottom 5 Classes:")
        for mod, acc in sorted_classes[-5:]:
            logger.info(f"  {mod:12s}: {acc*100:.2f}%")

        # SNR performance bands
        logger.info(f"\nPerformance Bands:")

        high_snr = [acc for snr, acc in metrics.per_snr_accuracy.items() if snr >= 18]
        med_snr = [acc for snr, acc in metrics.per_snr_accuracy.items() if 6 <= snr < 18]
        low_snr = [acc for snr, acc in metrics.per_snr_accuracy.items() if snr < 6]

        if high_snr:
            logger.info(f"  High SNR (≥18dB):  {np.mean(high_snr)*100:.2f}%")
        if med_snr:
            logger.info(f"  Med SNR (6-18dB):  {np.mean(med_snr)*100:.2f}%")
        if low_snr:
            logger.info(f"  Low SNR (<6dB):    {np.mean(low_snr)*100:.2f}%")

        logger.info("\n" + "="*60)

    def _save_results(self, metrics: ClassificationMetrics):
        """Save results to JSON"""

        from pathlib import Path

        results_dir = Path("~/zelda/data/benchmark_results").expanduser()
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"radioml_benchmark_{timestamp}.json"

        # Convert to serializable format
        results = {
            'accuracy': float(metrics.accuracy),
            'precision': float(metrics.precision),
            'recall': float(metrics.recall),
            'f1_score': float(metrics.f1_score),
            'inference_time_ms': float(metrics.inference_time_ms),
            'per_snr_accuracy': {str(k): float(v) for k, v in metrics.per_snr_accuracy.items()},
            'per_class_accuracy': {k: float(v) for k, v in metrics.per_class_accuracy.items()},
            'confusion_matrix': metrics.confusion_matrix.tolist(),
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python radioml_benchmark.py <dataset_path> [model_path]")
        print("\nDataset path should point to RADIOML_2018.01A.hdf5")
        print("Model path is optional (uses untrained model if not provided)")
        sys.exit(1)

    dataset_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Run benchmark
    benchmark = RadioMLBenchmark(dataset_path, model_path)
    metrics = benchmark.run_full_benchmark()

    # Compare to state-of-the-art
    print("\n" + "="*60)
    print("COMPARISON TO STATE-OF-THE-ART")
    print("="*60)
    print(f"Zelda:         {metrics.accuracy*100:.2f}%")
    print(f"SOTA (2024):   ~65-70% (trained models)")
    print(f"Baseline CNN:  ~60%")
    print(f"Random Guess:  ~4.2% (24 classes)")
    print("="*60)
