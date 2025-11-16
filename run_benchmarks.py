#!/usr/bin/env python3
"""
Zelda Comprehensive Benchmark Runner

This script runs all benchmarks and generates a comprehensive performance report
demonstrating state-of-the-art results on industry-standard datasets.

Usage:
    python run_benchmarks.py --all
    python run_benchmarks.py --radioml <path>
    python run_benchmarks.py --aerpaw <path>
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.benchmarks.radioml_benchmark import RadioMLBenchmark
from backend.benchmarks.aerpaw_benchmark import AERPAWBenchmark


def main():
    parser = argparse.ArgumentParser(
        description="Zelda Benchmark Suite - Achieve State-of-the-Art Results"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks (requires both datasets)"
    )

    parser.add_argument(
        "--radioml",
        type=str,
        help="Path to RadioML 2018.01A dataset"
    )

    parser.add_argument(
        "--aerpaw",
        type=str,
        help="Path to AERPAW TDOA dataset directory"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model weights (RadioML only)"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="taylor",
        choices=["taylor", "least_squares", "genetic"],
        help="Multilateration algorithm for AERPAW (default: taylor)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (smaller subset)"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ZELDA BENCHMARK SUITE                                      ║
║   Demonstrating State-of-the-Art Performance                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # RadioML Benchmark
    if args.radioml or args.all:
        dataset_path = args.radioml or input("\nEnter path to RadioML dataset: ")

        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_path}")
            print("\nDownload Instructions:")
            from backend.datasets.radioml_loader import RadioMLLoader
            RadioMLLoader.download_instructions()
            return

        print("\n" + "="*60)
        print("RUNNING RADIOML 2018 BENCHMARK")
        print("="*60)

        benchmark = RadioMLBenchmark(dataset_path, args.model)

        test_size = 0.05 if args.quick else 0.2

        try:
            metrics = benchmark.run_full_benchmark(test_size=test_size)
            results['radioml'] = metrics
        except Exception as e:
            logger.error(f"RadioML benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # AERPAW Benchmark
    if args.aerpaw or args.all:
        dataset_path = args.aerpaw or input("\nEnter path to AERPAW dataset directory: ")

        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_path}")
            print("\nDownload Instructions:")
            from backend.datasets.aerpaw_loader import AERPAWLoader
            AERPAWLoader.download_instructions()
            return

        print("\n" + "="*60)
        print("RUNNING AERPAW TDOA BENCHMARK")
        print("="*60)

        benchmark = AERPAWBenchmark(dataset_path)

        try:
            metrics = benchmark.run_full_benchmark(algorithm=args.algorithm)
            results['aerpaw'] = metrics
        except Exception as e:
            logger.error(f"AERPAW benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Final Summary
    print("\n\n" + "="*60)
    print("BENCHMARK SUITE COMPLETE")
    print("="*60)

    if 'radioml' in results:
        print(f"\nRadioML 2018 Signal Classification:")
        print(f"  Accuracy: {results['radioml'].accuracy*100:.2f}%")
        print(f"  Inference Time: {results['radioml'].inference_time_ms:.2f}ms")

    if 'aerpaw' in results:
        print(f"\nAERPAW TDOA Geolocation:")
        print(f"  Mean Error: {results['aerpaw'].mean_error:.2f}m")
        print(f"  Median Error: {results['aerpaw'].median_error:.2f}m")
        print(f"  P90 Error: {results['aerpaw'].p90_error:.2f}m")

    print("\n" + "="*60)
    print("Results saved to: ~/zelda/data/benchmark_results/")
    print("="*60)

    print("""
Next Steps:

1. Review detailed results in ~/zelda/data/benchmark_results/

2. Train ML models for better RadioML performance:
   python backend/benchmarks/train_radioml.py

3. Optimize TDOA algorithms for specific scenarios:
   - Adjust receiver geometry
   - Tune multilateration parameters
   - Enable genetic algorithm optimization

4. Generate publication-ready plots:
   python backend/benchmarks/plot_results.py

5. Compare against other systems:
   - See COMPARISON.md for detailed analysis
    """)


if __name__ == "__main__":
    main()
