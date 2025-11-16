"""
AERPAW TDOA Dataset Loader

Dataset: AERPAW RF Sensor Measurements with UAV (July 2024)
Source: https://aerpaw.org/dataset/aerpaw-rf-sensor-measurements-with-uav-july-2024/

Real-world TDOA measurements of UAV RF signals for 3D geolocation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from loguru import logger
from dataclasses import dataclass


@dataclass
class TDOAMeasurementRecord:
    """Single TDOA measurement record"""
    timestamp: float
    tdoa_estimate: Tuple[float, float, float]  # Estimated (x, y, z)
    true_position: Tuple[float, float, float]  # Ground truth (x, y, z)
    bandwidth: float  # Hz
    altitude: float  # meters
    los_nlos: Dict[str, bool]  # LOS/NLOS indicators per tower
    error: float  # meters (computed)


class AERPAWLoader:
    """
    Loader for AERPAW TDOA UAV Geolocation Dataset.

    This dataset contains:
    - TDOA position estimates
    - Ground-truth UAV GPS positions
    - LOS/NLOS indicators for each RF sensor
    - Multiple bandwidths (1.25, 2.5, 5 MHz)
    - Multiple altitudes (40, 70, 100 m)
    - RF center frequency: 3.32 GHz
    """

    def __init__(self, dataset_path: str):
        """
        Initialize AERPAW loader.

        Args:
            dataset_path: Path to AERPAW dataset directory
        """

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}.\n"
                "Download from: https://aerpaw.org/dataset/aerpaw-rf-sensor-measurements-with-uav-july-2024/"
            )

        self.records: List[TDOAMeasurementRecord] = []

        logger.info(f"AERPAW loader initialized: {dataset_path}")

    def load(self) -> List[TDOAMeasurementRecord]:
        """
        Load the dataset.

        Returns:
            List of TDOAMeasurementRecord objects
        """

        logger.info("Loading AERPAW TDOA dataset...")

        # Find CSV files in dataset
        csv_files = list(self.dataset_path.glob("**/*.csv"))

        if not csv_files:
            raise FileNotFoundError("No CSV files found in dataset directory")

        all_records = []

        for csv_file in csv_files:
            logger.info(f"Loading {csv_file.name}")
            records = self._load_csv(csv_file)
            all_records.extend(records)

        self.records = all_records

        logger.info(f"Loaded {len(all_records)} TDOA measurements")

        return all_records

    def _load_csv(self, csv_path: Path) -> List[TDOAMeasurementRecord]:
        """Load a single CSV file"""

        df = pd.read_csv(csv_path)

        records = []

        for _, row in df.iterrows():
            # Extract positions
            # (Format depends on actual AERPAW dataset structure)
            # Adjust column names based on actual data

            try:
                # TDOA estimate
                est_x = float(row.get('est_x', row.get('tdoa_x', 0)))
                est_y = float(row.get('est_y', row.get('tdoa_y', 0)))
                est_z = float(row.get('est_z', row.get('tdoa_z', 0)))

                # Ground truth
                true_x = float(row.get('true_x', row.get('gps_x', row.get('gt_x', 0))))
                true_y = float(row.get('true_y', row.get('gps_y', row.get('gt_y', 0))))
                true_z = float(row.get('true_z', row.get('gps_z', row.get('gt_z', row.get('altitude', 0)))))

                # Metadata
                timestamp = float(row.get('timestamp', row.get('time', 0)))
                bandwidth = float(row.get('bandwidth', row.get('bw', 2.5e6)))
                altitude = float(row.get('altitude', row.get('alt', true_z)))

                # LOS/NLOS indicators (if available)
                los_nlos = {}
                for col in df.columns:
                    if 'los' in col.lower() or 'nlos' in col.lower():
                        los_nlos[col] = bool(row[col])

                # Calculate error
                error = np.linalg.norm(
                    np.array([est_x, est_y, est_z]) - np.array([true_x, true_y, true_z])
                )

                record = TDOAMeasurementRecord(
                    timestamp=timestamp,
                    tdoa_estimate=(est_x, est_y, est_z),
                    true_position=(true_x, true_y, true_z),
                    bandwidth=bandwidth,
                    altitude=altitude,
                    los_nlos=los_nlos,
                    error=error
                )

                records.append(record)

            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue

        return records

    def get_filtered_records(
        self,
        bandwidth: Optional[float] = None,
        altitude: Optional[float] = None,
        max_error: Optional[float] = None,
        los_only: bool = False,
    ) -> List[TDOAMeasurementRecord]:
        """
        Get filtered records.

        Args:
            bandwidth: Filter by bandwidth (Hz)
            altitude: Filter by altitude (m)
            max_error: Maximum acceptable error (m)
            los_only: Only include LOS measurements

        Returns:
            Filtered list of records
        """

        if not self.records:
            self.load()

        filtered = self.records

        if bandwidth:
            filtered = [r for r in filtered if abs(r.bandwidth - bandwidth) < 1e3]

        if altitude:
            filtered = [r for r in filtered if abs(r.altitude - altitude) < 5]

        if max_error:
            filtered = [r for r in filtered if r.error <= max_error]

        if los_only:
            # Check if all towers have LOS
            filtered = [
                r for r in filtered
                if r.los_nlos and all(r.los_nlos.values())
            ]

        return filtered

    def get_statistics(self) -> dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary of statistics
        """

        if not self.records:
            self.load()

        errors = [r.error for r in self.records]
        bandwidths = [r.bandwidth for r in self.records]
        altitudes = [r.altitude for r in self.records]

        stats = {
            'total_records': len(self.records),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'p50_error': np.percentile(errors, 50),
            'p90_error': np.percentile(errors, 90),
            'p95_error': np.percentile(errors, 95),
            'unique_bandwidths': sorted(set(bandwidths)),
            'unique_altitudes': sorted(set(altitudes)),
        }

        return stats

    def get_performance_by_snr(self) -> Dict[str, dict]:
        """
        Analyze performance stratified by various factors.

        Returns:
            Dictionary of performance metrics
        """

        if not self.records:
            self.load()

        performance = {}

        # By bandwidth
        for bw in set(r.bandwidth for r in self.records):
            bw_records = [r for r in self.records if r.bandwidth == bw]
            errors = [r.error for r in bw_records]

            performance[f'bandwidth_{bw/1e6:.2f}MHz'] = {
                'count': len(bw_records),
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'p90_error': np.percentile(errors, 90),
            }

        # By altitude
        for alt in set(r.altitude for r in self.records):
            alt_records = [r for r in self.records if abs(r.altitude - alt) < 5]
            errors = [r.error for r in alt_records]

            performance[f'altitude_{alt:.0f}m'] = {
                'count': len(alt_records),
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'p90_error': np.percentile(errors, 90),
            }

        return performance

    def export_for_zelda(self) -> List[Dict]:
        """
        Export in Zelda-compatible format.

        Returns:
            List of measurement dictionaries
        """

        if not self.records:
            self.load()

        exports = []

        for record in self.records:
            export = {
                'timestamp': record.timestamp,
                'estimated_position': record.tdoa_estimate,
                'true_position': record.true_position,
                'error_meters': record.error,
                'metadata': {
                    'bandwidth': record.bandwidth,
                    'altitude': record.altitude,
                    'los_nlos': record.los_nlos,
                }
            }
            exports.append(export)

        return exports

    @staticmethod
    def download_instructions():
        """Print download instructions"""

        print("""
AERPAW TDOA Dataset Download Instructions:

1. Visit: https://aerpaw.org/dataset/aerpaw-rf-sensor-measurements-with-uav-july-2024/
2. Register for AERPAW account (if needed)
3. Download dataset (CSV or data files)
4. Extract to: ~/zelda/data/datasets/AERPAW_TDOA_2024/

Dataset Details:
- Frequency: 3.32 GHz
- Bandwidths: 1.25 MHz, 2.5 MHz, 5 MHz
- Altitudes: 40m, 70m, 100m
- Sensors: 4x Keysight N6841A RF sensors
- Ground Truth: High-precision GPS

Data includes:
- TDOA position estimates
- GPS ground truth
- LOS/NLOS indicators
- Flight trajectories
        """)


if __name__ == "__main__":
    # Example usage
    AERPAWLoader.download_instructions()
