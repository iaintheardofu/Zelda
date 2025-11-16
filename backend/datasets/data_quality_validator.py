"""
ZELDA Data Quality Validation Framework

Ensures integrity of training datasets and detects cross-contamination.
Based on research: "On Contamination in Modern Benchmarks for Generated Text Detection"

Key Features:
- Training/test set overlap detection
- Cross-dataset contamination checking
- Benchmark integrity validation
- Data quality metrics
- Automated cleaning recommendations

Critical for ensuring ML models are trained on clean, uncontaminated data.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import logging
from datetime import datetime
import hashlib
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class ContaminationReport:
    """Results from contamination detection analysis"""
    is_contaminated: bool
    contamination_type: str
    contamination_percentage: float
    total_samples: int
    contaminated_samples: int
    contaminated_indices: List[int]
    recommendations: List[str]
    details: Dict


class DataQualityValidator:
    """
    Validates quality and integrity of training datasets.

    Based on Paper 3: Detects various types of data contamination
    that can artificially inflate model performance.
    """

    def __init__(self):
        """Initialize data quality validator"""
        self.dataset_hashes = {}  # Store dataset fingerprints
        self.known_datasets = {}  # Store known dataset metadata

        logger.info("DataQualityValidator initialized")

    def validate_train_test_split(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        similarity_threshold: float = 0.95
    ) -> ContaminationReport:
        """
        Detect overlap between training and test sets.

        From Paper 3: "Even small train-test contamination can lead to
        15-30% artificial accuracy improvements"

        Args:
            train_data: Training dataset (N × features)
            test_data: Test dataset (M × features)
            similarity_threshold: Threshold for considering samples as duplicates

        Returns:
            ContaminationReport with analysis results
        """
        logger.info(
            f"Validating train/test split: "
            f"{len(train_data)} train, {len(test_data)} test samples"
        )

        contaminated_indices = []
        contamination_details = []

        # Check each test sample against training set
        for test_idx in range(len(test_data)):
            test_sample = test_data[test_idx]

            # Calculate similarity to all training samples
            similarities = self._calculate_similarities(
                test_sample, train_data
            )

            # Find maximum similarity
            max_similarity = np.max(similarities)
            max_similarity_idx = np.argmax(similarities)

            if max_similarity >= similarity_threshold:
                contaminated_indices.append(test_idx)
                contamination_details.append({
                    'test_idx': int(test_idx),
                    'train_idx': int(max_similarity_idx),
                    'similarity': float(max_similarity)
                })

        # Calculate contamination percentage
        contamination_pct = (len(contaminated_indices) / len(test_data)) * 100

        # Determine if contaminated (>1% is concerning)
        is_contaminated = contamination_pct > 1.0

        # Generate recommendations
        recommendations = self._generate_split_recommendations(
            contamination_pct, len(contaminated_indices)
        )

        return ContaminationReport(
            is_contaminated=is_contaminated,
            contamination_type="train_test_overlap",
            contamination_percentage=contamination_pct,
            total_samples=len(test_data),
            contaminated_samples=len(contaminated_indices),
            contaminated_indices=contaminated_indices,
            recommendations=recommendations,
            details={'overlap_details': contamination_details[:10]}  # Top 10
        )

    def detect_cross_dataset_contamination(
        self,
        dataset_a: np.ndarray,
        dataset_b: np.ndarray,
        dataset_a_name: str = "Dataset A",
        dataset_b_name: str = "Dataset B",
        similarity_threshold: float = 0.95
    ) -> ContaminationReport:
        """
        Detect contamination between two different datasets.

        From Paper 3: "Cross-dataset contamination occurs when samples
        from one benchmark appear in another, inflating generalization claims"

        Args:
            dataset_a: First dataset
            dataset_b: Second dataset
            dataset_a_name: Name of first dataset
            dataset_b_name: Name of second dataset
            similarity_threshold: Threshold for duplicate detection

        Returns:
            ContaminationReport with cross-contamination analysis
        """
        logger.info(
            f"Checking cross-contamination: {dataset_a_name} ({len(dataset_a)} samples) "
            f"vs {dataset_b_name} ({len(dataset_b)} samples)"
        )

        contaminated_indices = []
        contamination_details = []

        # Check each sample in dataset_a against dataset_b
        for idx_a in range(len(dataset_a)):
            sample_a = dataset_a[idx_a]

            similarities = self._calculate_similarities(sample_a, dataset_b)
            max_similarity = np.max(similarities)
            max_similarity_idx = np.argmax(similarities)

            if max_similarity >= similarity_threshold:
                contaminated_indices.append(idx_a)
                contamination_details.append({
                    f'{dataset_a_name}_idx': int(idx_a),
                    f'{dataset_b_name}_idx': int(max_similarity_idx),
                    'similarity': float(max_similarity)
                })

        contamination_pct = (len(contaminated_indices) / len(dataset_a)) * 100
        is_contaminated = contamination_pct > 0.5  # Even 0.5% is concerning

        recommendations = self._generate_cross_dataset_recommendations(
            contamination_pct, dataset_a_name, dataset_b_name
        )

        return ContaminationReport(
            is_contaminated=is_contaminated,
            contamination_type="cross_dataset_contamination",
            contamination_percentage=contamination_pct,
            total_samples=len(dataset_a),
            contaminated_samples=len(contaminated_indices),
            contaminated_indices=contaminated_indices,
            recommendations=recommendations,
            details={
                'dataset_a': dataset_a_name,
                'dataset_b': dataset_b_name,
                'overlap_details': contamination_details[:10]
            }
        )

    def detect_duplicate_samples(
        self,
        dataset: np.ndarray,
        similarity_threshold: float = 0.99
    ) -> ContaminationReport:
        """
        Detect duplicate or near-duplicate samples within a dataset.

        Args:
            dataset: Dataset to check (N × features)
            similarity_threshold: Threshold for considering samples as duplicates

        Returns:
            ContaminationReport with duplicate analysis
        """
        logger.info(f"Checking for duplicates in dataset ({len(dataset)} samples)")

        duplicate_pairs = []
        duplicate_indices = set()

        # Check all pairs (optimized with hashing for large datasets)
        sample_hashes = {}

        for idx, sample in enumerate(dataset):
            # Create hash for quick comparison
            sample_hash = self._hash_sample(sample)

            if sample_hash in sample_hashes:
                # Potential duplicate, verify with detailed comparison
                for other_idx in sample_hashes[sample_hash]:
                    similarity = self._calculate_sample_similarity(
                        dataset[idx], dataset[other_idx]
                    )

                    if similarity >= similarity_threshold:
                        duplicate_pairs.append((idx, other_idx, similarity))
                        duplicate_indices.add(idx)
                        duplicate_indices.add(other_idx)

            if sample_hash not in sample_hashes:
                sample_hashes[sample_hash] = []
            sample_hashes[sample_hash].append(idx)

        duplicate_list = sorted(list(duplicate_indices))
        contamination_pct = (len(duplicate_list) / len(dataset)) * 100
        is_contaminated = contamination_pct > 0.1  # >0.1% duplicates

        recommendations = self._generate_duplicate_recommendations(
            contamination_pct, len(duplicate_list)
        )

        return ContaminationReport(
            is_contaminated=is_contaminated,
            contamination_type="duplicate_samples",
            contamination_percentage=contamination_pct,
            total_samples=len(dataset),
            contaminated_samples=len(duplicate_list),
            contaminated_indices=duplicate_list,
            recommendations=recommendations,
            details={'duplicate_pairs': duplicate_pairs[:20]}
        )

    def validate_class_balance(
        self,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Validate class balance in dataset.

        From Paper 3: "Severe class imbalance can lead to biased models
        with artificially high accuracy on majority class"

        Args:
            labels: Class labels array
            class_names: Optional list of class names

        Returns:
            Dictionary with class balance analysis
        """
        unique_labels, counts = np.unique(labels, return_counts=True)

        total_samples = len(labels)
        class_distribution = {}

        for label, count in zip(unique_labels, counts):
            label_name = class_names[label] if class_names else f"Class_{label}"
            percentage = (count / total_samples) * 100
            class_distribution[label_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }

        # Calculate imbalance ratio
        max_count = np.max(counts)
        min_count = np.min(counts)
        imbalance_ratio = max_count / min_count

        # Determine if severely imbalanced
        is_imbalanced = imbalance_ratio > 10  # 10:1 ratio is concerning

        recommendations = []
        if is_imbalanced:
            recommendations.append(
                f"SEVERE IMBALANCE: {imbalance_ratio:.1f}:1 ratio detected"
            )
            recommendations.append("Consider class rebalancing techniques:")
            recommendations.append("  • Oversampling minority classes (SMOTE)")
            recommendations.append("  • Undersampling majority classes")
            recommendations.append("  • Class-weighted loss functions")
            recommendations.append("  • Stratified sampling")

        return {
            'class_distribution': class_distribution,
            'imbalance_ratio': float(imbalance_ratio),
            'is_imbalanced': is_imbalanced,
            'total_samples': int(total_samples),
            'num_classes': len(unique_labels),
            'recommendations': recommendations
        }

    def calculate_dataset_statistics(self, dataset: np.ndarray) -> Dict:
        """
        Calculate comprehensive dataset quality statistics.

        Args:
            dataset: Dataset array (N × features)

        Returns:
            Dictionary with statistical analysis
        """
        stats = {}

        # Basic statistics
        stats['num_samples'] = len(dataset)
        stats['num_features'] = dataset.shape[1] if dataset.ndim > 1 else 1

        # Feature statistics
        if dataset.ndim > 1:
            stats['feature_means'] = np.mean(dataset, axis=0).tolist()
            stats['feature_stds'] = np.std(dataset, axis=0).tolist()
            stats['feature_mins'] = np.min(dataset, axis=0).tolist()
            stats['feature_maxs'] = np.max(dataset, axis=0).tolist()

            # Check for zero-variance features
            zero_var_features = np.where(np.std(dataset, axis=0) < 1e-10)[0]
            stats['zero_variance_features'] = zero_var_features.tolist()

            # Check for features with missing/infinite values
            has_nan = np.any(np.isnan(dataset), axis=0)
            has_inf = np.any(np.isinf(dataset), axis=0)
            stats['features_with_nan'] = np.where(has_nan)[0].tolist()
            stats['features_with_inf'] = np.where(has_inf)[0].tolist()

        # Overall data quality score
        quality_score = 100.0

        if stats.get('zero_variance_features'):
            quality_score -= len(stats['zero_variance_features']) * 5
        if stats.get('features_with_nan'):
            quality_score -= len(stats['features_with_nan']) * 10
        if stats.get('features_with_inf'):
            quality_score -= len(stats['features_with_inf']) * 10

        stats['quality_score'] = max(quality_score, 0.0)

        return stats

    def _calculate_similarities(
        self,
        sample: np.ndarray,
        dataset: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between a sample and all samples in dataset.

        Uses cosine similarity for efficiency.
        """
        # Normalize
        sample_norm = sample / (np.linalg.norm(sample) + 1e-10)
        dataset_norm = dataset / (np.linalg.norm(dataset, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity
        similarities = np.dot(dataset_norm, sample_norm)

        return similarities

    def _calculate_sample_similarity(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> float:
        """Calculate similarity between two samples"""
        # Cosine similarity
        dot_product = np.dot(sample1, sample2)
        norm_product = np.linalg.norm(sample1) * np.linalg.norm(sample2)

        if norm_product == 0:
            return 0.0

        similarity = dot_product / norm_product
        return float(similarity)

    def _hash_sample(self, sample: np.ndarray) -> str:
        """Create hash of sample for quick duplicate detection"""
        # Round to reduce sensitivity to tiny differences
        rounded = np.round(sample, decimals=4)

        # Create hash
        sample_bytes = rounded.tobytes()
        hash_obj = hashlib.md5(sample_bytes)

        return hash_obj.hexdigest()

    def _generate_split_recommendations(
        self,
        contamination_pct: float,
        num_contaminated: int
    ) -> List[str]:
        """Generate recommendations for train/test split contamination"""
        recommendations = []

        if contamination_pct > 10:
            recommendations.append(
                f"CRITICAL: {contamination_pct:.1f}% train/test overlap detected!"
            )
            recommendations.append("Action: RECREATE train/test split immediately")
            recommendations.append("Use stratified random split with different seed")
        elif contamination_pct > 5:
            recommendations.append(
                f"SEVERE: {contamination_pct:.1f}% train/test overlap"
            )
            recommendations.append("Action: Remove contaminated samples from test set")
            recommendations.append(f"Remove {num_contaminated} samples")
        elif contamination_pct > 1:
            recommendations.append(
                f"WARNING: {contamination_pct:.1f}% train/test overlap"
            )
            recommendations.append("Consider removing contaminated samples")
            recommendations.append("Monitor performance metrics carefully")
        else:
            recommendations.append("✓ Train/test split appears clean")
            recommendations.append(f"Only {contamination_pct:.2f}% overlap (acceptable)")

        return recommendations

    def _generate_cross_dataset_recommendations(
        self,
        contamination_pct: float,
        dataset_a_name: str,
        dataset_b_name: str
    ) -> List[str]:
        """Generate recommendations for cross-dataset contamination"""
        recommendations = []

        if contamination_pct > 5:
            recommendations.append(
                f"SEVERE: {contamination_pct:.1f}% contamination between datasets"
            )
            recommendations.append(
                f"WARNING: Cannot claim {dataset_a_name} generalizes to {dataset_b_name}"
            )
            recommendations.append("Action: Remove overlapping samples before benchmarking")
        elif contamination_pct > 1:
            recommendations.append(
                f"WARNING: {contamination_pct:.1f}% cross-dataset overlap"
            )
            recommendations.append("Consider this when interpreting generalization results")
        else:
            recommendations.append("✓ Minimal cross-dataset contamination")
            recommendations.append(f"Only {contamination_pct:.2f}% overlap")

        return recommendations

    def _generate_duplicate_recommendations(
        self,
        contamination_pct: float,
        num_duplicates: int
    ) -> List[str]:
        """Generate recommendations for duplicate samples"""
        recommendations = []

        if contamination_pct > 5:
            recommendations.append(
                f"SEVERE: {contamination_pct:.1f}% duplicate samples"
            )
            recommendations.append(f"Action: Remove {num_duplicates} duplicate samples")
            recommendations.append("Duplicates artificially inflate model performance")
        elif contamination_pct > 0.5:
            recommendations.append(
                f"WARNING: {contamination_pct:.1f}% duplicates detected"
            )
            recommendations.append("Consider deduplication")
        else:
            recommendations.append("✓ Dataset appears clean of duplicates")

        return recommendations

    def generate_report(self, contamination_report: ContaminationReport) -> str:
        """Generate human-readable contamination report"""
        report = []
        report.append("=" * 70)
        report.append("ZELDA DATA QUALITY VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"Analysis Type: {contamination_report.contamination_type}")
        report.append(f"Total Samples: {contamination_report.total_samples}")
        report.append("")

        if contamination_report.is_contaminated:
            report.append(
                f"⚠️  CONTAMINATION DETECTED: "
                f"{contamination_report.contamination_percentage:.2f}%"
            )
            report.append(
                f"   Contaminated Samples: {contamination_report.contaminated_samples}"
            )
        else:
            report.append("✓  Dataset appears clean")

        if contamination_report.recommendations:
            report.append("")
            report.append("Recommendations:")
            for rec in contamination_report.recommendations:
                report.append(f"  {rec}")

        report.append("=" * 70)
        report.append("Data quality validation ensures ML model integrity")
        report.append("=" * 70)

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    print("ZELDA Data Quality Validation Framework")
    print("=" * 60)
    print("Based on: Contamination in Modern Benchmarks Research")
    print("=" * 60)

    # Initialize validator
    validator = DataQualityValidator()

    # Test Case 1: Train/Test Split Validation
    print("\nTest 1: Train/Test Split Validation")
    # Create synthetic dataset
    np.random.seed(42)
    train_data = np.random.randn(1000, 16)
    test_data = np.random.randn(200, 16)

    # Intentionally contaminate 5% of test set
    contamination_size = int(0.05 * len(test_data))
    test_data[:contamination_size] = train_data[:contamination_size]

    result = validator.validate_train_test_split(train_data, test_data)
    print(validator.generate_report(result))

    # Test Case 2: Duplicate Detection
    print("\nTest 2: Duplicate Sample Detection")
    dataset_with_dupes = np.random.randn(500, 16)
    # Add duplicates
    dataset_with_dupes[100] = dataset_with_dupes[50]
    dataset_with_dupes[200] = dataset_with_dupes[150]

    result = validator.detect_duplicate_samples(dataset_with_dupes)
    print(validator.generate_report(result))

    # Test Case 3: Class Balance Validation
    print("\nTest 3: Class Balance Validation")
    # Severely imbalanced labels
    labels = np.array([0]*900 + [1]*90 + [2]*10)  # 90:9:1 ratio
    class_names = ['Normal', 'Jamming', 'Spoofing']

    balance_result = validator.validate_class_balance(labels, class_names)
    print("\nClass Balance Analysis:")
    print(json.dumps(balance_result, indent=2))

    print("\n✓ Data quality validation framework operational")
    print("✓ Ensures ML model training integrity")
