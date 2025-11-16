"""
Comprehensive Evaluation and Benchmarking Script
Tests trained models on all three difficulty levels
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List
import seaborn as sns

from backend.datasets.zelda_loader import create_dataloaders
from backend.core.ml.advanced_detector import create_model


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(
        self,
        model_path: str,
        model_type: str = "ultra",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = create_model(model_type=model_type, input_length=4096).to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Best val accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_dir: str,
        difficulty: str,
        batch_size: int = 64,
    ) -> Dict:
        """Evaluate on a specific difficulty level"""

        logger.info(f"Evaluating on {difficulty} dataset...")

        # Create dataloader (use full dataset, no train/val split)
        _, val_loader = create_dataloaders(
            data_dir=data_dir,
            difficulty=difficulty,
            batch_size=batch_size,
            train_split=0.0001,  # Tiny train split to use almost all data for eval
            num_workers=4,
        )

        all_outputs = []
        all_labels = []
        all_filenames = []

        for iq_data, labels, metadata in tqdm(val_loader, desc=f"Evaluating {difficulty}"):
            iq_data = iq_data.to(self.device)

            outputs, strength = self.model(iq_data)
            probs = torch.sigmoid(outputs)

            all_outputs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_filenames.extend([m['file'] for m in metadata])

        # Concatenate all results
        y_pred_proba = np.concatenate(all_outputs)
        y_true = np.concatenate(all_labels)
        y_pred = (y_pred_proba > 0.5).astype(float)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        metrics = {
            'difficulty': difficulty,
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_true, y_pred_proba)),
            'num_samples': len(y_true),
            'num_positives': int(y_true.sum()),
            'num_negatives': int((1 - y_true).sum()),
        }

        logger.info(f"{difficulty.upper()} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1:        {metrics['f1']:.4f}")
        logger.info(f"  AUC:       {metrics['auc']:.4f}")

        return {
            'metrics': metrics,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'filenames': all_filenames,
            }
        }

    def evaluate_all(
        self,
        data_dir: str,
        output_dir: str = "./data/benchmark_results",
    ) -> Dict:
        """Evaluate on all difficulty levels"""

        results = {}

        for difficulty in ['easy', 'medium', 'hard']:
            try:
                result = self.evaluate_dataset(data_dir, difficulty)
                results[difficulty] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {difficulty}: {e}")
                results[difficulty] = None

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save metrics
        metrics_file = output_path / "evaluation_metrics.json"
        metrics_summary = {
            diff: res['metrics'] if res else None
            for diff, res in results.items()
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)

        logger.info(f"Saved metrics to {metrics_file}")

        # Generate visualizations
        self.plot_results(results, output_dir)

        return results

    def plot_results(self, results: Dict, output_dir: str):
        """Generate visualization plots"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 1. ROC curves for all difficulties
        plt.figure(figsize=(10, 8))

        for difficulty, result in results.items():
            if result is None:
                continue

            y_true = result['predictions']['y_true']
            y_pred_proba = result['predictions']['y_pred_proba']

            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{difficulty.capitalize()} (AUC = {roc_auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Difficulty Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROC curves to {output_path / 'roc_curves.png'}")

        # 2. Precision-Recall curves
        plt.figure(figsize=(10, 8))

        for difficulty, result in results.items():
            if result is None:
                continue

            y_true = result['predictions']['y_true']
            y_pred_proba = result['predictions']['y_pred_proba']

            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            plt.plot(recall, precision, label=f'{difficulty.capitalize()}', linewidth=2)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - All Difficulty Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved PR curves to {output_path / 'pr_curves.png'}")

        # 3. Metrics comparison bar chart
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        difficulties = [d for d in ['easy', 'medium', 'hard'] if results.get(d)]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(difficulties))
        width = 0.15

        for i, metric in enumerate(metrics_to_plot):
            values = [results[d]['metrics'][metric] for d in difficulties]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics Across Difficulty Levels')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([d.capitalize() for d in difficulties])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved metrics comparison to {output_path / 'metrics_comparison.png'}")

        # 4. Confusion matrices
        fig, axes = plt.subplots(1, len(difficulties), figsize=(15, 4))

        if len(difficulties) == 1:
            axes = [axes]

        for idx, difficulty in enumerate(difficulties):
            result = results[difficulty]
            y_true = result['predictions']['y_true']
            y_pred = result['predictions']['y_pred']

            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{difficulty.capitalize()}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')

        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrices to {output_path / 'confusion_matrices.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Zelda Signal Detector")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="ultra", choices=["ultra", "transformer", "ensemble"])
    parser.add_argument("--data-dir", type=str, default="./data/datasets")
    parser.add_argument("--output-dir", type=str, default="./data/benchmark_results")
    parser.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all"])

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        model_type=args.model_type,
    )

    if args.difficulty == "all":
        results = evaluator.evaluate_all(args.data_dir, args.output_dir)
    else:
        results = evaluator.evaluate_dataset(args.data_dir, args.difficulty)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
