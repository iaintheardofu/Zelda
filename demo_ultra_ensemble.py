"""
Demo: Ultra YOLO Ensemble System
Test the ultimate RF signal detection system combining all YOLO variants
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from backend.datasets.zelda_loader import ZeldaDataset
from backend.core.ml.advanced_detector import UltraDetector
from backend.core.ml.ultra_yolo_ensemble import create_ultra_ensemble


def demo_ensemble(
    ultra_model_path: str = "data/models/best_easy.pth",
    difficulty: str = "easy",
    num_samples: int = 50,
):
    """
    Demonstrate the Ultra YOLO Ensemble system

    Args:
        ultra_model_path: Path to trained UltraDetector
        difficulty: Dataset difficulty
        num_samples: Number of samples to test
    """

    print("=" * 70)
    print("ULTRA YOLO ENSEMBLE - RF SIGNAL DETECTION DEMO")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load UltraDetector
    print(f"\nLoading UltraDetector from {ultra_model_path}...")
    ultra_model = UltraDetector(input_length=4096)

    if Path(ultra_model_path).exists():
        checkpoint = torch.load(ultra_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            ultra_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Loaded checkpoint (Epoch {checkpoint.get('epoch', '?')}, "
                  f"Acc: {checkpoint.get('accuracy', '?'):.2f}%)")
        else:
            ultra_model.load_state_dict(checkpoint)
            print(f"  ✓ Loaded model weights")
    else:
        print(f"  ⚠ Model not found, using random initialization")

    ultra_model.to(device)
    ultra_model.eval()

    # Create ensemble
    print("\nCreating Ultra YOLO Ensemble...")
    ensemble = create_ultra_ensemble(
        ultra_model,
        fusion_method="adaptive",
        use_all_yolos=True
    )
    ensemble.to(device)
    ensemble.eval()

    print(f"\nTotal parameters: {sum(p.numel() for p in ensemble.parameters()):,}")

    # Load dataset
    print(f"\nLoading {difficulty} dataset...")
    dataset = ZeldaDataset(
        data_dir="data/datasets",
        difficulty=difficulty,
        window_size=4096,
        stride=4096  # No overlap for demo
    )

    print(f"  ✓ Dataset loaded: {len(dataset)} samples")

    # Test samples
    print(f"\nTesting {num_samples} random samples...")
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    results = {
        'ultra_confidences': [],
        'rf_yolo_confidences': [],
        'yolov11_confidences': [],
        'yolov12_confidences': [],
        'yolo_world_confidences': [],
        'rtdetr_confidences': [],
        'fused_confidences': [],
        'labels': [],
        'inference_times': [],
    }

    correct_ultra = 0
    correct_fused = 0

    with torch.no_grad():
        for idx in tqdm(indices, desc="Processing"):
            iq_tensor, label = dataset[idx]
            iq_batch = iq_tensor.unsqueeze(0).to(device)

            # Convert to numpy for spectrogram generation
            iq_numpy = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()
            iq_numpy_batch = np.array([iq_numpy])

            # Time inference
            start_time = time.time()

            # Get predictions
            individual_results = ensemble(
                iq_batch,
                return_individual=True,
                iq_numpy=iq_numpy_batch
            )

            inference_time = (time.time() - start_time) * 1000  # ms

            # Extract confidences
            ultra_conf = torch.sigmoid(individual_results['ultra_confidence']).item()
            fused_conf = torch.sigmoid(individual_results['fused_confidence']).item()

            results['ultra_confidences'].append(ultra_conf)
            results['fused_confidences'].append(fused_conf)
            results['labels'].append(label.item())
            results['inference_times'].append(inference_time)

            # Optional: Extract other model confidences if available
            for key in ['rf_yolo_confidence', 'yolov11_confidence', 'yolov12_confidence',
                        'yolo_world_confidence', 'rtdetr_confidence']:
                if key in individual_results:
                    conf = torch.sigmoid(individual_results[key]).item()
                    results[f"{key.replace('_confidence', '')}_confidences"].append(conf)

            # Accuracy
            predicted_ultra = 1 if ultra_conf > 0.5 else 0
            predicted_fused = 1 if fused_conf > 0.5 else 0

            if predicted_ultra == label.item():
                correct_ultra += 1
            if predicted_fused == label.item():
                correct_fused += 1

    # Calculate metrics
    ultra_acc = 100.0 * correct_ultra / len(indices)
    fused_acc = 100.0 * correct_fused / len(indices)
    avg_inference_time = np.mean(results['inference_times'])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy:")
    print(f"  UltraDetector Only:  {ultra_acc:.2f}%")
    print(f"  Ensemble (Fused):    {fused_acc:.2f}%")
    print(f"  Improvement:         {fused_acc - ultra_acc:+.2f}%")

    print(f"\nInference Speed:")
    print(f"  Average: {avg_inference_time:.1f}ms")
    print(f"  Min:     {min(results['inference_times']):.1f}ms")
    print(f"  Max:     {max(results['inference_times']):.1f}ms")

    # Individual model performance
    print(f"\nIndividual Model Performance:")
    for model_name in ['ultra', 'rf_yolo', 'yolov11', 'yolov12', 'yolo_world', 'rtdetr']:
        key = f"{model_name}_confidences"
        if key in results and len(results[key]) > 0:
            confs = results[key]
            avg_conf = np.mean(confs)
            print(f"  {model_name:15s}: {avg_conf:.4f} avg confidence")

    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Confidence comparison
    ax = axes[0, 0]
    sample_indices = np.arange(len(results['ultra_confidences']))
    ax.plot(sample_indices, results['ultra_confidences'], 'o-', label='UltraDetector', alpha=0.7)
    ax.plot(sample_indices, results['fused_confidences'], 's-', label='Ensemble (Fused)', alpha=0.7)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.scatter(sample_indices, results['labels'], c='black', marker='x', s=100,
               label='Ground Truth', zorder=5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Confidence')
    ax.set_title('Detection Confidence: UltraDetector vs Ensemble')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Confidence distribution
    ax = axes[0, 1]
    ax.hist(results['ultra_confidences'], bins=20, alpha=0.6, label='UltraDetector', color='blue')
    ax.hist(results['fused_confidences'], bins=20, alpha=0.6, label='Ensemble', color='orange')
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Inference time
    ax = axes[1, 0]
    ax.plot(results['inference_times'], 'g-', alpha=0.7)
    ax.axhline(y=avg_inference_time, color='r', linestyle='--',
               label=f'Average: {avg_inference_time:.1f}ms')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Speed per Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Model comparison (if multiple models available)
    ax = axes[1, 1]
    model_names = []
    model_accs = []

    # Calculate accuracy for each model
    for model_name in ['ultra', 'rf_yolo', 'yolov11', 'yolov12', 'yolo_world', 'rtdetr', 'fused']:
        key = f"{model_name}_confidences"
        if key in results and len(results[key]) > 0:
            confs = np.array(results[key])
            preds = (confs > 0.5).astype(int)
            acc = 100.0 * np.sum(preds == results['labels']) / len(results['labels'])
            model_names.append(model_name.replace('_', '\n'))
            model_accs.append(acc)

    if len(model_names) > 0:
        bars = ax.bar(model_names, model_accs, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan'][:len(model_names)])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

    plt.tight_layout()

    # Save plot
    output_path = "data/logs/ultra_ensemble_demo.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.show()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo Ultra YOLO Ensemble")
    parser.add_argument('--model', type=str, default='data/models/best_easy.pth',
                        help='Path to UltraDetector checkpoint')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to test')

    args = parser.parse_args()

    demo_ensemble(
        ultra_model_path=args.model,
        difficulty=args.difficulty,
        num_samples=args.num_samples,
    )
