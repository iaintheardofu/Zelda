"""
Train the Ultimate YOLO Ensemble System
Combines all state-of-the-art YOLO models for maximum RF signal detection performance

Training Strategy:
1. Load pre-trained UltraDetector
2. Generate spectrogram dataset
3. Train RF-YOLO and other YOLO variants on spectrograms
4. Fine-tune ensemble fusion weights
5. Achieve 97-98%+ accuracy target
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from backend.datasets.zelda_loader import ZeldaDataset
from backend.core.ml.advanced_detector import UltraDetector
from backend.core.ml.ultra_yolo_ensemble import UltraYOLOEnsemble, create_ultra_ensemble
from backend.core.ml.yolo_detector import SpectrogramGenerator


class SpectrogramDataset(Dataset):
    """
    Wrapper dataset that generates spectrograms on-the-fly
    from I/Q samples for YOLO training
    """

    def __init__(self, zelda_dataset, spec_generator):
        self.zelda_dataset = zelda_dataset
        self.spec_gen = spec_generator

    def __len__(self):
        return len(self.zelda_dataset)

    def __getitem__(self, idx):
        iq_tensor, label = self.zelda_dataset[idx]

        # Convert tensor to numpy complex
        iq_numpy = iq_tensor[0].numpy() + 1j * iq_tensor[1].numpy()

        # Generate spectrogram
        try:
            spectrogram = self.spec_gen.generate(iq_numpy)
            spec_tensor = torch.from_numpy(spectrogram).permute(2, 0, 1).float()
        except Exception as e:
            # Fallback to zeros if generation fails
            spec_tensor = torch.zeros(3, 640, 640)

        return iq_tensor, spec_tensor, label


class EnsembleTrainer:
    """
    Train the Ultra YOLO Ensemble
    """

    def __init__(
        self,
        ultra_model_path: str,
        difficulty: str = "easy",
        batch_size: int = 16,  # Smaller for spectrogram processing
        num_epochs: int = 10,
        lr: float = 1e-4,
        fusion_method: str = "learned",
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {self.device}")

        # Load pre-trained UltraDetector
        print(f"Loading UltraDetector from {ultra_model_path}")
        self.ultra_model = UltraDetector(input_length=4096)
        checkpoint = torch.load(ultra_model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.ultra_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.ultra_model.load_state_dict(checkpoint)

        self.ultra_model.to(self.device)
        self.ultra_model.eval()  # Freeze UltraDetector initially

        print(f"✓ UltraDetector loaded (frozen)")

        # Create ensemble
        self.ensemble = create_ultra_ensemble(
            self.ultra_model,
            fusion_method=fusion_method,
            use_all_yolos=True
        )
        self.ensemble.to(self.device)

        # Setup datasets
        print(f"\nLoading {difficulty} dataset...")
        self.spec_gen = SpectrogramGenerator()

        zelda_train = ZeldaDataset(
            data_dir="data/datasets",
            difficulty=difficulty,
            window_size=4096,
            stride=2048
        )

        # Limit dataset for faster training initially
        # In production: use full dataset
        train_size = min(10000, len(zelda_train))
        indices = np.random.choice(len(zelda_train), train_size, replace=False)
        zelda_train = torch.utils.data.Subset(zelda_train, indices)

        self.train_dataset = SpectrogramDataset(zelda_train, self.spec_gen)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        print(f"✓ Dataset loaded: {len(self.train_dataset)} samples")

        # Training configuration
        self.num_epochs = num_epochs
        self.criterion = nn.BCEWithLogitsLoss()

        # Only optimize YOLO models and fusion weights, keep Ultra frozen
        trainable_params = []
        for name, param in self.ensemble.named_parameters():
            if 'ultra_detector' not in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )

        print(f"✓ Optimizer configured ({len(trainable_params)} param groups)")

        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'individual_accs': {},
        }

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.ensemble.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, (iq_data, spec_data, labels) in enumerate(pbar):
            iq_data = iq_data.to(self.device)
            spec_data = spec_data.to(self.device)
            labels = labels.to(self.device).float()

            # Convert I/Q tensor to numpy for spectrogram generation
            iq_numpy = []
            for i in range(iq_data.shape[0]):
                iq_np = iq_data[i, 0].cpu().numpy() + 1j * iq_data[i, 1].cpu().numpy()
                iq_numpy.append(iq_np)
            iq_numpy = np.array(iq_numpy)

            # Forward pass
            self.optimizer.zero_grad()

            results = self.ensemble(iq_data, return_individual=True, iq_numpy=iq_numpy)

            fused_output = results['fused_confidence']

            # Loss on fused prediction
            loss = self.criterion(fused_output, labels.unsqueeze(1))

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            running_loss += loss.item()
            predicted = (torch.sigmoid(fused_output) > 0.5).float()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total,
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self):
        """Evaluate individual model contributions"""
        self.ensemble.eval()

        individual_correct = {}
        total = 0

        with torch.no_grad():
            for iq_data, spec_data, labels in tqdm(self.train_loader, desc="Evaluating"):
                iq_data = iq_data.to(self.device)
                spec_data = spec_data.to(self.device)
                labels = labels.to(self.device).float()

                # Convert to numpy
                iq_numpy = []
                for i in range(iq_data.shape[0]):
                    iq_np = iq_data[i, 0].cpu().numpy() + 1j * iq_data[i, 1].cpu().numpy()
                    iq_numpy.append(iq_np)
                iq_numpy = np.array(iq_numpy)

                results = self.ensemble(iq_data, return_individual=True, iq_numpy=iq_numpy)

                # Check each model
                for key, value in results.items():
                    if 'confidence' in key and isinstance(value, torch.Tensor):
                        predicted = (torch.sigmoid(value) > 0.5).float()
                        correct = (predicted == labels.unsqueeze(1)).sum().item()

                        if key not in individual_correct:
                            individual_correct[key] = 0
                        individual_correct[key] += correct

                total += labels.size(0)

        # Calculate accuracies
        individual_accs = {}
        for key, correct in individual_correct.items():
            individual_accs[key] = 100.0 * correct / total

        return individual_accs

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 60)
        print("STARTING ENSEMBLE TRAINING")
        print("=" * 60)

        best_acc = 0.0

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Accuracy: {train_acc:.2f}%")

            # Evaluate individual models every 5 epochs
            if (epoch + 1) % 5 == 0:
                print("\n  Individual Model Performance:")
                individual_accs = self.evaluate()

                for key, acc in individual_accs.items():
                    print(f"    {key}: {acc:.2f}%")

                self.history['individual_accs'][f'epoch_{epoch+1}'] = individual_accs

            # Save best model
            if train_acc > best_acc:
                best_acc = train_acc
                self.save_checkpoint('data/models/ultra_ensemble_best.pth', epoch, train_acc)
                print(f"  ✓ New best accuracy: {best_acc:.2f}%")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Accuracy: {best_acc:.2f}%")

        # Final evaluation
        print("\nFinal Individual Model Performance:")
        final_accs = self.evaluate()
        for key, acc in final_accs.items():
            print(f"  {key}: {acc:.2f}%")

        return self.history

    def save_checkpoint(self, path: str, epoch: int, accuracy: float):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.ensemble.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'history': self.history,
        }

        torch.save(checkpoint, path)
        print(f"  ✓ Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Ultra YOLO Ensemble")
    parser.add_argument('--ultra-model', type=str, required=True,
                        help='Path to pre-trained UltraDetector')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fusion-method', type=str, default='learned',
                        choices=['average', 'weighted', 'learned', 'adaptive'])

    args = parser.parse_args()

    # Create trainer
    trainer = EnsembleTrainer(
        ultra_model_path=args.ultra_model,
        difficulty=args.difficulty,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        fusion_method=args.fusion_method,
    )

    # Train
    history = trainer.train()

    # Save history
    history_path = f"data/logs/ensemble_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to floats for JSON
    json_history = {
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'individual_accs': history['individual_accs'],
    }

    with open(history_path, 'w') as f:
        json.dump(json_history, f, indent=2)

    print(f"\n✓ Training history saved: {history_path}")


if __name__ == "__main__":
    main()
