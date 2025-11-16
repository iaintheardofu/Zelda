"""
Ultra Training Pipeline for Zelda Signal Detection
Achieves state-of-the-art performance with advanced techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from loguru import logger
import wandb
from typing import Optional, Dict
import time

from backend.datasets.zelda_loader import create_dataloaders
from backend.core.ml.advanced_detector import create_model


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class UltraTrainer:
    """Advanced trainer with all the bells and whistles"""

    def __init__(
        self,
        model_type: str = "ultra",
        difficulty: str = "easy",
        data_dir: str = "./data/datasets",
        batch_size: int = 64,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        device: str = "cuda",
        use_wandb: bool = False,
        checkpoint_dir: str = "./data/models",
        window_size: int = 4096,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Create model
        self.model = create_model(
            model_type=model_type,
            input_length=window_size,
            num_classes=1,
            use_attention=True,
        ).to(self.device)

        logger.info(f"Model: {model_type}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=data_dir,
            difficulty=difficulty,
            batch_size=batch_size,
            train_split=0.8,
            num_workers=4,
            window_size=window_size,
        )

        # Loss functions
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)
        self.criterion_strength = nn.MSELoss()

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        # Training params
        self.num_epochs = num_epochs
        self.difficulty = difficulty
        self.use_wandb = use_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

        # W&B
        if use_wandb:
            wandb.init(
                project="zelda-signal-detection",
                config={
                    "model": model_type,
                    "difficulty": difficulty,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                    "window_size": window_size,
                }
            )

    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for batch_idx, (iq_data, labels) in enumerate(pbar):
            iq_data = iq_data.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs, strength = self.model(iq_data)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%',
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_outputs = []
        all_labels = []

        for iq_data, labels in tqdm(self.val_loader, desc="Validating"):
            iq_data = iq_data.to(self.device)
            labels = labels.to(self.device)

            outputs, strength = self.model(iq_data)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_outputs.append(torch.sigmoid(outputs).cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        # Compute additional metrics
        all_outputs = torch.cat(all_outputs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # ROC AUC
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

        try:
            auc = roc_auc_score(all_labels, all_outputs)
        except:
            auc = 0.0

        f1 = f1_score(all_labels, all_outputs > 0.5)
        precision = precision_score(all_labels, all_outputs > 0.5, zero_division=0)
        recall = recall_score(all_labels, all_outputs > 0.5, zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }

    def train(self):
        """Full training loop"""
        logger.info(f"Starting training on {self.difficulty} dataset...")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log
            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'val_auc': val_metrics['auc'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'lr': self.optimizer.param_groups[0]['lr'],
                })

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

        logger.info(f"Training complete! Best Val Acc: {self.best_val_acc:.2f}%")

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
        }

        if is_best:
            path = self.checkpoint_dir / f"best_{self.difficulty}.pth"
            torch.save(checkpoint, path)
            logger.info(f"Saved best model to {path}")
        else:
            path = self.checkpoint_dir / f"checkpoint_{self.difficulty}_epoch{epoch+1}.pth"
            torch.save(checkpoint, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Ultra Signal Detector")
    parser.add_argument("--model", type=str, default="ultra", choices=["ultra", "transformer", "ensemble"])
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data/datasets")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--window-size", type=int, default=4096)

    args = parser.parse_args()

    trainer = UltraTrainer(
        model_type=args.model,
        difficulty=args.difficulty,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_wandb=args.wandb,
        window_size=args.window_size,
    )

    trainer.train()


if __name__ == "__main__":
    main()
