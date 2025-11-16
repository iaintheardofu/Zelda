"""
Deep Learning Signal Classifier for RF signals

Classifies modulation types and signal characteristics using CNNs.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - ML signal classification disabled")


class ModulationType(Enum):
    """Common modulation types"""
    UNKNOWN = 0
    AM = 1
    FM = 2
    PM = 3
    FSK = 4
    PSK = 5
    BPSK = 6
    QPSK = 7
    PSK8 = 8
    QAM16 = 9
    QAM64 = 10
    OFDM = 11
    GMSK = 12
    NOISE = 13


@dataclass
class ClassificationResult:
    """Result of signal classification"""
    modulation: ModulationType
    confidence: float
    probabilities: Dict[ModulationType, float]


class IQResNet(nn.Module):
    """
    Residual Network for I/Q signal classification.

    Based on ResNet architecture, adapted for complex-valued I/Q data.
    Input shape: (batch, 2, signal_length) where dim 1 is [I, Q]
    """

    def __init__(self, signal_length: int = 1024, num_classes: int = 14):
        super(IQResNet, self).__init__()

        self.signal_length = signal_length

        # Initial convolution
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling and FC
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a residual layer"""
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    """Basic residual block for 1D conv"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SignalClassifier:
    """
    High-level interface for signal classification.

    Handles:
    - Model loading/initialization
    - Preprocessing of I/Q data
    - Classification and confidence estimation
    - Model training (optional)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        signal_length: int = 1024,
        device: Optional[str] = None,
    ):
        """
        Initialize signal classifier.

        Args:
            model_path: Path to pretrained model weights (None for random init)
            signal_length: Expected signal length in samples
            device: 'cpu', 'cuda', or None (auto-detect)
        """

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.signal_length = signal_length

        # Create model
        self.model = IQResNet(signal_length=signal_length, num_classes=len(ModulationType))
        self.model = self.model.to(self.device)

        # Load weights if provided
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.warning("Using randomly initialized model")

        self.model.eval()

        logger.info(f"SignalClassifier initialized on {self.device}")

    def preprocess(self, iq_data: np.ndarray) -> torch.Tensor:
        """
        Preprocess I/Q data for classification.

        Args:
            iq_data: Complex I/Q samples (1D numpy array)

        Returns:
            Preprocessed tensor of shape (1, 2, signal_length)
        """

        # Ensure correct length
        if len(iq_data) > self.signal_length:
            # Take center portion
            start = (len(iq_data) - self.signal_length) // 2
            iq_data = iq_data[start:start + self.signal_length]
        elif len(iq_data) < self.signal_length:
            # Zero-pad
            padding = self.signal_length - len(iq_data)
            iq_data = np.pad(iq_data, (padding // 2, padding - padding // 2))

        # Split into I and Q
        i_data = np.real(iq_data)
        q_data = np.imag(iq_data)

        # Normalize
        i_data = i_data / (np.max(np.abs(i_data)) + 1e-10)
        q_data = q_data / (np.max(np.abs(q_data)) + 1e-10)

        # Stack and create tensor
        iq_tensor = np.stack([i_data, q_data])
        iq_tensor = torch.from_numpy(iq_tensor).float()
        iq_tensor = iq_tensor.unsqueeze(0)  # Add batch dimension

        return iq_tensor

    def classify(self, iq_data: np.ndarray) -> ClassificationResult:
        """
        Classify a signal.

        Args:
            iq_data: Complex I/Q samples

        Returns:
            ClassificationResult with modulation type and confidence
        """

        # Preprocess
        x = self.preprocess(iq_data).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probabilities = F.softmax(logits, dim=1)

        # Get prediction
        probs = probabilities.cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        confidence = float(probs[predicted_idx])

        # Map to ModulationType
        modulation = list(ModulationType)[predicted_idx]

        # Create probability dict
        prob_dict = {
            mod: float(probs[i])
            for i, mod in enumerate(ModulationType)
        }

        result = ClassificationResult(
            modulation=modulation,
            confidence=confidence,
            probabilities=prob_dict
        )

        logger.debug(f"Classification: {modulation.name} (confidence={confidence:.3f})")

        return result

    def batch_classify(self, iq_batch: list) -> list:
        """
        Classify multiple signals in a batch.

        Args:
            iq_batch: List of I/Q signal arrays

        Returns:
            List of ClassificationResult objects
        """

        # Preprocess all signals
        tensors = [self.preprocess(iq) for iq in iq_batch]
        batch_tensor = torch.cat(tensors, dim=0).to(self.device)

        # Batch inference
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probabilities = F.softmax(logits, dim=1)

        # Parse results
        results = []
        for i in range(len(iq_batch)):
            probs = probabilities[i].cpu().numpy()
            predicted_idx = np.argmax(probs)
            confidence = float(probs[predicted_idx])
            modulation = list(ModulationType)[predicted_idx]

            prob_dict = {
                mod: float(probs[j])
                for j, mod in enumerate(ModulationType)
            }

            results.append(ClassificationResult(
                modulation=modulation,
                confidence=confidence,
                probabilities=prob_dict
            ))

        return results

    def is_valid_signal(
        self,
        iq_data: np.ndarray,
        min_confidence: float = 0.5
    ) -> bool:
        """
        Check if signal is valid (not noise/interference).

        Args:
            iq_data: I/Q samples
            min_confidence: Minimum confidence threshold

        Returns:
            True if signal is valid
        """

        result = self.classify(iq_data)

        # Filter out noise
        if result.modulation == ModulationType.NOISE:
            return False

        # Check confidence
        if result.confidence < min_confidence:
            return False

        return True


def create_simple_cnn(signal_length: int = 1024, num_classes: int = 14):
    """
    Create a simpler CNN for signal classification (faster training/inference).

    Args:
        signal_length: Input signal length
        num_classes: Number of modulation classes

    Returns:
        Simple CNN model
    """

    if not TORCH_AVAILABLE:
        return None

    model = nn.Sequential(
        # Conv block 1
        nn.Conv1d(2, 64, kernel_size=7, padding=3),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2),

        # Conv block 2
        nn.Conv1d(64, 128, kernel_size=5, padding=2),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2),

        # Conv block 3
        nn.Conv1d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(2),

        # Global pooling and classification
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

    return model
