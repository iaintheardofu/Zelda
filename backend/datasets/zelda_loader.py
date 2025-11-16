"""
Zelda Dataset Loader for Signal Detection
Loads .bin (I/Q samples) and .json (detection metadata) files
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from loguru import logger
import struct


class ZeldaDataset(Dataset):
    """Dataset for Zelda signal detection challenge"""

    def __init__(
        self,
        data_dir: str,
        difficulty: str = "easy",
        transform=None,
        window_size: int = 4096,
        stride: int = 2048,
    ):
        """
        Initialize Zelda dataset

        Args:
            data_dir: Path to data/datasets directory
            difficulty: 'easy', 'medium', or 'hard'
            transform: Optional transform to apply
            window_size: Size of I/Q windows to extract
            stride: Stride for sliding window
        """
        self.data_dir = Path(data_dir) / f"{difficulty}_final"
        self.transform = transform
        self.window_size = window_size
        self.stride = stride

        if not self.data_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.data_dir}")

        # Find all JSON files
        self.json_files = sorted(list(self.data_dir.glob("*.json")))
        logger.info(f"Found {len(self.json_files)} files in {difficulty} dataset")

        # Build index of all samples
        self.samples = self._build_sample_index()
        logger.info(f"Created {len(self.samples)} windowed samples")

    def _build_sample_index(self) -> List[Tuple[Path, Path, int, Dict]]:
        """Build index of all windowed samples"""
        samples = []

        for json_path in self.json_files:
            # Get corresponding .bin file
            bin_path = json_path.with_suffix('.bin')
            if not bin_path.exists():
                logger.warning(f"Missing .bin file for {json_path.name}")
                continue

            # Load metadata
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # Get bin file size to determine number of samples
            bin_size = bin_path.stat().st_size
            # I/Q samples are complex64 (8 bytes each: 4 for I, 4 for Q)
            n_samples = bin_size // 8

            # Create windowed samples with stride
            for start_idx in range(0, n_samples - self.window_size, self.stride):
                samples.append((bin_path, json_path, start_idx, metadata))

        return samples

    def _load_iq_window(self, bin_path: Path, start_idx: int) -> np.ndarray:
        """Load a window of I/Q samples from binary file"""
        # Each sample is 2 float32 values (I and Q)
        with open(bin_path, 'rb') as f:
            # Seek to start position (8 bytes per complex sample)
            f.seek(start_idx * 8)

            # Read window_size complex samples
            data = f.read(self.window_size * 8)

            # Parse as float32 pairs
            iq_values = struct.unpack(f'{self.window_size * 2}f', data)

            # Convert to complex numpy array
            i_vals = np.array(iq_values[0::2], dtype=np.float32)
            q_vals = np.array(iq_values[1::2], dtype=np.float32)
            iq_samples = i_vals + 1j * q_vals

        return iq_samples

    def _create_labels(self, metadata: Dict, start_idx: int) -> np.ndarray:
        """
        Create labels for this window based on detections

        Returns binary label: 1 if signal present, 0 otherwise
        """
        detections = metadata.get('detections', [])
        fs = metadata.get('fs', 40e6)  # Sample rate

        # Convert start_idx to time
        window_start_time = start_idx / fs
        window_end_time = (start_idx + self.window_size) / fs

        # Check if any detection overlaps with this window
        for detection in detections:
            det_start_time, det_duration, det_freq, det_offset = detection
            det_end_time = det_start_time + det_duration

            # Check for overlap
            if (det_start_time < window_end_time and det_end_time > window_start_time):
                return np.array([1.0], dtype=np.float32)  # Signal present

        return np.array([0.0], dtype=np.float32)  # No signal

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bin_path, json_path, start_idx, metadata = self.samples[idx]

        # Load I/Q data
        iq_data = self._load_iq_window(bin_path, start_idx)

        # Create labels
        label = self._create_labels(metadata, start_idx)

        # Convert to tensor format (2 channels: I and Q)
        i_channel = np.real(iq_data)
        q_channel = np.imag(iq_data)

        # Normalize
        max_val = max(np.abs(i_channel).max(), np.abs(q_channel).max()) + 1e-10
        i_channel = i_channel / max_val
        q_channel = q_channel / max_val

        # Stack as (2, window_size)
        iq_tensor = np.stack([i_channel, q_channel], axis=0)
        iq_tensor = torch.from_numpy(iq_tensor).float()
        label_tensor = torch.from_numpy(label).float()

        if self.transform:
            iq_tensor = self.transform(iq_tensor)

        # Return simple tuple to avoid collation issues
        return iq_tensor, label_tensor


class MultiTaskZeldaDataset(ZeldaDataset):
    """Extended dataset with multiple detection tasks"""

    def _create_labels(self, metadata: Dict, start_idx: int) -> Dict[str, torch.Tensor]:
        """Create multi-task labels"""
        detections = metadata.get('detections', [])
        fs = metadata.get('fs', 40e6)
        fc_ref = metadata.get('fc_ref', 9.5e9)

        window_start_time = start_idx / fs
        window_end_time = (start_idx + self.window_size) / fs

        # Task 1: Binary detection (signal present/absent)
        signal_present = 0.0

        # Task 2: Count number of signals
        signal_count = 0

        # Task 3: Frequency estimation (normalized)
        frequencies = []

        for detection in detections:
            det_start_time, det_duration, det_freq, det_offset = detection
            det_end_time = det_start_time + det_duration

            if (det_start_time < window_end_time and det_end_time > window_start_time):
                signal_present = 1.0
                signal_count += 1
                # Normalize frequency offset
                freq_normalized = det_offset / (fs / 2)  # Normalize to [-1, 1]
                frequencies.append(freq_normalized)

        return {
            'binary': torch.tensor([signal_present], dtype=torch.float32),
            'count': torch.tensor([min(signal_count, 5)], dtype=torch.float32),  # Cap at 5
            'frequency': torch.tensor(frequencies[:5] if frequencies else [0.0] * 5, dtype=torch.float32),
        }


def create_dataloaders(
    data_dir: str,
    difficulty: str = "easy",
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 4,
    window_size: int = 4096,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    dataset = ZeldaDataset(
        data_dir=data_dir,
        difficulty=difficulty,
        window_size=window_size,
    )

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Created dataloaders: train={train_size}, val={val_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "/home/iaintheardofu/Downloads/zelda/zelda/data/datasets"

    dataset = ZeldaDataset(data_dir, difficulty="easy", window_size=4096)
    print(f"Dataset size: {len(dataset)}")

    # Test loading a sample
    iq, label, meta = dataset[0]
    print(f"I/Q shape: {iq.shape}")
    print(f"Label: {label}")
    print(f"Metadata: {meta['file']}")
