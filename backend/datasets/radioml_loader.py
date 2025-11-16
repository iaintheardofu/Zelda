"""
RadioML 2018.01A Dataset Loader

Dataset: https://www.deepsig.ai/datasets/
Kaggle: https://www.kaggle.com/datasets/pinxau1000/radioml2018

24 modulation types, 26 SNR levels, 4096 frames per SNR
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import pickle

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available - RadioML loader disabled")


class RadioMLLoader:
    """
    Loader for RadioML 2018.01A dataset.

    Dataset structure:
    - 24 modulation types (BPSK, QPSK, 8PSK, 16QAM, 64QAM, etc.)
    - 26 SNR levels (-20dB to +30dB in 2dB steps)
    - 4096 samples per (modulation, SNR) pair
    - 1024 I/Q samples per signal
    """

    MODULATIONS = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
        '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
        '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
        'FM', 'GMSK', 'OQPSK'
    ]

    SNR_LEVELS = list(range(-20, 32, 2))  # -20dB to +30dB in 2dB steps

    def __init__(self, dataset_path: str):
        """
        Initialize RadioML loader.

        Args:
            dataset_path: Path to RADIOML_2018.01A.hdf5 or .dat file
        """

        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py required for RadioML. Install: pip install h5py")

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}.\n"
                f"Download from: https://www.deepsig.ai/datasets/\n"
                f"Or Kaggle: https://www.kaggle.com/datasets/pinxau1000/radioml2018"
            )

        self.data = None
        self.labels = None
        self.snrs = None

        logger.info(f"RadioML loader initialized: {dataset_path}")

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the entire dataset into memory.

        Returns:
            Tuple of (data, labels, snrs)
            - data: Complex I/Q samples, shape (N, 1024)
            - labels: Modulation type indices, shape (N,)
            - snrs: SNR values, shape (N,)
        """

        logger.info("Loading RadioML dataset (this may take a few minutes)...")

        if str(self.dataset_path).endswith('.hdf5') or str(self.dataset_path).endswith('.h5'):
            return self._load_hdf5()
        else:
            return self._load_pickle()

    def _load_hdf5(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load HDF5 format"""

        with h5py.File(self.dataset_path, 'r') as f:
            # Extract data
            X = f['X'][:]  # Shape: (N, 2, 1024) where dim 1 is [I, Q]
            Y = f['Y'][:]  # Shape: (N, 24) one-hot encoded
            Z = f['Z'][:]  # Shape: (N,) SNR values

            # Convert I/Q to complex
            data = X[:, 0, :] + 1j * X[:, 1, :]

            # Convert one-hot to indices
            labels = np.argmax(Y, axis=1)

            # SNR values
            snrs = Z

        logger.info(f"Loaded {len(data)} samples from HDF5")

        self.data = data
        self.labels = labels
        self.snrs = snrs

        return data, labels, snrs

    def _load_pickle(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load pickle format (older RadioML versions)"""

        with open(self.dataset_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')

        # Extract and reorganize
        all_data = []
        all_labels = []
        all_snrs = []

        for (mod, snr), samples in data_dict.items():
            # samples shape: (num_samples, 2, 1024)
            # Convert to complex
            iq_samples = samples[:, 0, :] + 1j * samples[:, 1, :]

            all_data.append(iq_samples)
            all_labels.extend([self.MODULATIONS.index(mod)] * len(iq_samples))
            all_snrs.extend([snr] * len(iq_samples))

        data = np.vstack(all_data)
        labels = np.array(all_labels)
        snrs = np.array(all_snrs)

        logger.info(f"Loaded {len(data)} samples from pickle")

        self.data = data
        self.labels = labels
        self.snrs = snrs

        return data, labels, snrs

    def get_samples(
        self,
        modulation: Optional[str] = None,
        snr: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get filtered samples.

        Args:
            modulation: Filter by modulation type (e.g., 'QPSK')
            snr: Filter by SNR level (e.g., 10)
            num_samples: Limit number of samples

        Returns:
            Tuple of (data, labels, snrs)
        """

        if self.data is None:
            self.load()

        # Create filter mask
        mask = np.ones(len(self.data), dtype=bool)

        if modulation:
            mod_idx = self.MODULATIONS.index(modulation)
            mask &= (self.labels == mod_idx)

        if snr is not None:
            mask &= (self.snrs == snr)

        # Apply filter
        data = self.data[mask]
        labels = self.labels[mask]
        snrs = self.snrs[mask]

        # Limit samples
        if num_samples and len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            data = data[indices]
            labels = labels[indices]
            snrs = snrs[indices]

        return data, labels, snrs

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        """
        Get train/test split.

        Args:
            test_size: Fraction for test set
            stratify: Stratify by modulation type

        Returns:
            ((X_train, y_train, snr_train), (X_test, y_test, snr_test))
        """

        if self.data is None:
            self.load()

        from sklearn.model_selection import train_test_split

        if stratify:
            stratify_labels = self.labels
        else:
            stratify_labels = None

        X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
            self.data,
            self.labels,
            self.snrs,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=42
        )

        return (X_train, y_train, snr_train), (X_test, y_test, snr_test)

    def get_snr_performance_split(self) -> dict:
        """
        Get samples organized by SNR for performance vs SNR analysis.

        Returns:
            Dictionary mapping SNR -> (data, labels)
        """

        if self.data is None:
            self.load()

        snr_splits = {}

        for snr in self.SNR_LEVELS:
            mask = self.snrs == snr
            snr_splits[snr] = (self.data[mask], self.labels[mask])

        return snr_splits

    @staticmethod
    def download_instructions():
        """Print download instructions"""

        print("""
RadioML 2018.01A Dataset Download Instructions:

Option 1: DeepSig (Official)
  1. Visit: https://www.deepsig.ai/datasets/
  2. Register for an account
  3. Download RADIOML_2018.01A.tar.gz
  4. Extract: tar -xzf RADIOML_2018.01A.tar.gz

Option 2: Kaggle (Easier)
  1. Install kaggle: pip install kaggle
  2. Setup API credentials: https://www.kaggle.com/docs/api
  3. Download: kaggle datasets download -d pinxau1000/radioml2018
  4. Unzip: unzip radioml2018.zip

Dataset Size: ~24GB
Format: HDF5 (.hdf5) or Pickle (.dat)
Samples: 2,555,904 total

Save to: ~/zelda/data/datasets/RADIOML_2018.01A/
        """)


if __name__ == "__main__":
    # Example usage
    RadioMLLoader.download_instructions()
