import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def ensure_mitbih_data_exists(data_path):
    """Creates dummy data if the target MIT-BIH files do not exist."""
    os.makedirs(data_path, exist_ok=True)
    for split in ["train", "val", "test"]:
        x_path = os.path.join(data_path, f"X_{split}.npy")
        y_path = os.path.join(data_path, f"y_{split}.npy")
        if not os.path.exists(x_path):
            print(f"Creating dummy MIT-BIH data for {split}...")
            # MIT-BIH: (N, 1, 187) shape, 5 classes
            n = 500 if split == "train" else 100
            X_dummy = np.random.randn(n, 1, 187).astype(np.float32)
            y_dummy = np.random.randint(0, 5, size=(n,))
            np.save(x_path, X_dummy)
            np.save(y_path, y_dummy)


class MITBIHDataset(Dataset):
    """
    MIT-BIH Arrhythmia Dataset.

    Data format: (N, 1, 187) — already has channel dimension.
    Classes: 5 (Normal, Supraventricular, Ventricular, Fusion, Unknown)
    Sampling rate: 125 Hz
    """
    def __init__(self, data_path, split="train", use_freq_bands=True):
        x_path = os.path.join(data_path, f"X_{split}.npy")
        y_path = os.path.join(data_path, f"y_{split}.npy")

        self.X = np.load(x_path)
        self.y = np.load(y_path).astype(np.int64)
        self.use_freq_bands = use_freq_bands
        self.fs = 125.0  # MIT-BIH sampling rate

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        # Data is already (1, 187) — ensure channel dim
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Preprocessing: Z-score Normalization (Standardization)
        x = (x - x.mean()) / (x.std() + 1e-6)

        if self.use_freq_bands:
            # 1. Compute Fast Fourier Transform (FFT)
            fft_vals = torch.abs(torch.fft.rfft(x.squeeze()))
            freqs = torch.fft.rfftfreq(x.size(-1), d=1/self.fs)

            # 2. Define standard ECG/EEG frequency bands
            #    Adapted for 125 Hz sampling rate (Nyquist = 62.5 Hz)
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta':  (13, 30),
                'gamma': (30, 62),   # Capped at Nyquist
            }

            band_powers = []
            for fmin, fmax in bands.values():
                idx_mask = torch.logical_and(freqs >= fmin, freqs <= fmax)
                # Average power in the band
                power = torch.mean(fft_vals[idx_mask]**2) if idx_mask.any() else torch.tensor(0.0)
                band_powers.append(power)

            # 3. Log transform to manage large scale variations
            band_powers = [torch.log10(p + 1e-6) for p in band_powers]

            # 4. Tile the 5 scalar values across the entire sequence length
            seq_len = x.size(-1)
            tiled_bands = torch.tensor(band_powers, dtype=torch.float32).unsqueeze(1).repeat(1, seq_len)

            # 5. Concatenate: (1, 187) + (5, 187) -> (6, 187)
            x = torch.cat([x, tiled_bands], dim=0)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def get_mitbih_dataloaders(data_dir, batch_size=64, use_freq_bands=True):
    """Get DataLoaders for the MIT-BIH dataset."""
    ensure_mitbih_data_exists(data_dir)

    train_ds = MITBIHDataset(data_dir, "train", use_freq_bands=use_freq_bands)
    val_ds = MITBIHDataset(data_dir, "val", use_freq_bands=use_freq_bands)
    test_ds = MITBIHDataset(data_dir, "test", use_freq_bands=use_freq_bands)

    # Calculate class weights for imbalance (MIT-BIH is heavily imbalanced)
    labels = train_ds.y
    class_counts = np.bincount(labels, minlength=5)
    # Avoid division by zero for missing classes
    class_counts = np.maximum(class_counts, 1)
    total = len(labels)
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, class_weights
