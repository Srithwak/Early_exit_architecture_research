import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def ensure_data_exists(data_path):
    """Creates dummy data if the target files do not exist."""
    os.makedirs(data_path, exist_ok=True)
    for split in ["train", "val", "test"]:
        x_path = os.path.join(data_path, f"X_{split}.npy")
        y_path = os.path.join(data_path, f"y_{split}.npy")
        if not os.path.exists(x_path):
            print(f"Creating dummy data for {split}...")
            X_dummy = np.random.randn(100, 4097) # 100 samples, 4097 sequence length
            y_dummy = np.random.randint(0, 2, size=(100,))
            np.save(x_path, X_dummy)
            np.save(y_path, y_dummy)

class BonnDataset(Dataset):
    def __init__(self, data_path, split="train", use_freq_bands=True):
        x_path = os.path.join(data_path, f"X_{split}.npy")
        y_path = os.path.join(data_path, f"y_{split}.npy")

        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.use_freq_bands = use_freq_bands
        self.fs = 173.61 # Bonn dataset sampling rate

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        # Ensure sequence has a channel dimension for Conv1D: (1, Sequence_Length)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Preprocessing: Z-score Normalization (Standardization)
        x = (x - x.mean()) / (x.std() + 1e-6)

        if self.use_freq_bands:
            # 1. Compute Fast Fourier Transform (FFT)
            fft_vals = torch.abs(torch.fft.rfft(x.squeeze()))
            freqs = torch.fft.rfftfreq(x.size(-1), d=1/self.fs)

            # 2. Define standard EEG frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
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
            # to serve as global frequency context channels for the Conv1D
            seq_len = x.size(-1)
            tiled_bands = torch.tensor(band_powers, dtype=torch.float32).unsqueeze(1).repeat(1, seq_len)

            # 5. Concatenate: (1, 4097) + (5, 4097) -> (6, 4097)
            x = torch.cat([x, tiled_bands], dim=0)

        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def get_dataloaders(data_dir, batch_size=32, use_freq_bands=True):
    ensure_data_exists(data_dir)

    train_ds = BonnDataset(data_dir, "train", use_freq_bands=use_freq_bands)
    val_ds = BonnDataset(data_dir, "val", use_freq_bands=use_freq_bands)
    test_ds = BonnDataset(data_dir, "test", use_freq_bands=use_freq_bands)

    # Calculate class weights for imbalance
    labels = train_ds.y
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = total / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, class_weights
