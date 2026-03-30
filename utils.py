import torch
import numpy as np

def inspect_data(train_dl):
    print("\n--- Preprocessed Data Attributes ---")
    X_sample, y_sample = next(iter(train_dl))
    print(f"1. Structured Organization: Batch Shape (X) is {X_sample.shape} -> (Batch Size, Channels, Sequence Length)")
    print(f"   Batch Shape (y): {y_sample.shape}")

    has_nans = torch.isnan(X_sample).any().item()
    print(f"2. Cleanliness: Contains NaNs? {'Yes' if has_nans else 'No'}")

    mean_val = X_sample.mean().item()
    std_val = X_sample.std().item()
    print(f"3. Normalization: Mean ≈ {mean_val:.4f}, Std Dev ≈ {std_val:.4f} (Z-score normalized)")

    y_all = train_dl.dataset.y
    counts = np.bincount(y_all)
    print("4. Class Imbalance (Training Set):")
    for i, count in enumerate(counts):
        print(f"   Class {i}: {count} samples ({count/len(y_all)*100:.1f}%)")

    print("5. Reduced Complexity: Sequences are preprocessed and formatted specifically for Conv1D.")
    print("------------------------------------\n")
