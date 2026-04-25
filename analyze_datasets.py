import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_dataset(dataset_name, data_dir, output_dir):
    print(f"\nAnalyzing dataset: {dataset_name}")
    print("-" * 40)
    
    # Load data
    try:
        X = np.load(os.path.join(data_dir, 'X_train.npy'))
        y = np.load(os.path.join(data_dir, 'y_train.npy'))
    except Exception as e:
        print(f"Could not load data for {dataset_name}: {e}")
        return

    # Basic Metrics
    num_samples, num_channels, seq_len = X.shape
    classes, counts = np.unique(y, return_counts=True)
    
    print(f"Shape: {X.shape}")
    print(f"Total Samples (Train): {num_samples}")
    print(f"Number of Channels: {num_channels}")
    print(f"Sequence Length: {seq_len}")
    print(f"Number of Classes: {len(classes)}")
    
    print("\nClass Distribution:")
    for c, count in zip(classes, counts):
        percentage = (count / num_samples) * 100
        print(f"  Class {c}: {count} samples ({percentage:.2f}%)")
        
    print(f"\nData Statistics:")
    print(f"  Mean: {np.mean(X):.4f}")
    print(f"  Std Dev: {np.std(X):.4f}")
    print(f"  Min: {np.min(X):.4f}")
    print(f"  Max: {np.max(X):.4f}")

    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x=classes, y=counts, palette='viridis')
    plt.title(f"{dataset_name.upper()} - Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_class_dist.png"), dpi=300)
    plt.close()

    # 2. Sample Data Format Visualization (Plotting one sample per class)
    plt.figure(figsize=(12, 2 * len(classes)))
    for i, c in enumerate(classes):
        # Find first sample of class c
        idx = np.where(y == c)[0][0]
        sample = X[idx]  # Shape: (channels, seq_len)
        
        plt.subplot(len(classes), 1, i + 1)
        # We plot the first channel (raw time series)
        plt.plot(sample[0, :], color='blue', alpha=0.7)
        plt.title(f"Class {c} - Sample (Channel 0)")
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_samples.png"), dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    datasets = {
        'bonn': 'data/bonn',
        'mitbih': 'data/mitbih',
        'ecg': 'data/ecg'
    }
    output_dir = 'plots_datasets'
    
    for name, path in datasets.items():
        if os.path.exists(path):
            analyze_and_plot_dataset(name, path, output_dir)
        else:
            print(f"Directory {path} not found.")

if __name__ == "__main__":
    main()
