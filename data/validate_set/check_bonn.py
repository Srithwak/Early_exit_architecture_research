import numpy as np
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', 'data', 'bonn', '*.npy')
files = glob.glob(data_path)
files.sort()

output_file = os.path.join(script_dir, 'bonn_stats.txt')
with open(output_file, 'w', encoding='utf-8') as out:
    for f in files:
        try:
            arr = np.load(f)
            if 'X' in os.path.basename(f):
                out.write(f'Feature File: {f}\n')
                out.write(f'  Shape: {arr.shape}\n')
                out.write(f'  Dtype: {arr.dtype}\n')
                out.write(f'  Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f}\n')
                out.write(f'  Mean: {np.mean(arr):.4f}, Std: {np.std(arr):.4f}\n')
                out.write(f'  NaN count: {np.isnan(arr).sum()}\n')
            elif 'y' in os.path.basename(f):
                out.write(f'Label File: {f}\n')
                out.write(f'  Shape: {arr.shape}\n')
                out.write(f'  Unique classes: {np.unique(arr, return_counts=True)}\n')
        except Exception as e:
            out.write(f'Error reading {f}: {e}\n')
