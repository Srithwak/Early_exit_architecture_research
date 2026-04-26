"""
Build a self-contained Google Colab notebook for the ECG experiments.

Embeds the ECG dataset (base64-encoded zip) and all source code
so the notebook can run end-to-end on Colab with zero external dependencies.
"""
import json
import os
import shutil
import base64

def create_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")] if isinstance(source, str) else source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def strip_local_imports(content):
    local_modules = [
        'dataset', 'dataset_ecg', 'models', 'models_mitbih',
        'train', 'evaluate', 'analysis', 'visualize',
        'statistical_tests', 'tuning'
    ]
    cleaned_lines = []
    in_multiline_import = False
    for line in content.split('\n'):
        if in_multiline_import:
            if ')' in line:
                in_multiline_import = False
            continue

        is_local_import = any(
            line.strip().startswith(f'from {mod} import') or
            line.strip().startswith(f'import {mod}')
            for mod in local_modules
        )
        if is_local_import:
            if '(' in line and ')' not in line:
                in_multiline_import = True
            continue

        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


cells = []

# ─── Title ───
cells.append(create_cell("markdown",
    "# Early-Exit Neural Network Architectures — ECG\n"
    "Energy-Efficient Arrhythmia Detection from ECG Signals\n\n"
    "This notebook runs the full early-exit architecture ablation study on the\n"
    "**ECG Arrhythmia** dataset (5-class, 187-length heartbeats, 125 Hz)."
))

# ─── Setup: embed data ───
cells.append(create_cell("markdown",
    "## 0. Setup & Data\n"
    "The ECG dataset is embedded directly in this notebook (base64-encoded zip).\n"
    "Just run this cell — no Google Drive mount needed."
))

print("Packaging ECG data into zip...")
shutil.make_archive('ecg_data', 'zip', 'data/ecg')
with open('ecg_data.zip', 'rb') as f:
    encoded_data = base64.b64encode(f.read()).decode('utf-8')
print(f"  Encoded size: {len(encoded_data) / 1024 / 1024:.1f} MB")

colab_setup = f"""import os
import base64
import zipfile

# 1. Decode and extract the embedded dataset
print("Extracting embedded ECG dataset...")
os.makedirs('data', exist_ok=True)
with open('data/ecg_data.zip', 'wb') as f:
    f.write(base64.b64decode("{encoded_data}"))

with zipfile.ZipFile('data/ecg_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/ecg')

# 2. Set environment paths
DATA_DIR = './data/ecg'
PLOTS_DIR = './plots_ecg'
RESULTS_DIR = './results_ecg'

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete! Data is ready in ./data/ecg")

# Verify
import numpy as np
for split in ['train', 'val', 'test']:
    X = np.load(f'{{DATA_DIR}}/X_{{split}}.npy')
    y = np.load(f'{{DATA_DIR}}/y_{{split}}.npy')
    print(f"  {{split}}: X={{X.shape}}, y={{y.shape}}, classes={{np.unique(y)}}")
"""
cells.append(create_cell("code", colab_setup))

# ─── Source code cells ───
# We include both the original shared modules AND the ECG-specific ones.
# The ECG models use the shared building blocks (PolicyHead, ClassifierHead, etc.)
# from models.py, so we must include models.py first.

files_to_add = [
    ("1. ECG Dataset & Preprocessing", "dataset_ecg.py"),
    ("2. Base Models (Architectures & Losses)", "models.py"),
    ("3. ECG Models (Short-Sequence Architectures)", "models_mitbih.py"),
    ("4. Training Logic", "train.py"),
    ("5. Evaluation Logic", "evaluate.py"),
    ("6. Analysis & Exit Behavior", "analysis.py"),
    ("7. Visualization", "visualize.py"),
    ("8. Statistical Tests", "statistical_tests.py"),
]

for title, filename in files_to_add:
    cells.append(create_cell("markdown", f"## {title} (`{filename}`)"))
    content = read_file(filename)
    cleaned_content = strip_local_imports(content)
    cells.append(create_cell("code", cleaned_content))

# ─── Main pipeline cell ───
cells.append(create_cell("markdown",
    "## 9. Main Pipeline Execution (ECG)\n"
    "This cell runs the full architecture ablation study with 3 trials per model."
))

main_content = read_file("main_ecg.py")
main_content = strip_local_imports(main_content)

# Strip the __main__ block
cleaned_main = []
for line in main_content.split('\n'):
    if line.startswith('if __name__ == "__main__":'):
        break
    cleaned_main.append(line)

main_execution = """# ─── Configuration ───
cfg = MITBIHExperimentConfig(
    base_seed=42,
    num_trials=1,
    threshold_strategy="confidence",
    warmup_epochs=1,    # Full epochs — Colab GPU/TPU is fast enough
    joint_epochs=1,
)

# ─── Initialize Pipeline ───
pipeline = MITBIHResearchPipeline(cfg)
inspect_data(pipeline.train_dl)

# ─── Run Architecture Ablation ───
print("\\n" + "="*60)
print("ARCHITECTURE ABLATION STUDY — ECG")
print("="*60)
df_ablation = pipeline.run_architecture_ablation()
print("\\n" + df_ablation.to_string())

# ─── Generate Visualizations & Save Results ───
pipeline.generate_visualizations()
pipeline.save_results()
print("\\n[DONE] ECG ablation complete.")
"""
cells.append(create_cell("code", "\n".join(cleaned_main) + "\n\n" + main_execution))

# ─── Optional: additional experiments ───
cells.append(create_cell("markdown",
    "## 10. Additional Experiments (Optional)\n"
    "Uncomment and run any of these cells for further analysis."
))

optional_experiments = """# ─── Threshold Strategy Comparison ───
# print("\\n" + "="*60)
# print("THRESHOLD STRATEGY COMPARISON — ECG")
# print("="*60)
# pipeline.run_threshold_strategy_comparison()

# ─── Model Size Scaling ───
# print("\\n" + "="*60)
# print("MODEL SIZE SCALING — ECG")
# print("="*60)
# df_sizes = pipeline.run_model_size_experiment()
# print("\\n" + df_sizes.to_string())

# ─── Structured Pruning ───
# print("\\n" + "="*60)
# print("STRUCTURED PRUNING — ECG")
# print("="*60)
# df_pruning = pipeline.run_pruning_experiment()
# print("\\n" + df_pruning.to_string())

# ─── Re-save if additional experiments were run ───
# pipeline.generate_visualizations()
# pipeline.save_results()
# print("\\n[DONE] All additional ECG experiments complete.")
"""
cells.append(create_cell("code", optional_experiments))

# ─── Build notebook ───
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 4,
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "ecg_test_nb.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"\nCreated {output_path} successfully.")
print(f"Upload this file to Google Colab and run all cells.")
