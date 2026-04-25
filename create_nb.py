import json
import os

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
    local_modules = ['dataset', 'models', 'train', 'evaluate', 'analysis', 'visualize', 'statistical_tests', 'tuning']
    cleaned_lines = []
    in_multiline_import = False
    for line in content.split('\n'):
        if in_multiline_import:
            if ')' in line:
                in_multiline_import = False
            continue
            
        is_local_import = any(line.strip().startswith(f'from {mod} import') for mod in local_modules)
        if is_local_import:
            if '(' in line and ')' not in line:
                in_multiline_import = True
            continue
            
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

cells = []

# Title and setup
cells.append(create_cell("markdown", "# Early-Exit Neural Network Architectures\nEnergy-Efficient Seizure Detection from EEG Signals"))

# Setup for Google Colab
cells.append(create_cell("markdown", "## 0. Setup & Data Path\nIn Google Colab, mount your drive and set `DATA_DIR` to the path containing the Bonn dataset."))
import shutil
import base64

# Package the data dynamically so it's embedded in the notebook
shutil.make_archive('bonn_data', 'zip', 'data/bonn')
with open('bonn_data.zip', 'rb') as f:
    encoded_data = base64.b64encode(f.read()).decode('utf-8')

colab_setup = f"""import os
import base64
import zipfile

# 1. Decode and extract the embedded dataset
print("Extracting embedded dataset...")
os.makedirs('data', exist_ok=True)
with open('data/bonn_data.zip', 'wb') as f:
    f.write(base64.b64decode("{encoded_data}"))

with zipfile.ZipFile('data/bonn_data.zip', 'r') as zip_ref:
    zip_ref.extractall('data/bonn')

# 2. Set environment paths
DATA_DIR = './data/bonn'
PLOTS_DIR = './plots'
RESULTS_DIR = './results'

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
print("Setup complete! Data is ready in ./data/bonn")
"""
cells.append(create_cell("code", colab_setup))

files_to_add = [
    ("1. Dataset & Preprocessing", "dataset.py"),
    ("2. Models (Architectures & Losses)", "models.py"),
    ("3. Training Logic", "train.py"),
    ("4. Evaluation Logic", "evaluate.py"),
    ("5. Analysis & Exit Behavior", "analysis.py"),
    ("6. Visualization", "visualize.py"),
    ("7. Statistical Tests", "statistical_tests.py"),
    ("8. Hyperparameter Tuning", "tuning.py"),
]

for title, filename in files_to_add:
    cells.append(create_cell("markdown", f"## {title} (`{filename}`)"))
    # Read file and remove any local imports since everything will be in one notebook
    content = read_file(filename)
    cleaned_content = strip_local_imports(content)
    cells.append(create_cell("code", cleaned_content))

# Main Execution cell
cells.append(create_cell("markdown", "## 9. Main Pipeline Execution (`main.py`)"))
main_content = read_file("main.py")
main_content = strip_local_imports(main_content)

cleaned_main = []
for line in main_content.split('\n'):
    if line.startswith('if __name__ == "__main__":'):
        break
    cleaned_main.append(line)

main_execution = """# Configuration
cfg = ExperimentConfig(
    base_seed=42,
    num_trials=5,
    threshold_strategy="confidence",
)

# Initialize pipeline
pipeline = ResearchPipeline(cfg)
inspect_data(pipeline.train_dl)

# Run Architecture Ablation
print("\\n" + "="*60)
print("ARCHITECTURE ABLATION STUDY")
print("="*60)
df_ablation = pipeline.run_architecture_ablation()
print("\\n" + df_ablation.to_string())

# Run Threshold Strategies (Optional - Uncomment to run)
# print("\\n" + "="*60)
# print("THRESHOLD STRATEGY COMPARISON")
# print("="*60)
# pipeline.run_threshold_strategy_comparison()

# Run Model Size Scaling (Optional - Uncomment to run)
# print("\\n" + "="*60)
# print("MODEL SIZE SCALING")
# print("="*60)
# df_sizes = pipeline.run_model_size_experiment()
# print("\\n" + df_sizes.to_string())

# Run Structured Pruning (Optional - Uncomment to run)
# print("\\n" + "="*60)
# print("STRUCTURED PRUNING")
# print("="*60)
# df_pruning = pipeline.run_pruning_experiment()
# print("\\n" + df_pruning.to_string())

# Generate all visualizations and save results
pipeline.generate_visualizations()
pipeline.save_results()
print("\\n[DONE] All experiments complete.")
"""
cells.append(create_cell("code", "\n".join(cleaned_main) + "\n\n" + main_execution))

# Tune Adaptive Width Execution cell
cells.append(create_cell("markdown", "## 10. Tuned Adaptive Width A/B Test (`tune_adaptive.py`)"))
tune_adaptive_content = read_file("tune_adaptive.py")
tune_adaptive_content = strip_local_imports(tune_adaptive_content)

cleaned_tune = []
for line in tune_adaptive_content.split('\n'):
    if line.startswith('if __name__ == "__main__":'):
        cleaned_tune.append("main()")
        break
    cleaned_tune.append(line)

cells.append(create_cell("code", "\n".join(cleaned_tune)))

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
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("final_nb.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Created final_nb.ipynb successfully.")
