# Early-Exit Neural Network Architectures

Energy-efficient biomedical signal classification using early-exit CNNs. Compares five architectures across three clinical datasets (Bonn EEG, ECG Arrhythmia, MIT-BIH).

**Rithwak Somepalli · Suryaprakash Murugavvel · Monique Gaye · Amrutha Kodali**

## Quick Start

```bash
git clone https://github.com/Srithwak/Early_exit_architecture_research.git
cd Early_exit_architecture_research

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place datasets in `data/bonn/`, `data/ecg/`, `data/mitbih/` as NumPy arrays (`X_train.npy`, `y_train.npy`, etc.). Dummy data is generated automatically if files are missing.

## Running Experiments

```bash
# Bonn EEG (seizure detection, binary, 280 samples)
python main.py --run-all --trials 5

# ECG Arrhythmia (5-class, 87K samples)
python main_ecg.py --run-all --trials 1

# MIT-BIH Arrhythmia (5-class, 77K samples)
python main_mitbih.py --run-all --trials 1
```

Results (CSV, JSON) go to `results*/`, plots (PNG, PDF) to `plots*/`.

## CLI Options

```
--run-ablation    Architecture comparison (5 models)
--run-sizes       Model size scaling experiment
--run-pruning     Structured pruning experiment
--run-strategies  Compare threshold strategies (confidence/entropy/patience)
--run-tuning      Hyperparameter grid search
--run-all         Run everything
--trials N        Number of trials per config (default: 3)
--seed S          Random seed (default: 42)
--strategy STR    Threshold strategy: confidence|entropy|patience
```

## Project Structure

```
main.py              Bonn EEG experiment pipeline
main_ecg.py          ECG Arrhythmia pipeline
main_mitbih.py       MIT-BIH pipeline
models.py            All architectures, losses, pruning
dataset.py           Bonn EEG data loader
dataset_ecg.py       ECG data loader
dataset_mitbih.py    MIT-BIH data loader
train.py             Two-phase training + threshold calibration
evaluate.py          Evaluation (accuracy, F1, ECE, latency)
analysis.py          Exit behavior analysis
visualize.py         Plot generation
statistical_tests.py Statistical significance testing
tuning.py            Hyperparameter search
metrics.md           All experimental results and architecture details
presentation.md      Presentation slides and script
requirements.txt     Dependencies
```

## Datasets

| Dataset | Samples | Seq Len | Classes | Domain |
|---|---|---|---|---|
| Bonn EEG | 400 | 4,097 | 2 | EEG Seizure Detection |
| ECG Arrhythmia | ~109K | 187 | 5 | ECG Classification |
| MIT-BIH | ~96K | 187 | 5 | ECG Classification |

All use 6-channel input (1 raw signal + 5 FFT frequency bands).

## Colab

| Notebook | Purpose |
|---|---|
| [Preprocessing](https://colab.research.google.com/drive/1cZWTCWrdIi9rCaKxJfOZ2dzz9EKVXI86?usp=sharing) | Data loading, normalization |
| [Implementation](https://colab.research.google.com/drive/1lwuYQafT6nnBWmTOMrky1WJo3uo7_UbN?usp=sharing) | Training, evaluation, plots |

Dataset: [Google Drive](https://drive.google.com/drive/folders/132mm7W9rXuarB2wd_cMy4n9-qeTmIJFX?usp=sharing)

## References

1. Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
2. Bonn EEG Dataset. University of Bonn, Dept. of Epileptology.
3. Moody & Mark (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE EMB Magazine*, 20(3).
