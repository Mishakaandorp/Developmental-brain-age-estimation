# Developmental Brain Age Estimation

This repository contains training and evaluation code for deep learning-based developmental brain age estimation from 3D fetal and neonatal MRI data or derived brain label maps.

The code accompanies the medRxiv preprint:

> Kaandorp, M. et al. *Developmental brain age gap in prematurity and postnatally emerging delay in congenital heart disease*. medRxiv, 2026.  
> Available at: https://www.medrxiv.org/content/10.64898/2026.04.01.26349523v1

## Overview

Brain age estimation aims to predict chronological or gestational age from neuroimaging data. The difference between predicted brain age and chronological age is commonly referred to as the brain age gap, BAG. In developmental cohorts, BAG can be used to quantify deviations from normative fetal and neonatal brain maturation.

This implementation trains a 3D DenseNet201 regression model using MONAI and PyTorch. The model takes a single-channel 3D NIfTI volume as input and predicts age as a continuous scalar.

The current scripts support:

- training a 3D DenseNet201 brain age regression model,
- optional test-set evaluation during training,
- checkpoint saving for the latest and best model,
- CSV logging of training and validation metrics,
- plotting training curves,
- model evaluation on held-out NIfTI volumes,
- estimation of mean absolute error and brain age gap statistics.

## Repository structure

```text
.
├── training.py              # Training script for the 3D DenseNet201 age regression model
├── eval.py                  # Evaluation script for a trained model
├── requirements.txt         # Python package requirements
├── Datasets/                # Placeholder for local datasets, not tracked by Git
├── Trained_models/          # Placeholder for trained model checkpoints, not tracked by Git
└── eval/                    # Evaluation outputs, plots, and metrics
```

## Installation

Create and activate a Python environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU use, make sure that your installed PyTorch build is compatible with your CUDA driver. The provided `requirements.txt` reflects the original development environment and may need adaptation on a different server.

## Data format

The training and evaluation scripts expect 3D NIfTI files:

```text
.nii
.nii.gz
```

The corresponding age labels are read from a tab-separated `.tsv` file. The scripts currently look for one of the following age columns:

```text
scan_age_mri0gamri
scan_age
```

The number of image files in each image directory must match the number of valid age rows in the corresponding TSV file. Files are loaded in sorted filename order, so the TSV rows must be in the same subject order as the sorted image list.

Example dataset layout:

```text
Datasets/
└── FETA_dHCP/
    ├── T2_train/
    │   ├── subject_001.nii.gz
    │   ├── subject_002.nii.gz
    │   └── ...
    ├── train.tsv
    ├── T2_test/
    │   ├── subject_101.nii.gz
    │   ├── subject_102.nii.gz
    │   └── ...
    └── test.tsv
```

## Configuration

Before running the scripts, edit the `Config` class in `training.py` and `eval.py`.

In `training.py`, update:

```python
train_dir = "./Datasets/FETA_dHCP/T2_train"
train_tsv = "./Datasets/FETA_dHCP/train.tsv"
test_dir = "./Datasets/FETA_dHCP/T2_test"
test_tsv = "./Datasets/FETA_dHCP/test.tsv"
checkpoint_dir = "./Trained_models/Train_1"
```

In `eval.py`, update:

```python
test_dir = "./Datasets/FETA_dHCP/T2_test"
test_tsv = "./Datasets/FETA_dHCP/test.tsv"
model_name = "Model_A"
model_path = f"./Trained_models/Model_A/{model_name}.pth"
```

## Training

Run:

```bash
python training.py
```

The training script uses:

- optimizer: Adam,
- loss: Smooth L1 loss, also known as Huber loss,
- learning rate: `1e-3`,
- scheduler: `ReduceLROnPlateau`,
- default batch size: `8`,
- default number of epochs: `300`,
- model: MONAI 3D DenseNet201.

Training outputs are written to the configured checkpoint directory:

```text
Trained_models/Train_1/
├── latest_model.pth
├── best_model.pth
├── training_log.csv
└── training_curve.png
```

## Evaluation

Run:

```bash
python eval.py
```

The evaluation script loads the configured model checkpoint and test set, then reports:

- mean absolute error, MAE,
- mean brain age gap, BAG mean,
- standard deviation of brain age gap, BAG std.

It also saves a BAG violin plot:

```text
eval/<model_name>/bag_violin.png
```

## Notes on outputs and large files

Large datasets, model checkpoints, and generated evaluation outputs should generally not be committed directly to GitHub. Keep these files locally, on institutional storage, or in a dedicated model/data repository.

Recommended ignored paths include:

```text
Datasets/
Trained_models/**/*.pth
eval/
*.nii
*.nii.gz
```

## Citation

If you use this code, please cite:

```bibtex
@article{kaandorp2026developmentalbrainagegap,
  title = {Developmental brain age gap in prematurity and postnatally emerging delay in congenital heart disease},
  author = {Kaandorp, M. and others},
  journal = {medRxiv},
  year = {2026},
  note = {Preprint},
  url = {https://www.medrxiv.org/content/10.64898/2026.04.01.26349523v1}
}
```
