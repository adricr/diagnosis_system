# Diagnosis System

A machine learning system for diagnosing plant diseases from leaf images, built on the
[PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset). It fine-tunes a
MobileNetV3-Small model (via `torchvision`) to classify leaf images into their disease category.

## Project layout

```
src/
  diagnosis_system/   # dataset loading, transforms, and training entry point
  utils/               # dataset metadata scanning/tracking
PlantDoc-Dataset/
  train/<class_name>/  # training images, one folder per class
  test/<class_name>/   # test images, one folder per class
```

Each class folder name (e.g. `Tomato leaf`, `Apple rust leaf`) is the `class_name` used
throughout the code. `src/utils/metadata_update.py` scans these folders and maintains
`PlantDoc-Dataset/metadata.csv`, which assigns each class a stable `label_index` and tracks
its `train_count` and `test_count`. Label indices are preserved across re-scans so a trained
model's output indices keep referring to the same class even as the dataset changes.

## Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

The `PlantDoc-Dataset` folder (train/test splits) is expected at the project root.

## Usage

Rebuild `metadata.csv` from the dataset folders:

```bash
uv run python src/utils/__init__.py
```

Run the main training entry point:

```bash
uv run diagnosis-system
```

## Status

This project is under active development as part of an Honours project. Training and
evaluation are not yet complete.
