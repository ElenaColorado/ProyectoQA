
# ProyectoQA — Extractive Question Answering

## Project Overview

This project implements an extractive question-answering system using DistilBERT fine-tuned on SQuAD 2.0.

The model will extract answers from a given context or return null when no answer is available.

## Selected Model

- Model: `distilbert-base-uncased`
- Dataset: SQuAD 2.0
- Maximum sequence length: 384
- Document stride: 128
- Language: English

DistilBERT was selected because it is a lightweight Transformer encoder suitable for CPU inference.

## Requirements

- Python 3.12
- Git
- Python dependencies from `requirements.txt`

## Installation (Windows PowerShell)

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Dataset Preparation

Generate the training and validation datasets:

```powershell
python src/data/build_dataset.py
```

The processed datasets will be saved in:

- `data/processed/train/`
- `data/processed/validation/`

## Dataset Verification

Verify that the processed datasets can be loaded:

```powershell
python src/data/verify_dataset.py
```

Expected results with the current configuration:

- Training: 131,754 windows
- Validation: 12,134 windows

The processed datasets are excluded from Git and can be regenerated using the preprocessing script.

## Project Status

- [x] SQuAD 2.0 downloaded
- [x] DistilBERT tokenizer configured
- [x] Answerable questions processed
- [x] Unanswerable questions processed
- [x] Long contexts supported through sliding windows
- [x] Training and validation features generated
- [x] Structural validation completed
- [ ] Model fine-tuning
- [ ] Inference and no-answer calibration
- [ ] API implementation and deployment

## Next Stage

The next team member will fine-tune DistilBERT using the prepared datasets and evaluate its performance.
