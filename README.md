
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
- [x] Model fine-tuning (DistilBERT on SQuAD 2.0 completed)
- [ ] Inference and no-answer calibration
- [ ] API implementation and deployment

## Next Stage

The next team member (Member 3) will implement the inference pipeline, answer extraction logic, and calibrate the no-answer (null) threshold using the fine-tuned model weights.


## Using the Preprocessed Dataset (Team Member 2)

The preprocessed SQuAD 2.0 dataset is shared separately as:

`SQuAD2_procesado.zip`

### Instructions

1. Clone the GitHub repository.
2. Install the dependencies from `requirements.txt`.
3. Download `SQuAD2_procesado.zip` and save it in your Windows Downloads folder.
4. Open PowerShell in the root of the project.
5. Extract the dataset using:

```powershell
Expand-Archive -Path "$HOME\Downloads\SQuAD2_procesado.zip" -DestinationPath ".\data\processed" -Force
```

The resulting directory structure must be:

```text
data/
  processed/
    train/
    validation/
```

6. Verify the datasets:

```powershell
python src/data/verify_dataset.py
```

Expected output:

- Training: 131,754 windows
- Validation: 12,134 windows

If verification succeeds, the datasets are ready for DistilBERT fine-tuning.

**Note:** Do not upload the processed dataset or model weights to Git. The original SQuAD 2.0 dataset can also be regenerated using `python src/data/build_dataset.py`.


## Using the Fine-Tuned Model (Team Member 3)

The fine-tuned model weights and tokenizer are exported to models/qa-final/ (excluded from Git).

### Instructions for Member 3

1. Transfer the models/qa-final/ directory via shared cloud storage into your local ProyectoQA/models/qa-final/.
2. Run unit tests to verify your environment and offline model loading:
`powershell
pytest tests/test_training.py -v
`
3. Load the model and tokenizer offline:
`python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained('models/qa-final', local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained('models/qa-final', local_files_only=True)
`
Refer to HANDOFF_INTEGRANTE_2.md for complete handover metrics, hyperparameters, and instructions.
