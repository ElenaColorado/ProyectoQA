"""
test_training.py - Pruebas unitarias para la etapa de entrenamiento (Integrante 2)
"""

import json
from pathlib import Path
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


def test_config_structure():
    config_path = Path("configs/training_config.json")
    assert config_path.exists(), "El archivo configs/training_config.json no existe"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    required_keys = [
        "model_name_or_path",
        "max_seq_length",
        "doc_stride",
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "seed"
    ]
    for key in required_keys:
        assert key in config, f"Falta la clave requerida '{key}' en training_config.json"


def test_model_initialization_and_forward():
    # Preferir modelo final local si existe (para ejecucion rapida y offline)
    final_dir = Path("models/qa-final")
    if final_dir.exists() and (final_dir / "model.safetensors").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(final_dir), local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(str(final_dir), local_files_only=True)
    else:
        model_name = "distilbert/distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Probar pase hacia adelante (forward pass) con un par pregunta-contexto sintetico
    question = "Where is El Salvador located?"
    context = "El Salvador is a country in Central America."
    
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=64,
        truncation=True
    )
    
    # Agregar etiquetas ficticias
    inputs["start_positions"] = torch.tensor([5])
    inputs["end_positions"] = torch.tensor([7])
    
    outputs = model(**inputs)
    
    assert outputs.loss is not None, "El modelo debe calcular la funcion de perdida (loss)"
    assert outputs.start_logits is not None, "El modelo debe generar start_logits"
    assert outputs.end_logits is not None, "El modelo debe generar end_logits"
    assert not torch.isnan(outputs.loss), "La perdida no debe ser NaN"


def test_training_artifacts_and_metrics():
    # Verificar que los artefactos obligatorios de la Etapa 2 existan y sean validos
    metrics_path = Path("logs/validation_metrics.json")
    history_path = Path("logs/training_history.csv")
    loss_png_path = Path("logs/loss.png")
    
    assert metrics_path.exists(), "Falta logs/validation_metrics.json"
    assert history_path.exists(), "Falta logs/training_history.csv"
    assert loss_png_path.exists(), "Falta logs/loss.png"
    
    # Validar que metrics sea un JSON coherente
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "train_loss" in metrics and metrics["train_loss"] is not None
    assert "eval_loss" in metrics and metrics["eval_loss"] is not None
    assert metrics["eval_loss"] < 2.0, "La perdida de evaluacion debe reflejar aprendizaje real (< 2.0)"
    
    # Validar que history contenga registros
    df = pd.read_csv(history_path)
    assert len(df) > 0, "logs/training_history.csv no debe estar vacio"
    assert "step" in df.columns
