"""
train.py - Entrenamiento y Fine-Tuning de DistilBERT para Question Answering (SQuAD 2.0)
Rol: Integrante 2 (Entrenamiento, Validación y Curvas de Pérdida)

Este script realiza el fine-tuning del encoder base 'distilbert-base-uncased'
usando los datasets preprocesados de SQuAD 2.0 generados en la Etapa 1.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Configurar directorio de caché para matplotlib para evitar advertencias de permisos
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import torch
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    set_seed
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tuning de DistilBERT para Extractive QA (SQuAD 2.0)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.json",
        help="Ruta al archivo JSON de configuracion de hiperparametros"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Ejecutar una prueba rapida con 300 muestras para verificar pipeline y GPU"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Entrenamiento piloto con subconjunto representativo (20,000 train, 2,000 val)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Sobrescribir numero de epocas definido en config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sobrescribir batch size por dispositivo"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Sobrescribir tasa de aprendizaje"
    )
    return parser.parse_args()


def load_config(config_path):
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"[AVISO] Archivo de configuracion '{config_path}' no encontrado. Usando valores por defecto.")
    return {
        "model_name_or_path": "distilbert/distilbert-base-uncased",
        "max_seq_length": 384,
        "doc_stride": 128,
        "learning_rate": 3e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "seed": 42,
        "output_dir": "models/qa-checkpoints",
        "final_model_dir": "models/qa-final",
        "logging_dir": "logs"
    }


def plot_and_save_losses(log_history, output_png_path, output_csv_path):
    """
    Extrae la historia de perdida de entrenamiento y evaluacion,
    la guarda en CSV y genera la grafica loss.png requerida.
    """
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    # Guardar en CSV
    history_records = []
    for i in range(len(train_steps)):
        history_records.append({
            "step": train_steps[i],
            "train_loss": train_losses[i],
            "type": "train"
        })
    for i in range(len(eval_steps)):
        history_records.append({
            "step": eval_steps[i],
            "eval_loss": eval_losses[i],
            "type": "eval"
        })

    df = pd.DataFrame(history_records)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[OK] Historial de entrenamiento guardado en: {output_csv_path}")

    # Generar grafica
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss", color="#1f77b4", linewidth=2)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="Validation Loss", color="#ff7f0e", linewidth=2, marker="o")

    plt.title("Curva de Aprendizaje (Loss vs Steps) - DistilBERT QA", fontsize=14, fontweight="bold")
    plt.xlabel("Pasos de Entrenamiento (Steps)", fontsize=12)
    plt.ylabel("Pérdida (Loss)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"[OK] Curva de perdida guardada en: {output_png_path}")


def main():
    args = parse_arguments()
    config = load_config(args.config)

    # Sobrescribir con argumentos CLI si se proporcionaron
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size
        config["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    # Fijar semilla para reproducibilidad
    set_seed(config.get("seed", 42))

    print("\n" + "=" * 60)
    print("ETAPA 2: FINE-TUNING DE DISTILBERT EN SQUAD 2.0")
    print("=" * 60)

    # 1. Comprobacion de dispositivo de aceleracion
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Dispositivo detectado: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[INFO] GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[INFO] GPU no detectada. El entrenamiento se realizara sobre CPU.")

    # 2. Carga de datasets preprocesados (de Etapa 1)
    data_dir = Path("data/processed")
    train_path = data_dir / "train"
    val_path = data_dir / "validation"

    if not train_path.exists() or not val_path.exists():
        print(f"[ERROR] No se encontraron los datos preprocesados en {data_dir.resolve()}.")
        print("Asegurate de haber ejecutado src/data/verify_dataset.py primero.")
        sys.exit(1)

    print(f"\n[1/5] Cargando datos procesados desde: {data_dir}...")
    train_dataset = load_from_disk(str(train_path))
    eval_dataset = load_from_disk(str(val_path))

    # Ajuste segun modo (smoke test, pilot o full)
    if args.smoke_test:
        print("\n>>> MODO SMOKE TEST ACTIVO <<<")
        print("Seleccionando subconjunto minimo de prueba (300 train, 100 val)...")
        train_dataset = train_dataset.select(range(min(300, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(100, len(eval_dataset))))
        config["num_train_epochs"] = 1
        logging_steps = 10
        eval_steps = 20
    elif args.pilot:
        print("\n>>> MODO PILOTO ACTIVO <<<")
        print("Seleccionando 20,000 ejemplos de train y 2,000 de val...")
        train_dataset = train_dataset.select(range(min(20000, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(2000, len(eval_dataset))))
        logging_steps = 100
        eval_steps = 250
    else:
        print("\n>>> MODO COMPLETO ACTIVO <<<")
        logging_steps = 200
        eval_steps = 500

    print(f"Total ventanas de entrenamiento: {len(train_dataset):,}")
    print(f"Total ventanas de validacion:   {len(eval_dataset):,}")

    # 3. Cargar Tokenizer y Modelo Base
    # REGLA ESTRICTA: Partir de checkpoint BASE, no uno previamente ajustado con SQuAD
    model_name = config.get("model_name_or_path", "distilbert/distilbert-base-uncased")
    print(f"\n[2/5] Descargando/Cargando checkpoint BASE: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    print(f"[OK] Modelo inicializado. Total parametros: {model.num_parameters():,}")

    # 4. Configuracion del Trainer
    output_dir = Path(config.get("output_dir", "models/qa-checkpoints"))
    logs_dir = Path(config.get("logging_dir", "logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = (device == "cuda")

    # Adaptabilidad entre versiones de transformers
    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": config.get("learning_rate", 3e-5),
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 8),
        "per_device_eval_batch_size": config.get("per_device_eval_batch_size", 8),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 2),
        "num_train_epochs": config.get("num_train_epochs", 2),
        "weight_decay": config.get("weight_decay", 0.01),
        "logging_steps": logging_steps,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "seed": config.get("seed", 42),
        "push_to_hub": False,
        "report_to": "none"  # Desactivar integraciones externas (W&B, etc.)
    }

    # Estrategia de evaluacion y guardado
    if args.smoke_test:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = eval_steps
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["save_steps"] = eval_steps
    elif args.pilot:
        training_kwargs["eval_strategy"] = "steps"
        training_kwargs["eval_steps"] = eval_steps
        training_kwargs["save_strategy"] = "steps"
        training_kwargs["save_steps"] = eval_steps
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "loss"
        training_kwargs["greater_is_better"] = False
    else:
        training_kwargs["eval_strategy"] = "epoch"
        training_kwargs["save_strategy"] = "epoch"
        training_kwargs["load_best_model_at_end"] = True
        training_kwargs["metric_for_best_model"] = "loss"
        training_kwargs["greater_is_better"] = False

    training_args = TrainingArguments(**training_kwargs)

    # Compatible con transformers 5.x (processing_class) y versiones anteriores (tokenizer)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    # 5. Ejecucion del entrenamiento
    print("\n[3/5] Iniciando entrenamiento con Hugging Face Trainer...")
    print(f"Hiperparametros: Epochs={config.get('num_train_epochs')}, LR={config.get('learning_rate')}, BatchSize={config.get('per_device_train_batch_size')}, FP16={use_fp16}")

    train_result = trainer.train()

    print("\n[4/5] Entrenamiento finalizado con exito!")
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # Evaluacion final
    print("\nEvaluando modelo en el conjunto de validacion...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Guardar metricas en JSON para el reporte
    metrics_summary_path = logs_dir / "validation_metrics.json"
    summary_data = {
        "model_name": model_name,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "epochs": config.get("num_train_epochs"),
        "batch_size": config.get("per_device_train_batch_size"),
        "learning_rate": config.get("learning_rate"),
        "device": device,
        "train_loss": train_metrics.get("train_loss", None),
        "eval_loss": eval_metrics.get("eval_loss", None),
        "total_flos": train_metrics.get("total_flos", None),
        "train_runtime_seconds": train_metrics.get("train_runtime", None)
    }
    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Metricas consolidadas guardadas en: {metrics_summary_path}")

    # Guardar graficas y logs
    plot_and_save_losses(
        trainer.state.log_history,
        output_png_path=str(logs_dir / "loss.png"),
        output_csv_path=str(logs_dir / "training_history.csv")
    )

    # 6. Exportar modelo final y tokenizer para el Integrante 3
    final_dir = Path(config.get("final_model_dir", "models/qa-final"))
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[5/5] Exportando pesos finales y tokenizer a '{final_dir}'...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\n" + "=" * 60)
    print("PAQUETE DE ENTREGA (RELEVO 2 -> 3) LISTO:")
    print(f"1. Checkpoint y Tokenizer: {final_dir.resolve()}")
    print(f"2. Curva de perdida:       {(logs_dir / 'loss.png').resolve()}")
    print(f"3. Historial CSV:          {(logs_dir / 'training_history.csv').resolve()}")
    print(f"4. Metricas JSON:          {metrics_summary_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

