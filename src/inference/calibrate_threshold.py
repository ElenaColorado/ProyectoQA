"""Calibra el umbral de no-respuesta sobre el validation set de SQuAD 2.0.

Requiere el modelo REAL en models/qa-final/. NO se calibra con public_sample.json
(5 ejemplos => sobreajuste) ni con el test oculto.

Uso (desde la raiz del repo):
    python -m src.inference.calibrate_threshold --n 2000

Se corre el modelo UNA vez por ejemplo (analyze) y luego se barren los umbrales
sobre los valores guardados, asi que el barrido es casi gratis.

Regla: se devuelve null si diff = null_score - best_span_score > threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.evaluate import aggregate, score_example
from src.inference.qa_inference import NO_ANSWER_CFG, QAPredictor

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="ejemplos de validation a usar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lo", type=float, default=-10.0)
    ap.add_argument("--hi", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.25)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=args.seed).select(range(args.n))
    predictor = QAPredictor(threshold=0.0)

    cache = []  # (golds, answer, diff)
    for i, ex in enumerate(ds):
        r = predictor.analyze(ex["context"], ex["question"])
        cache.append((ex["answers"]["text"], r["answer"], r["diff"]))
        if (i + 1) % 200 == 0:
            print(f"  analizados {i + 1}/{args.n}")

    sweep = []
    for t in np.arange(args.lo, args.hi + 1e-9, args.step):
        rows = []
        for golds, answer, diff in cache:
            pred = None if (answer is None or diff > t) else answer
            em, f1 = score_example(golds, pred)
            rows.append((bool(golds), em, f1))
        m = aggregate(rows)
        sweep.append({"threshold": round(float(t), 3), **{k: m[k] for k in ("overall", "HasAns", "NoAns")}})

    best = max(sweep, key=lambda s: s["overall"]["F1"])
    print(f"\nMejor umbral: {best['threshold']}  F1={best['overall']['F1']}  EM={best['overall']['EM']}")
    print(f"  HasAns F1={best['HasAns']['F1']}  NoAns F1={best['NoAns']['F1']}")

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))
    NO_ANSWER_CFG.write_text(
        json.dumps(
            {
                "threshold": best["threshold"],
                "criterion": "max F1 overall",
                "n_validation_examples": args.n,
                "seed": args.seed,
                "rule": "null if (null_score - best_span_score) > threshold",
            },
            indent=2,
        )
    )
    print(f"Guardado en {NO_ANSWER_CFG} y logs/threshold_sweep.json")


if __name__ == "__main__":
    main()
