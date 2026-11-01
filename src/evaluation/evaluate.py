"""Metricas SQuAD 2.0 (EM / F1), evaluacion en validation, public sample y latencia.

Uso (desde la raiz del repo, con models/qa-final/ presente):
    python -m src.evaluation.evaluate --n 1000          # EM/F1 en validation
    python -m src.evaluation.evaluate --public          # public_sample.json
    python -m src.evaluation.evaluate --latency         # latencia CPU
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import string
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"


# ---- logica oficial de SQuAD v2 (normalizacion, EM, F1) -----------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


def compute_f1(gold: str, pred: str) -> float:
    g = normalize_answer(gold).split()
    p = normalize_answer(pred).split()
    if not g or not p:  # si alguno es "no answer", F1 = 1 solo si ambos lo son
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def score_example(gold_answers: list, pred: str | None) -> tuple:
    """gold_answers vacio => pregunta imposible (se compara contra '')."""
    golds = gold_answers if gold_answers else [""]
    pred = pred or ""
    return (
        max(compute_exact(g, pred) for g in golds),
        max(compute_f1(g, pred) for g in golds),
    )


def aggregate(rows: list) -> dict:
    """rows: [(has_answer: bool, em, f1)] -> metricas global / HasAns / NoAns (0-100)."""
    def agg(sub):
        if not sub:
            return {"n": 0, "EM": None, "F1": None}
        return {
            "n": len(sub),
            "EM": round(100 * sum(r[1] for r in sub) / len(sub), 2),
            "F1": round(100 * sum(r[2] for r in sub) / len(sub), 2),
        }
    return {
        "overall": agg(rows),
        "HasAns": agg([r for r in rows if r[0]]),
        "NoAns": agg([r for r in rows if not r[0]]),
    }


# ---- ejecuciones --------------------------------------------------------
def eval_validation(predictor, n: int, seed: int = 42) -> dict:
    from datasets import load_dataset

    # Se toman los ULTIMOS n del validation barajado: la calibracion usa los PRIMEROS
    # (calibrate_threshold.py), asi que no se solapan mientras n_calib + n_eval <= total.
    ds = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=seed)
    ds = ds.select(range(len(ds) - n, len(ds)))
    rows = []
    for ex in ds:
        golds = ex["answers"]["text"]
        out = predictor.predict(ex["context"], ex["question"])
        em, f1 = score_example(golds, out["answer"])
        rows.append((bool(golds), em, f1))
    res = aggregate(rows)
    res.update({"threshold": predictor.threshold, "seed": seed, "n_examples": n})
    return res


def _load_public() -> list:
    with open(ROOT / "public_sample.json", encoding="utf-8") as f:
        data = json.load(f)
    # Formato no verificado: aceptamos lista directa o dict con una lista dentro.
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return data


def run_public(predictor) -> list:
    results = []
    for ex in _load_public():
        out = predictor.predict(ex["context"], ex["question"])
        results.append({"id": ex.get("id"), **out})
    return results


def run_latency(predictor, repeats: int = 5) -> dict:
    items = _load_public()
    times = []
    predictor.predict(items[0]["context"], items[0]["question"])  # warm-up
    for _ in range(repeats):
        for ex in items:
            t0 = time.perf_counter()
            predictor.predict(ex["context"], ex["question"])
            times.append(time.perf_counter() - t0)
    times.sort()
    mean = statistics.mean(times)
    return {
        "n_calls": len(times),
        "mean_s": round(mean, 3),
        "p95_s": round(times[int(0.95 * (len(times) - 1))], 3),
        "max_s": round(times[-1], 3),
        "projected_50_questions_s": round(mean * 50, 1),
        "note": "solo inferencia local, SIN tunel/red; el limite global es 150 s",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="N ejemplos de validation")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    from src.inference.qa_inference import QAPredictor

    predictor = QAPredictor()
    LOGS.mkdir(exist_ok=True)
    if args.n:
        res = eval_validation(predictor, args.n)
        # nombre distinto al del Integrante 2 (logs/validation_metrics.json) para no pisarlo
        (LOGS / "inference_validation_metrics.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    if args.public:
        res = run_public(predictor)
        (LOGS / "public_sample_results.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    if args.latency:
        res = run_latency(predictor)
        (LOGS / "cpu_latency.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
