"""Inferencia de extractive QA (SQuAD 2.0) -- Integrante 3.

Interfaz publica (la que usa el Integrante 4 en app.py):

    predict(context: str, question: str) -> {"answer": str|None, "start": int|None, "score": float}

Idea general:
  1. Tokenizar (pregunta, contexto) en ventanas solapadas (max_seq_length / doc_stride).
  2. El modelo da start_logits y end_logits por ventana.
  3. Se generan candidatos (inicio, fin) SOLO sobre tokens del contexto.
  4. Con offset_mapping se recuperan los caracteres EXACTOS del contexto original
     (nunca se reconstruye texto desde subwords).
  5. Se comparan candidatos de TODAS las ventanas y se elige el mejor.
  6. Se compara contra el score "null" (token [CLS], indice 0) y un umbral calibrado.

Convencion del score null (documentada, ver REPORT.md seccion 3):
    null_score = min sobre ventanas de (start_logit[CLS] + end_logit[CLS])
    best_score = max sobre ventanas y candidatos de (start_logit[s] + end_logit[e])
    diff       = null_score - best_score
    Se devuelve null si diff > threshold.   (threshold MAYOR => MENOS respuestas null)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "qa-final"
PROJECT_CFG = ROOT / "configs" / "project.json"
TRAINING_CFG = ROOT / "configs" / "training_config.json"  # lo que uso el Integrante 2
NO_ANSWER_CFG = ROOT / "configs" / "no_answer.json"


def _load_json(path: Path, default: dict) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def sigmoid(x: float) -> float:
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def extract_candidates(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    offsets: Sequence[Optional[tuple]],
    top_k: int = 20,
    max_answer_len: int = 30,
) -> list:
    """Candidatos (score, start_char, end_char) de UNA ventana.

    `offsets[i]` es (char_ini, char_fin) si el token i pertenece al contexto,
    o None si es pregunta / especial / padding (esos tokens se excluyen).
    """
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)
    start_idx = np.argsort(start_logits)[::-1][:top_k]
    end_idx = np.argsort(end_logits)[::-1][:top_k]
    out = []
    for s in start_idx:
        if offsets[s] is None:
            continue
        for e in end_idx:
            if offsets[e] is None:
                continue
            if e < s or (e - s + 1) > max_answer_len:
                continue
            score = float(start_logits[s] + end_logits[e])
            out.append((score, int(offsets[s][0]), int(offsets[e][1])))
    return out


class QAPredictor:
    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        max_seq_length: Optional[int] = None,
        doc_stride: Optional[int] = None,
        threshold: Optional[float] = None,
        top_k: int = 20,
        max_answer_len: int = 30,
        batch_size: int = 8,
    ):
        import torch  # imports perezosos: los tests puros no necesitan torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        # Orden: argumento > configs/project.json > configs/training_config.json > 384/128.
        # (Las ventanas de inferencia deben ser las mismas con las que se entreno.)
        proj = {**_load_json(TRAINING_CFG, {}), **_load_json(PROJECT_CFG, {})}
        self.max_seq_length = max_seq_length or proj.get("max_seq_length") or proj.get("max_length") or 384
        self.doc_stride = doc_stride or proj.get("doc_stride") or proj.get("stride") or 128
        if threshold is None:
            threshold = _load_json(NO_ANSWER_CFG, {}).get("threshold", 0.0)
        self.threshold = float(threshold)
        self.top_k = top_k
        self.max_answer_len = max_answer_len
        self.batch_size = batch_size

        self._torch = torch
        model_dir = str(model_dir)
        # local_files_only: la inferencia NO debe depender de la red
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
        self.model.to("cpu").eval()

    # ------------------------------------------------------------------
    def analyze(self, context: str, question: str) -> dict:
        """Mejor span + score null, SIN aplicar umbral (lo usa la calibracion)."""
        torch = self._torch
        enc = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        )
        n_windows = len(enc["input_ids"])
        keys = [k for k in self.tokenizer.model_input_names if k in enc]

        best = None  # (score, start_char, end_char)
        null_score = None
        for i in range(0, n_windows, self.batch_size):
            batch = {k: torch.tensor(enc[k][i : i + self.batch_size]) for k in keys}
            with torch.inference_mode():
                out = self.model(**batch)
            sl = out.start_logits.numpy()
            el = out.end_logits.numpy()
            for j in range(sl.shape[0]):
                w = i + j
                seq_ids = enc.sequence_ids(w)
                offsets = [
                    tuple(o) if seq_ids[t] == 1 else None
                    for t, o in enumerate(enc["offset_mapping"][w])
                ]
                ns = float(sl[j][0] + el[j][0])  # token [CLS] = indice 0
                null_score = ns if null_score is None else min(null_score, ns)
                for cand in extract_candidates(sl[j], el[j], offsets, self.top_k, self.max_answer_len):
                    if context[cand[1] : cand[2]].strip() == "":
                        continue
                    if best is None or cand[0] > best[0]:
                        best = cand

        if best is None:
            # Sin candidatos validos: tratar como "no hay respuesta".
            return {"answer": None, "start": None, "span_score": None, "null_score": null_score, "diff": 20.0}
        score, a, b = best
        return {
            "answer": context[a:b],  # substring EXACTO del contexto original
            "start": a,
            "span_score": score,
            "null_score": null_score,
            "diff": null_score - score,
        }

    # ------------------------------------------------------------------
    def predict(self, context: str, question: str) -> dict:
        r = self.analyze(context, question)
        margin = r["diff"] - self.threshold
        p_null = sigmoid(margin)  # confianza heuristica de "no hay respuesta"
        if r["answer"] is None or margin > 0:
            return {"answer": None, "start": None, "score": float(p_null)}
        return {"answer": r["answer"], "start": int(r["start"]), "score": float(1.0 - p_null)}


_default: Optional[QAPredictor] = None


def predict(context: str, question: str) -> dict:
    """Interfaz para app.py. Carga el modelo una sola vez."""
    global _default
    if _default is None:
        _default = QAPredictor()
    return _default.predict(context, question)
