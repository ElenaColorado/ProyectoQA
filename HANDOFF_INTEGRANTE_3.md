# Acta de Entrega y Relevo: Integrante 3 ➔ Integrante 4

**Etapa:** 3 — Inferencia, null threshold, offsets y evaluación
**Responsable:** Integrante 3
**Fecha:** 1 de noviembre de 2026  
**Rama y commit fusionado:** `feature/etapa-3` / commit `3eecd2d`  

## 1. Qué se entrega
- `predict(context: str, question: str) -> dict` en `src/inference/qa_inference.py`. Devuelve exactamente `answer`, `start` y `score`. El `id` y la validación HTTP los hace el Integrante 4 en `app.py`.
- Para cualquier respuesta no nula, `context[start:start+len(answer)] == answer` (el texto sale de `offset_mapping`, no de decodificar subwords, y `start` es relativo al contexto completo).
- Si no hay respuesta: `answer = null` y `start = null`. `score` siempre está en [0, 1] (confianza heurística, **no** probabilidad calibrada).
- Umbral de no-respuesta final: **1.0** (en `configs/no_answer.json`). Regla: `null` si `null_score - best_span_score > threshold`.
- Ventanas de inferencia: 384 tokens, stride 128 (leídas de `configs/`, mismos valores del entrenamiento).

## 2. Archivos creados o modificados
```text
src/inference/qa_inference.py        # predict(), QAPredictor, extract_candidates()
src/inference/calibrate_threshold.py # barrido de umbrales sobre validation
src/inference/reweight_threshold.py  # reponderación 75/25 del barrido (no cambia el modelo)
src/evaluation/evaluate.py           # EM/F1, public sample, latencia CPU
tests/test_inference.py              # contrato + casos semánticos (12 tests)
configs/no_answer.json               # umbral final
logs/threshold_sweep.json
logs/inference_validation_metrics.json
logs/public_sample_results.json
logs/cpu_latency.json
public_sample.json                   # copiado a la raíz (viene del ZIP del docente)
REPORT.md                            # secciones 3, 5, 8 y fila de herramientas de IA (sección 0)
```

## 3. Comandos exactos (desde la raíz del repo)
```powershell
python -m pytest -q
python -m src.inference.calibrate_threshold --n 2000
python -m src.inference.reweight_threshold
python -m src.evaluation.evaluate --n 1000 --public --latency
```
Requisito: `models/qa-final/` con modelo y tokenizer (fuera de Git).
Uso desde la API: `from src.inference.qa_inference import predict` (carga el modelo una sola vez).

## 4. Tests
`python -m pytest -q` → `15 passed in 9.64s`
## 5. Métricas reales
Validación en 1000 ejemplos que la calibración no usó (umbral 1.0):

| | n | EM | F1 |
|---|---|---|---|
| Global | 1000 | 59.0 | 63.25 |
| HasAns | 534 | 62.17 | 70.13 |
| NoAns | 466 | 55.36 | 55.36 |

Public sample: sample-01 `16` (start 383) ✔, sample-02 `3.5 days` (437) ✔, sample-03 `null` ✔, sample-04 respondió `3` (start 310) ✘ (debía ser `null`), sample-05 `50,257` (3157) ✔.

Latencia CPU, **solo inferencia local, sin túnel**: media 0.126 s, p95 0.404 s, máx 0.439 s por pregunta (25 llamadas). **Falta medir end-to-end por HTTPS con el túnel.**

## 6. Dependencias y versiones
 Python 3.13.7 · transformers 5.2.0 · datasets 5.1.0 · pytest 9.2.0 · torch 2.15.0+cpu 
`requirements.txt` **no incluye torch, fastapi ni uvicorn**: agregarlos antes de documentar la instalación "desde un equipo limpio".

## 7. Problemas reales y fixes
1. `load_dataset("squad_v2")` falló con `HfUriError: Repository id must be 'namespace/name'` en las versiones instaladas. Fix: usar `rajpurkar/squad_v2`. Commit: `3eecd2d`
2. La evaluación reutilizaba los mismos ejemplos de la calibración (misma semilla, primeros n), lo que daba métricas optimistas. Fix: la calibración usa los primeros 2000 del validation barajado y la evaluación los últimos n. Commit: `3eecd2d`

## 8. Pendientes explícitos y avisos para el Integrante 4
- Medir latencia por el túnel y proyectar los 150 s para 50 preguntas.
- `REPORT.md` sección 6: poner CPU y RAM **reales** de la laptop de servicio (hoy dice "Intel Core i5 / AMD Ryzen, 16 GB", que es genérico).
- `REPORT.md` sección 7: reemplazar `[Stage 2 commit]` por el hash real (Integrante 2).
- Falta `configs/project.json` (previsto para el Integrante 1). Inferencia lee 384/128 de `configs/training_config.json`.
- El modelo es solo en inglés; entradas en otro idioma no están garantizadas (documentar en la sección 9).
- sample-04 falla (pregunta con entidad no sustentada); está explicado en la sección 5 del reporte. No se ajustó el umbral para corregirlo.
- `reweight_threshold.py --write` ahora exige `--threshold X`; no sobrescribe el umbral solo.

**REVISADO Y RECIBIDO POR:** ____________________
