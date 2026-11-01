"""Reelige el umbral usando la mezcla del test oculto (~25% imposibles).

El validation de SQuAD 2.0 tiene ~50% de preguntas sin respuesta; el test del
docente ~25%. Aqui se reutiliza el barrido ya calculado (logs/threshold_sweep.json),
SIN volver a correr el modelo, y se pondera:

    F1_ponderado = (1 - p_noans) * F1_HasAns + p_noans * F1_NoAns

Uso (desde la raiz del repo):
    python -m src.inference.reweight_threshold                          # solo muestra
    python -m src.inference.reweight_threshold --write --threshold 1.0  # guarda EL UMBRAL QUE TU ELIJAS

--write ya NO guarda automaticamente el maximo: el umbral final es una decision del
equipo (p. ej. un valor dentro de la meseta), y debe ser uno de los valores del barrido.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "logs" / "threshold_sweep.json"
CFG = ROOT / "configs" / "no_answer.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-noans", type=float, default=0.25, help="proporcion de imposibles esperada")
    ap.add_argument("--write", action="store_true", help="guardar en configs/no_answer.json (requiere --threshold)")
    ap.add_argument("--threshold", type=float, default=None, help="umbral elegido (debe estar en el barrido)")
    args = ap.parse_args()
    p = args.p_noans

    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    for s in sweep:
        s["w_f1"] = (1 - p) * s["HasAns"]["F1"] + p * s["NoAns"]["F1"]
        s["w_em"] = (1 - p) * s["HasAns"]["EM"] + p * s["NoAns"]["EM"]

    best_plain = max(sweep, key=lambda s: s["overall"]["F1"])
    best_w = max(sweep, key=lambda s: s["w_f1"])

    print(f"Mezcla usada: {round((1 - p) * 100)}% con respuesta / {round(p * 100)}% sin respuesta\n")
    print(f"{'umbral':>7} {'F1 50/50':>9} {'F1 pond.':>9} {'HasAns':>8} {'NoAns':>8}")
    # tabla corta: del mejor 50/50 hasta un poco mas alla del mejor ponderado
    lo = min(best_plain["threshold"], best_w["threshold"]) - 1.0
    hi = max(best_plain["threshold"], best_w["threshold"]) + 1.0
    for s in sweep:
        if lo <= s["threshold"] <= hi:
            mark = "  <- mejor 50/50" if s is best_plain else ""
            mark += "  <- mejor ponderado" if s is best_w else ""
            print(f"{s['threshold']:>7} {s['overall']['F1']:>9} {s['w_f1']:>9.2f} "
                  f"{s['HasAns']['F1']:>8} {s['NoAns']['F1']:>8}{mark}")

    print(f"\nMejor umbral 50/50     : {best_plain['threshold']}  (F1={best_plain['overall']['F1']})")
    print(f"Mejor umbral ponderado : {best_w['threshold']}  (F1 ponderado={best_w['w_f1']:.2f})")

    if args.write:
        if args.threshold is None:
            raise SystemExit("--write requiere --threshold X (el umbral que elegiste; el maximo no se guarda solo).")
        chosen = next((x for x in sweep if abs(x["threshold"] - args.threshold) < 1e-9), None)
        if chosen is None:
            raise SystemExit(f"{args.threshold} no esta en el barrido de logs/threshold_sweep.json")
        CFG.write_text(
            json.dumps(
                {
                    "threshold": chosen["threshold"],
                    "criterion": f"valor elegido dentro de la meseta del F1 ponderado {round((1 - p) * 100)}/{round(p * 100)} (HasAns/NoAns)",
                    "threshold_max_f1_validation_50_50": best_plain["threshold"],
                    "threshold_max_weighted_f1": best_w["threshold"],
                    "weighted_f1": round(chosen["w_f1"], 2),
                    "rule": "null if (null_score - best_span_score) > threshold",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nGuardado umbral {chosen['threshold']} en {CFG}")
    else:
        print("\n(No se guardo nada. Agrega --write para guardar.)")


if __name__ == "__main__":
    main()
