# Acta de Entrega y Relevo: Integrante 2 ➔ Integrante 3

**Etapa:** 2 — Fine-tuning, Validación y Curvas de Pérdida  
**Responsable:** Integrante 2  
**Fecha:** 1 de Noviembre de 2026  
**Hardware Utilizado:** NVIDIA GeForce RTX 3050 Laptop GPU (4.00 GB VRAM), CUDA con precisión mixta (FP16)  

---

## 1. Resumen de Ejecución y Logros
- Se tomó como punto de partida el checkpoint **BASE oficial** `distilbert/distilbert-base-uncased` (cumpliendo con la restricción de no usar modelos ya entrenados en SQuAD).
- Se ejecutó el fine-tuning **completo** utilizando las **131,754 ventanas de entrenamiento** y **12,134 ventanas de validación** de SQuAD 2.0.
- El modelo se entrenó durante **2 épocas** (16,470 pasos de optimización con tamaño de lote efectivo de 16).
- La pérdida de entrenamiento (*train loss*) descendió sostenidamente desde **3.30** hasta **0.83**, y la pérdida de validación (*eval loss*) alcanzó su valor óptimo de **1.2619**.
- Se exportaron los pesos finales y el tokenizador en formato estándar Hugging Face (`models/qa-final/`).
- Se verificó que el modelo y tokenizador pueden cargarse **100% fuera de línea** (*offline*) sin conexión a internet.

---

## 2. Archivos Creados y Modificados
```text
ProyectoQA/
├── configs/
│   └── training_config.json        # Hiperparámetros reales del entrenamiento
├── src/
│   └── training/
│       └── train.py                # Script principal de entrenamiento y exportación
├── tests/
│   └── test_training.py            # Pruebas unitarias de configuración y forward pass
├── logs/
│   ├── loss.png                    # Gráfica de curvas de entrenamiento y validación
│   ├── training_history.csv        # Registro paso a paso de la pérdida
│   ├── validation_metrics.json     # Métricas consolidadas del entrenamiento completo
│   └── pilot/                      # Respaldo del entrenamiento piloto (20K muestras)
│       ├── loss_pilot.png
│       ├── training_history_pilot.csv
│       └── validation_metrics_pilot.json
└── models/
    └── qa-final/                   # Pesos y tokenizador (EXCLUIDOS de Git)
        ├── config.json
        ├── model.safetensors       # 253 MB
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_args.bin
```

---

## 3. Métricas Reales Obtenidas

### Entrenamiento Completo (resultado final)
| Métrica | Valor Real |
| :--- | :--- |
| **Modelo Base** | `distilbert/distilbert-base-uncased` |
| **Parámetros Totales** | 66,364,418 |
| **Muestras Train** | 131,754 |
| **Muestras Validation** | 12,134 |
| **Épocas** | 2 |
| **Learning Rate** | 3 × 10⁻⁵ |
| **Batch Size Efectivo** | 16 (8 por dispositivo × 2 acumulación de gradiente) |
| **Tiempo de Entrenamiento** | 3,434.73 segundos (~57 minutos y 15 segundos) |
| **Train Loss Final** | **1.2074** |
| **Eval Loss Óptimo** | **1.2619** |
| **Total FLOPs** | 1.83 × 10¹⁶ |

### Comparación: Piloto vs Completo
| Métrica | Piloto (20K) | Completo (131K) | Mejora |
| :--- | :--- | :--- | :--- |
| Train Loss | 1.5573 | 1.2074 | ↓ 22.5% |
| Eval Loss | 1.7067 | 1.2619 | ↓ 26.1% |
| Tiempo | 541s (~9 min) | 3,435s (~57 min) | 6.3× más datos |
| Gap Train-Eval | 0.1494 | 0.0545 | Mejor generalización |

---

## 4. Comandos para Reproducir y Verificar

### Ejecutar Pruebas Unitarias
```powershell
pytest tests/test_training.py
```
*Resultado esperado:* 2 passed.

### Reproducir Entrenamiento Completo
```powershell
python src/training/train.py
```
*Nota:* Requiere GPU CUDA con al menos 4 GB VRAM. Tiempo estimado: ~57 minutos.

### Probar Carga Offline del Modelo Final (Para Integrante 3)
```powershell
python -c "from transformers import AutoModelForQuestionAnswering, AutoTokenizer; model = AutoModelForQuestionAnswering.from_pretrained('models/qa-final', local_files_only=True); tokenizer = AutoTokenizer.from_pretrained('models/qa-final', local_files_only=True); print('Modelo y tokenizador cargados OK offline')"
```

---

## 5. Instrucciones para el Integrante 3 (Inferencia y Calibración)
1. **Acceso a los pesos:** La carpeta `models/qa-final/` contiene los archivos del modelo. Si se trabaja en máquinas separadas, transferir esta carpeta por almacenamiento compartido (Drive/OneDrive), sin subirla a Git.
2. **Entrada de inferencia:** Construir la función `predict(context: str, question: str) -> dict` cargando el modelo desde `models/qa-final/`.
3. **Calibración de umbral (`threshold`):** Utilizar las puntuaciones de no-respuesta sobre el conjunto de validación para determinar cuándo responder `{"answer": null, "start": null}` frente a una respuesta extraída.
4. **Validación CPU:** Medir la latencia de inferencia en CPU para asegurar que cumpla con los límites de tiempo (< 3 segundos promedio por pregunta).
5. **Métricas de referencia:** El eval loss de este modelo es **1.2619** — usar como baseline para validar que la inferencia reproduce resultados coherentes.
