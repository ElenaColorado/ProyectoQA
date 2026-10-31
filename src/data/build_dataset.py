from datasets import load_dataset
from transformers import AutoTokenizer

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# Cargar SQuAD 2.0
print("Cargando SQuAD 2.0...")

dataset = load_dataset("rajpurkar/squad_v2")

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print("\n=== CONFIGURACIÓN ===")
print("Modelo:", MODEL_NAME)
print("Longitud máxima:", MAX_LENGTH)
print("Solapamiento:", DOC_STRIDE)
print("Preguntas de entrenamiento:", len(train_dataset))
print("Preguntas de validación:", len(validation_dataset))

# ==========================================
# FUNCION DE PREPROCESAMIENTO
# ==========================================

def prepare_train_features(examples):

    # Tokenizar preguntas y contextos
    tokenized = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    # Relacionar ventanas con ejemplos originales
    sample_mapping = tokenized.pop(
        "overflow_to_sample_mapping"
    )

    offset_mapping = tokenized.pop(
        "offset_mapping"
    )

    start_positions = []
    end_positions = []

    for i, sample_index in enumerate(sample_mapping):

        answers = examples["answers"][sample_index]
        input_ids = tokenized["input_ids"][i]
        offsets = offset_mapping[i]
        sequence_ids = tokenized.sequence_ids(i)

        cls_index = input_ids.index(
            tokenizer.cls_token_id
        )

        context_tokens = [
            j for j, seq_id in enumerate(sequence_ids)
            if seq_id == 1
        ]

        # Pregunta sin respuesta
        if len(answers["text"]) == 0:
            start_position = cls_index
            end_position = cls_index

        else:
            answer_start = answers["answer_start"][0]
            answer_end = (
                answer_start + len(answers["text"][0])
            )

            window_start = offsets[context_tokens[0]][0]
            window_end = offsets[context_tokens[-1]][1]

            # La respuesta completa esta en la ventana
            if (
                window_start <= answer_start
                and answer_end <= window_end
            ):

                start_position = next(
                    j for j in context_tokens
                    if offsets[j][0] <= answer_start < offsets[j][1]
                )

                end_position = next(
                    j for j in reversed(context_tokens)
                    if offsets[j][0] < answer_end <= offsets[j][1]
                )

            else:
                # Respuesta fuera de esta ventana
                start_position = cls_index
                end_position = cls_index

        start_positions.append(start_position)
        end_positions.append(end_position)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    
    # DistilBERT no utiliza token_type_ids
    tokenized.pop("token_type_ids", None)
    
    return tokenized

# ==========================================
# PRUEBA DE LA FUNCION
# ==========================================

print("\n=== PRUEBA AUTOMATICA ===")

# Seleccionar una pregunta con respuesta
answerable = train_dataset[0]

# Buscar una pregunta sin respuesta
unanswerable = next(
    item for item in train_dataset
    if len(item["answers"]["text"]) == 0
)

# Crear un lote con dos preguntas
test_examples = {
    "question": [
        answerable["question"],
        unanswerable["question"]
    ],
    "context": [
        answerable["context"],
        unanswerable["context"]
    ],
    "answers": [
        answerable["answers"],
        unanswerable["answers"]
    ]
}

# Ejecutar nuestra funcion
features = prepare_train_features(test_examples)

print("Ventanas generadas:", len(features["input_ids"]))
print("Posiciones iniciales:", features["start_positions"])
print("Posiciones finales:", features["end_positions"])

# Verificar que las etiquetas sean correctas
assert features["start_positions"] == [75, 0]
assert features["end_positions"] == [78, 0]

print("Prueba completada correctamente.")

# ==========================================
# PRUEBA AUTOMATICA CON CONTEXTO LARGO
# ==========================================

print("\n=== PRUEBA DE CONTEXTO LARGO ===")

# Crear un contexto largo
long_context = (
    "The experiment includes several observations and measurements. " * 60
    + "The final result was 50,257 tokens."
)

long_question = "What was the final result?"
long_answer = "50,257 tokens"

# Preparar un ejemplo con respuesta al final
long_examples = {
    "question": [long_question],
    "context": [long_context],
    "answers": [{
        "text": [long_answer],
        "answer_start": [long_context.index(long_answer)]
    }]
}

# Ejecutar nuestra funcion de preprocesamiento
long_features = prepare_train_features(long_examples)

print("Ventanas generadas:", len(long_features["input_ids"]))
print("Posiciones iniciales:", long_features["start_positions"])
print("Posiciones finales:", long_features["end_positions"])

# Comprobar la cantidad de ventanas
assert len(long_features["input_ids"]) == 2

# La primera ventana no contiene la respuesta
assert long_features["start_positions"][0] == 0
assert long_features["end_positions"][0] == 0

# La segunda ventana contiene la respuesta
start = long_features["start_positions"][1]
end = long_features["end_positions"][1]

assert start > 0
assert end >= start

# Obtener los offsets para verificar el texto recuperado
long_encoded = tokenizer(
    long_question,
    long_context,
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

offsets = long_encoded["offset_mapping"][1]

extracted = long_context[
    offsets[start][0]:offsets[end][1]
]

print("Respuesta original:", long_answer)
print("Respuesta recuperada:", extracted)

assert extracted == long_answer

print("Prueba de contexto largo completada correctamente.")

# ==========================================
# PRUEBA CON 20 EJEMPLOS REALES DE SQuAD 2.0
# ==========================================

print("\n=== PRUEBA CON 20 EJEMPLOS ===")

# Seleccionar 20 preguntas del dataset
sample_dataset = train_dataset.select(range(20))

# Aplicar el preprocesamiento automaticamente
processed_sample = sample_dataset.map(
    prepare_train_features,
    batched=True,
    batch_size=8,
    remove_columns=sample_dataset.column_names,
    desc="Procesando muestra de SQuAD 2.0"
)

print("Preguntas originales:", len(sample_dataset))
print("Ventanas procesadas:", len(processed_sample))
print("Columnas generadas:", processed_sample.column_names)

# Verificar las etiquetas y las longitudes
for example in processed_sample:

    num_tokens = len(example["input_ids"])

    assert num_tokens <= MAX_LENGTH

    assert len(example["attention_mask"]) == num_tokens

    assert 0 <= example["start_positions"] < num_tokens
    assert 0 <= example["end_positions"] < num_tokens

    assert example["start_positions"] <= example["end_positions"]

print("\nTodas las ventanas tienen etiquetas validas.")
print("Prueba de 20 ejemplos completada correctamente.")

# ==========================================
# PROCESAMIENTO COMPLETO DE TRAINING
# ==========================================

print("\n=== PROCESAMIENTO COMPLETO DE TRAINING ===")

# Procesar todas las preguntas de entrenamiento
processed_train = train_dataset.map(
    prepare_train_features,
    batched=True,
    batch_size=64,
    remove_columns=train_dataset.column_names,
    desc="Procesando SQuAD 2.0 Training"
)

# Mostrar resultados
print("\n=== RESULTADOS TRAINING ===")
print("Preguntas originales:", len(train_dataset))
print("Ventanas generadas:", len(processed_train))
print("Columnas:", processed_train.column_names)

# Verificaciones generales
assert len(processed_train) >= len(train_dataset)

assert all(
    column in processed_train.column_names
    for column in [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions"
    ]
)

print("Dataset de entrenamiento procesado correctamente.")

# ==========================================
# PROCESAMIENTO COMPLETO DE VALIDATION
# ==========================================

print("\n=== PROCESAMIENTO DE VALIDATION ===")

# Procesar todas las preguntas de validacion
processed_validation = validation_dataset.map(
    prepare_train_features,
    batched=True,
    batch_size=64,
    remove_columns=validation_dataset.column_names,
    desc="Procesando SQuAD 2.0 Validation"
)

# Mostrar resultados
print("\n=== RESULTADOS VALIDATION ===")
print("Preguntas originales:", len(validation_dataset))
print("Ventanas generadas:", len(processed_validation))
print("Columnas:", processed_validation.column_names)

# Verificar que se generaron las columnas necesarias
required_columns = [
    "input_ids",
    "attention_mask",
    "start_positions",
    "end_positions"
]

assert len(processed_validation) >= len(validation_dataset)

assert all(
    column in processed_validation.column_names
    for column in required_columns
)

# Verificar las etiquetas de todas las ventanas
for example in processed_validation:

    num_tokens = len(example["input_ids"])

    assert num_tokens <= MAX_LENGTH
    assert len(example["attention_mask"]) == num_tokens

    assert 0 <= example["start_positions"] < num_tokens
    assert 0 <= example["end_positions"] < num_tokens

    assert example["start_positions"] <= example["end_positions"]

print("\nDataset de validacion procesado correctamente.")

# ==========================================
# VALIDACION GENERAL DE LOS DATASETS
# ==========================================

print("\n=== VALIDACION FINAL DE DATOS ===")

def validate_features(data, name):

    total = len(data)
    no_answer = 0
    with_answer = 0

    for example in data:

        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]

        start = example["start_positions"]
        end = example["end_positions"]

        num_tokens = len(input_ids)

        # Verificar longitud de cada ventana
        assert 0 < num_tokens <= MAX_LENGTH
        assert len(attention_mask) == num_tokens

        # Verificar posiciones validas
        assert 0 <= start < num_tokens
        assert 0 <= end < num_tokens
        assert start <= end

        # Verificar casos sin respuesta
        if start == 0:
            assert end == 0
            no_answer += 1
        else:
            assert end > 0
            with_answer += 1

    print(f"\n{name}")
    print("Ventanas verificadas:", total)
    print("Con respuesta:", with_answer)
    print("Sin respuesta:", no_answer)
    print("Errores encontrados: 0")

    assert with_answer > 0
    assert no_answer > 0

    return True


# Validar entrenamiento
train_ok = validate_features(
    processed_train,
    "TRAINING"
)

# Validar validacion
validation_ok = validate_features(
    processed_validation,
    "VALIDATION"
)

print("\n=== RESULTADO GENERAL ===")
print("Training correcto:", train_ok)
print("Validation correcto:", validation_ok)
print("Validacion completada correctamente.")

# ==========================================
# GUARDAR DATASETS PROCESADOS
# ==========================================

from pathlib import Path

print("\n=== GUARDANDO DATASETS ===")

# Crear carpeta de salida
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Guardar conjunto de entrenamiento
processed_train.save_to_disk(
    str(output_dir / "train")
)

print("Training guardado correctamente.")

# Guardar conjunto de validacion
processed_validation.save_to_disk(
    str(output_dir / "validation")
)

print("Validation guardado correctamente.")

print("\n=== RESULTADO FINAL ===")
print("Ubicacion:", output_dir.resolve())
print("Training:", len(processed_train), "ventanas")
print("Validation:", len(processed_validation), "ventanas")
print("Datasets guardados exitosamente.")