from datasets import load_dataset
from transformers import AutoTokenizer

# Configuración del proyecto
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

# Cargar dataset y tokenizer
print("Cargando SQuAD 2.0...")

dataset = load_dataset("rajpurkar/squad_v2")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# Seleccionar un ejemplo real
example = dataset["train"][0]

print("\n=== EJEMPLO REAL ===")
print("Pregunta:", example["question"])
print("Respuesta:", example["answers"])

# Tokenizar la pregunta y el contexto
encoded = tokenizer(
    example["question"].strip(),
    example["context"],
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print("\n=== PREPROCESAMIENTO ===")
print("Ventanas generadas:", len(encoded["input_ids"]))
print("Tokens de la primera ventana:", len(encoded["input_ids"][0]))
print("Longitud máxima:", MAX_LENGTH)
print("Solapamiento:", DOC_STRIDE)

# Obtener la respuesta correcta
answer_text = example["answers"]["text"][0]
answer_start = example["answers"]["answer_start"][0]
answer_end = answer_start + len(answer_text)

# Obtener offsets y tipos de tokens de la primera ventana
offsets = encoded["offset_mapping"][0]
sequence_ids = encoded.sequence_ids(0)

# Identificar tokens pertenecientes al contexto
context_tokens = [
    i for i, seq_id in enumerate(sequence_ids)
    if seq_id == 1
]

# Encontrar el token donde empieza la respuesta
start_position = next(
    i for i in context_tokens
    if offsets[i][0] <= answer_start < offsets[i][1]
)

# Encontrar el token donde termina la respuesta
end_position = next(
    i for i in context_tokens
    if offsets[i][0] <= answer_end - 1 < offsets[i][1]
)

# Recuperar la respuesta utilizando las posiciones
extracted_answer = example["context"][
    offsets[start_position][0]:offsets[end_position][1]
]

print("\n=== ETIQUETAS DE ENTRENAMIENTO ===")
print("Respuesta original:", answer_text)
print("Token inicial:", start_position)
print("Token final:", end_position)
print("Respuesta recuperada:", extracted_answer)
print("Coinciden:", answer_text == extracted_answer)

# Buscar una pregunta real sin respuesta
example_no_answer = next(
    item for item in dataset["train"]
    if len(item["answers"]["text"]) == 0
)

# Tokenizar pregunta y contexto
encoded_no_answer = tokenizer(
    example_no_answer["question"].strip(),
    example_no_answer["context"],
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

# Encontrar el token CLS
cls_index = encoded_no_answer["input_ids"][0].index(
    tokenizer.cls_token_id
)

# Etiquetas para una pregunta sin respuesta
start_position = cls_index
end_position = cls_index

print("\n=== EJEMPLO REAL SIN RESPUESTA ===")
print("Pregunta:", example_no_answer["question"])
print("Respuestas:", example_no_answer["answers"])
print("Token CLS:", tokenizer.cls_token)
print("Token inicial:", start_position)
print("Token final:", end_position)
print("Cantidad de ventanas:", len(encoded_no_answer["input_ids"]))

# Crear un contexto largo de prueba
long_context = (
    "The experiment includes several observations and measurements. " * 60
    + "The final result was 50,257 tokens."
)

long_question = "What was the final result?"

# Tokenizar utilizando ventanas con solapamiento
encoded_long = tokenizer(
    long_question,
    long_context,
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

print("\n=== PRUEBA DE CONTEXTO LARGO ===")
print("Caracteres del contexto:", len(long_context))
print("Ventanas generadas:", len(encoded_long["input_ids"]))

for i, window in enumerate(encoded_long["input_ids"]):
    print(f"Ventana {i + 1}: {len(window)} tokens")

print(
    "Se generaron varias ventanas:",
    len(encoded_long["input_ids"]) > 1
)

# Buscar una respuesta al final del contexto largo
long_answer = "50,257 tokens"

answer_start = long_context.index(long_answer)
answer_end = answer_start + len(long_answer)

print("\n=== RESPUESTA EN CONTEXTO LARGO ===")

found = False

for i in range(len(encoded_long["input_ids"])):

    offsets = encoded_long["offset_mapping"][i]
    sequence_ids = encoded_long.sequence_ids(i)

    context_tokens = [
        j for j, seq_id in enumerate(sequence_ids)
        if seq_id == 1
    ]

    # Verificar si la respuesta completa está en esta ventana
    window_start = offsets[context_tokens[0]][0]
    window_end = offsets[context_tokens[-1]][1]

    if window_start <= answer_start and answer_end <= window_end:

        start_token = next(
            j for j in context_tokens
            if offsets[j][0] <= answer_start < offsets[j][1]
        )

        end_token = next(
            j for j in context_tokens
            if offsets[j][0] < answer_end <= offsets[j][1]
        )

        extracted = long_context[
            offsets[start_token][0]:offsets[end_token][1]
        ]

        print("Respuesta original:", long_answer)
        print("Ventana que contiene la respuesta:", i + 1)
        print("Token inicial:", start_token)
        print("Token final:", end_token)
        print("Respuesta recuperada:", extracted)
        print("Coinciden:", extracted == long_answer)

        found = True

print("Respuesta encontrada:", found)

# Preparar etiquetas para cada ventana
print("\n=== ETIQUETAS DE TODAS LAS VENTANAS ===")

for i in range(len(encoded_long["input_ids"])):

    input_ids = encoded_long["input_ids"][i]
    offsets = encoded_long["offset_mapping"][i]
    sequence_ids = encoded_long.sequence_ids(i)

    cls_index = input_ids.index(tokenizer.cls_token_id)

    context_tokens = [
        j for j, seq_id in enumerate(sequence_ids)
        if seq_id == 1
    ]

    window_start = offsets[context_tokens[0]][0]
    window_end = offsets[context_tokens[-1]][1]

    # Comprobar si la respuesta completa está en la ventana
    if window_start <= answer_start and answer_end <= window_end:

        start_position = next(
            j for j in context_tokens
            if offsets[j][0] <= answer_start < offsets[j][1]
        )

        end_position = next(
            j for j in context_tokens
            if offsets[j][0] < answer_end <= offsets[j][1]
        )

    else:
        # Esta ventana no contiene la respuesta completa
        start_position = cls_index
        end_position = cls_index

    print(f"\nVentana {i + 1}")
    print("Start position:", start_position)
    print("End position:", end_position)