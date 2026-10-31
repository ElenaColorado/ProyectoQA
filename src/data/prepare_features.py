from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

# Cargar dataset y tokenizer
dataset = load_dataset("rajpurkar/squad_v2")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# Seleccionar un ejemplo con respuesta
answerable = dataset["train"][0]

# Seleccionar uno sin respuesta
unanswerable = next(
    item for item in dataset["train"]
    if len(item["answers"]["text"]) == 0
)

# Crear un ejemplo con contexto largo
long_context = (
    "The experiment includes several observations and measurements. " * 60
    + "The final result was 50,257 tokens."
)

long_answer = "50,257 tokens"

long_example = {
    "question": "What was the final result?",
    "context": long_context,
    "answers": {
        "text": [long_answer],
        "answer_start": [long_context.index(long_answer)]
    }
}

# Ahora utilizamos tres ejemplos
examples = [answerable, unanswerable, long_example]

# Tokenizar varios ejemplos simultáneamente
encoded = tokenizer(
    [item["question"].strip() for item in examples],
    [item["context"] for item in examples],
    truncation="only_second",
    max_length=MAX_LENGTH,
    stride=DOC_STRIDE,
    return_overflowing_tokens=True,
    return_offsets_mapping=True
)

# Relacionar cada ventana con su ejemplo original
sample_mapping = encoded["overflow_to_sample_mapping"]

print("\n=== PREPROCESAMIENTO AUTOMATICO ===")
print("Ejemplos originales:", len(examples))
print("Ventanas generadas:", len(encoded["input_ids"]))

for i, sample_index in enumerate(sample_mapping):
    print(f"\nVentana {i + 1}")
    print("Pertenece al ejemplo:", sample_index + 1)
    print("Pregunta:", examples[sample_index]["question"])
    print("Cantidad de tokens:", len(encoded["input_ids"][i]))

# Generar etiquetas automáticamente
start_positions = []
end_positions = []

print("\n=== ETIQUETAS AUTOMATICAS ===")

for i, sample_index in enumerate(sample_mapping):

    answers = examples[sample_index]["answers"]
    input_ids = encoded["input_ids"][i]
    offsets = encoded["offset_mapping"][i]
    sequence_ids = encoded.sequence_ids(i)

    cls_index = input_ids.index(tokenizer.cls_token_id)

    # Identificar tokens del contexto
    context_tokens = [
        j for j, seq_id in enumerate(sequence_ids)
        if seq_id == 1
    ]

    # Caso 1: pregunta sin respuesta
    if len(answers["text"]) == 0:
        start_position = cls_index
        end_position = cls_index

    else:
        # Caso 2: pregunta con respuesta
        answer_start = answers["answer_start"][0]
        answer_end = answer_start + len(answers["text"][0])

        window_start = offsets[context_tokens[0]][0]
        window_end = offsets[context_tokens[-1]][1]

        # Comprobar si la respuesta está en esta ventana
        if window_start <= answer_start and answer_end <= window_end:

            start_position = next(
                j for j in context_tokens
                if offsets[j][0] <= answer_start < offsets[j][1]
            )

            end_position = next(
                j for j in reversed(context_tokens)
                if offsets[j][0] < answer_end <= offsets[j][1]
            )

        else:
            # La respuesta no aparece en esta ventana
            start_position = cls_index
            end_position = cls_index

    start_positions.append(start_position)
    end_positions.append(end_position)

    print(f"\nVentana {i + 1}")
    print("Ejemplo original:", sample_index + 1)
    print("Start position:", start_position)
    print("End position:", end_position)

print("\nTotal de etiquetas generadas:", len(start_positions))