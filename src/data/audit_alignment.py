from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

print("=== AUDITORIA DE RESPUESTAS ===")

# Cargar 20 preguntas originales
original = load_dataset(
    "rajpurkar/squad_v2",
    split="train[:20]"
)

# Cargar los datos que preparamos
processed = load_from_disk("data/processed/train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

verified = 0

for i, example in enumerate(original):

    # Tokenizar nuevamente para recuperar offsets
    encoded = tokenizer(
        example["question"].strip(),
        example["context"],
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True
    )

    saved = processed[i]

    # Comprobar que los tokens coincidan
    assert encoded["input_ids"] == saved["input_ids"]

    start = saved["start_positions"]
    end = saved["end_positions"]

    if example["answers"]["text"]:

        # Respuesta correcta original
        answer = example["answers"]["text"][0]
        char_start = example["answers"]["answer_start"][0]
        char_end = char_start + len(answer)

        offsets = encoded["offset_mapping"]

        # Comprobar que los tokens cubren la respuesta
        assert start > 0
        assert end >= start

        assert offsets[start][0] <= char_start < offsets[start][1]
        assert offsets[end][0] < char_end <= offsets[end][1]

    else:
        # Pregunta sin respuesta
        assert start == 0
        assert end == 0

    verified += 1

print("Ejemplos verificados:", verified)
print("Errores de alineacion: 0")
print("Auditoria completada correctamente.")