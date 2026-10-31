from datasets import load_dataset

# Descargar el dataset SQuAD 2.0
print("Descargando SQuAD 2.0...")

dataset = load_dataset("rajpurkar/squad_v2")

# Separar los conjuntos
train = dataset["train"]
validation = dataset["validation"]

# Mostrar información
print("\n=== DATASET SQuAD 2.0 ===")
print("Preguntas de entrenamiento:", len(train))
print("Preguntas de validación:", len(validation))

# Mostrar un ejemplo
example = train[0]

print("\n=== EJEMPLO ===")
print("Pregunta:", example["question"])
print("Contexto:", example["context"])
print("Respuesta:", example["answers"])

# Verificar la posicion exacta de la respuesta
answer_text = example["answers"]["text"][0]
start = example["answers"]["answer_start"][0]

extracted = example["context"][start:start + len(answer_text)]

print("\n=== VERIFICACION ===")
print("Respuesta original:", answer_text)
print("Respuesta extraida:", extracted)
print("Coinciden:", answer_text == extracted)

# Buscar una pregunta sin respuesta
unanswerable = next(
    item for item in train
    if len(item["answers"]["text"]) == 0
)

print("\n=== PREGUNTA SIN RESPUESTA ===")
print("Pregunta:", unanswerable["question"])
print("Contexto:", unanswerable["context"])
print("Respuestas:", unanswerable["answers"])

# Comprobar si tiene respuesta
has_answer = len(unanswerable["answers"]["text"]) > 0

print("Tiene respuesta:", has_answer)