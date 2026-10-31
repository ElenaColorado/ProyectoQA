from transformers import AutoTokenizer

# Modelo seleccionado para el proyecto
MODEL_NAME = "distilbert-base-uncased"

print("Descargando tokenizer de DistilBERT...")

# Descargar el tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

print("\n=== TOKENIZER ===")
print("Modelo:", MODEL_NAME)
print("Vocabulario:", tokenizer.vocab_size)
print("Fast tokenizer:", tokenizer.is_fast)

# Probar una oración
text = "The Transformer uses attention."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\nTexto:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

# Tokenizar una pregunta y su contexto
question = "What does the Transformer use?"
context = "The Transformer uses attention."

encoded = tokenizer(
    question,
    context,
    return_offsets_mapping=True,
    truncation="only_second",
    max_length=384
)

print("\n=== QUESTION ANSWERING ===")
print("Pregunta:", question)
print("Contexto:", context)

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

print("\nOffsets:")
print(encoded["offset_mapping"])

print("\nTipos de tokens:")
print(encoded.sequence_ids())

# Identificar la respuesta correcta
answer_text = "attention"

# Posiciones de caracteres
answer_start = context.index(answer_text)
answer_end = answer_start + len(answer_text)

# Obtener offsets y sequence IDs
offsets = encoded["offset_mapping"]
sequence_ids = encoded.sequence_ids()

# Buscar los tokens que pertenecen al contexto
context_tokens = [
    i for i, seq_id in enumerate(sequence_ids)
    if seq_id == 1
]

# Encontrar el token inicial
start_position = next(
    i for i in context_tokens
    if offsets[i][0] <= answer_start < offsets[i][1]
)

# Encontrar el token final
end_position = next(
    i for i in reversed(context_tokens)
    if offsets[i][0] < answer_end <= offsets[i][1]
)

print("\n=== POSICIONES DE LA RESPUESTA ===")
print("Respuesta:", answer_text)
print("Inicio en caracteres:", answer_start)
print("Final en caracteres:", answer_end)
print("Token inicial:", start_position)
print("Token final:", end_position)

print(
    "Tokens de la respuesta:",
    tokenizer.convert_ids_to_tokens(
        encoded["input_ids"][start_position:end_position + 1]
    )
)

# Ejemplo de una pregunta sin respuesta
question_no_answer = "Who invented the Transformer?"

encoded_no_answer = tokenizer(
    question_no_answer,
    context,
    return_offsets_mapping=True,
    truncation="only_second",
    max_length=384
)

# Identificar el token especial CLS
cls_index = encoded_no_answer["input_ids"].index(
    tokenizer.cls_token_id
)

# Etiquetas para una pregunta sin respuesta
start_position_no_answer = cls_index
end_position_no_answer = cls_index

print("\n=== PREGUNTA SIN RESPUESTA ===")
print("Pregunta:", question_no_answer)
print("Contexto:", context)
print("Respuesta esperada:", None)
print("Token CLS:", tokenizer.cls_token)
print("Posicion inicial:", start_position_no_answer)
print("Posicion final:", end_position_no_answer)