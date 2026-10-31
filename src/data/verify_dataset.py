from datasets import load_from_disk
from pathlib import Path

# Carpeta donde guardamos los datos
DATA_DIR = Path("data/processed")

print("=== VERIFICACION DE DATASETS ===")

# Cargar los datasets desde el disco
train = load_from_disk(str(DATA_DIR / "train"))
validation = load_from_disk(str(DATA_DIR / "validation"))

print("\nTraining:", len(train), "ventanas")
print("Validation:", len(validation), "ventanas")

# Verificar las cantidades
assert len(train) == 131754
assert len(validation) == 12134

# Verificar las columnas
required_columns = [
    "input_ids",
    "attention_mask",
    "start_positions",
    "end_positions"
]

assert train.column_names == required_columns
assert validation.column_names == required_columns

# Verificar que podemos acceder a una muestra
example = train[0]

print("\n=== PRIMER EJEMPLO ===")
print("Cantidad de tokens:", len(example["input_ids"]))
print("Start position:", example["start_positions"])
print("End position:", example["end_positions"])

print("\nDatasets cargados correctamente.")