import os
from pathlib import Path

# Determina la ruta absoluta de la raíz del proyecto (donde vive config.py)
BASE_DIR = Path(__file__).resolve().parent

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de artefactos
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Archivos específicos
DATASET_PATH = RAW_DATA_DIR / "Clasificacion_picos_intensidad.csv"
BASE_DATA_PKL = PROCESSED_DATA_DIR / "base_data.pkl"
RUN_IDS_TXT = METADATA_DIR / "run_ids.txt"

# Crear carpetas en caso de que no existan
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METADATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

    