"""
pipeline/ingest.py — Etapa 1: Ingesta y Limpieza de datos.

Entrada : data/df.csv (o la ruta configurada en config.py)
Salida  : artifacts/data_clean.csv

Ejecutar desde la raíz: python pipeline/ingest.py
"""
import sys
import logging
from pathlib import Path

# Garantizar que Python encuentre el módulo src/ desde la raíz
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as C
from src.data_loader import cargar_datos
from src.preprocessing import (
    limpiar_na_strings,
    winsorizar_columnas,
    imputar_nulos
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | INGEST    | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def run():
    log.info('=== ETAPA 1: INGESTION Y LIMPIEZA ===')
    
    # 1. Cargar dataset crudo usando src/data_loader.py
    df = cargar_datos(str(C.RAW_DATA_PATH))
    
    # 2. Aplicar limpieza reutilizando src/preprocessing.py
    df = limpiar_na_strings(df)
    df = winsorizar_columnas(df)
    df = imputar_nulos(df)
    
    # 3. Guardar el artefacto resultante en disco
    C.CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_DATA_PATH, index=False)
    
    log.info('✓ Artefacto guardado exitosamente en: %s', C.CLEAN_DATA_PATH)
    log.info('=== ETAPA 1 COMPLETADA ===')
    return df


if __name__ == '__main__':
    run()