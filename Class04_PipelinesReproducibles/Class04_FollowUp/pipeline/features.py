"""
pipeline/features.py — Etapa 2: Feature Engineering y Selección.

Entrada : artifacts/data_clean.csv
Salida  : artifacts/features.csv

Ejecutar desde la raíz: python pipeline/features.py
"""
import sys
import logging
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as C
from src.features import (
    crear_score_retrasos,
    crear_categorias_edad,
    crear_categorias_dependientes,
    estandarizar_variables,
    seleccionar_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | FEATURES | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def run():
    log.info('=== ETAPA 2: FEATURE ENGINEERING ===')
    
    if not C.CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f'Falta el artefacto de la Etapa 1: {C.CLEAN_DATA_PATH}')
        
    df = pd.read_csv(C.CLEAN_DATA_PATH)
    log.info('Cargado: %d filas x %d columnas', *df.shape)
    
    # 1. Transformaciones de ingeniería de variables usando src/features.py
    df = crear_score_retrasos(df)
    df = crear_categorias_edad(df)
    df = crear_categorias_dependientes(df)
    df = estandarizar_variables(df)
    
    # 2. Filtrado y Selección de variables (SelectKBest)
    cols_excluir = [C.TARGET, C.ID_COL, 'Edad_cat', 'Deps_cat']
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    X = X.select_dtypes(include=['number'])
    y = df[C.TARGET]
    
    top_k_cols = seleccionar_features(X, y, k=C.K_FEATURES)
    
    # 3. Preservar ID y Target junto con las mejores variables
    cols_finales = [c for c in [C.ID_COL, C.TARGET] if c in df.columns] + top_k_cols
    df_final = df[cols_finales]
    
    # 4. Guardar artefacto en disco
    df_final.to_csv(C.FEATURES_PATH, index=False)
    log.info('✓ Artefacto guardado exitosamente en: %s', C.FEATURES_PATH)
    log.info('=== ETAPA 2 COMPLETADA ===')
    return df_final


if __name__ == '__main__':
    run()