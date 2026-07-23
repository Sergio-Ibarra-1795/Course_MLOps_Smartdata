"""
Etapa 1 — Ingestion y limpieza de datos.

Entrada : data/Dataset Endeudamiento Crediticio.csv
Salida  : artifacts/data_clean.csv

Ejecutar: python pipeline/ingest.py
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import config as C

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | INGEST   | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def limpiar_na_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte 'NA' como string a NaN real y castea a numérico."""
    df = df.copy()
    for col in C.COLUMNAS_NA_STRING:
        if col in df.columns:
            antes = df[col].astype(str).str.upper().str.contains('NA', na=False).sum()
            df[col] = df[col].replace(['NA', 'na', 'Na', 'N/A'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            log.info('  %s: %d NA-strings convertidos a NaN', col, antes)
    return df


def winsorizar(df: pd.DataFrame) -> pd.DataFrame:
    """Clip a percentiles 5-95 para tratar outliers extremos."""
    df = df.copy()
    for col in C.COLUMNAS_WINSORIZAR:
        if col in df.columns and df[col].notna().any():
            lo = df[col].quantile(C.P_WINSOR_LOW)
            hi = df[col].quantile(C.P_WINSOR_HIGH)
            n  = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            log.info('  %s: %d outliers tratados [%.2f, %.2f]', col, n, lo, hi)
    return df


def imputar(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa nulos con mediana o moda según la variable."""
    df = df.copy()
    estrategias = {col: 'median' for col in C.COLUMNAS_REQUERIDAS
                   if col not in [C.TARGET, C.ID_COL]}
    estrategias['Nro_dependiente'] = 'mode'
    for col, strat in estrategias.items():
        if col in df.columns and df[col].isnull().sum() > 0:
            val = df[col].median() if strat == 'median' else df[col].mode()[0]
            #df[col].fillna(val, inplace=True)
            df.fillna({col:val}, inplace=True)
            log.info('  %s: %d nulos imputados con %s (%.4f)', col, df[col].isnull().sum()+1, strat, val)
    return df


def validar_salida(df: pd.DataFrame) -> None:
    """Valida que el DataFrame limpio cumpla condiciones mínimas.
    Lanza ValueError si falla — el pipeline no continúa.
    """
    errores = []
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        errores.append(f'NULOS: {nulos} valores nulos en data_clean')
    faltantes = set(C.COLUMNAS_REQUERIDAS) - set(df.columns)
    if faltantes:
        errores.append(f'COLUMNAS FALTANTES: {faltantes}')
    vals_target = set(df[C.TARGET].unique())
    if not vals_target.issubset({0, 1}):
        errores.append(f'TARGET no binario: {vals_target}')
    if len(df) < 500:
        errores.append(f'FILAS INSUFICIENTES: {len(df)} (mínimo 500)')
    if errores:
        raise ValueError('VALIDACIÓN ETAPA 1 FALLIDA:\n' + '\n'.join(errores))
    log.info('  ✓ Validación OK: %d filas | 0 nulos | target binario', len(df))


def run() -> pd.DataFrame:
    """Ejecuta la etapa completa de ingestion."""
    log.info('=== ETAPA 1: INGESTION ===')
    if not C.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset no encontrado: {C.RAW_DATA_PATH}')
    df = pd.read_csv(C.RAW_DATA_PATH, sep=';', decimal='.')
    log.info('Cargado: %d filas x %d columnas', *df.shape)
    df = limpiar_na_strings(df)
    df = winsorizar(df)
    df = imputar(df)
    validar_salida(df)
    df.to_csv(C.CLEAN_DATA_PATH, index=False)
    log.info('Artefacto guardado: %s', C.CLEAN_DATA_PATH)
    log.info('=== ETAPA 1 COMPLETADA ===')
    return df


if __name__ == '__main__':
    run()
