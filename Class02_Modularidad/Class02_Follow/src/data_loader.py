"""
data_loader.py — Carga y validación del dataset de riesgo crediticio Banco Wiesse.
"""
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

COLUMNAS_REQUERIDAS = {
    'Default', 'Prct_uso_tc', 'Prct_deuda_vs_ingresos',
    'Mto_ingreso_mensual', 'Nro_dependiente', 'Edad',
    'Nro_prestao_retrasados', 'Nro_retraso_60dias',
    'Nro_prod_financieros_deuda', 'Nro_creditos_hipotecarios',
    'Nro_retraso_ultm3anios',
}

def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV de endeudamiento crediticio del Banco Wiesse.

    Args:
        ruta: Ruta al archivo CSV (sep=';', decimal='.').
    Returns:
        DataFrame crudo con los datos originales.
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas.
    """
    path = Path(ruta)
    if not path.exists():
        raise FileNotFoundError(f'Archivo no encontrado: {ruta}')
    logger.info('Cargando datos desde %s', ruta)
    df = pd.read_csv(path, sep=';', decimal='.')
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f'Columnas faltantes en el dataset: {faltantes}')
    logger.info('Dataset cargado: %d filas x %d columnas', *df.shape)
    logger.info('Distribucion Default: %s', df['Default'].value_counts().to_dict())
    return df