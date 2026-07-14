"""preprocessing.py — Limpieza, winsorization, imputacion y balanceo."""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)
COLUMNAS_NA_STRING  = ['Mto_ingreso_mensual', 'Nro_dependiente']
COLUMNAS_WINSORIZAR = ['Prct_uso_tc', 'Prct_deuda_vs_ingresos', 'Mto_ingreso_mensual']

def limpiar_na_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte valores "NA" como string a NaN (Parte 2.1 del notebook)."""
    df = df.copy()
    for col in COLUMNAS_NA_STRING:
        if col in df.columns:
            antes = df[col].astype(str).str.upper().str.contains('NA', na=False).sum()
            df[col] = df[col].replace(['NA', 'na', 'Na', 'N/A'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info('%s: %d valores NA-string convertidos a NaN', col, antes)
    return df

def winsorizar_columnas(df, columnas=None, p_low=0.05, p_high=0.95):
    """Aplica winsorization a percentiles definidos (Parte 2.2 del notebook)."""
    df = df.copy()
    for col in (columnas or COLUMNAS_WINSORIZAR):
        if col in df.columns and df[col].notna().any():
            lower, upper = df[col].quantile(p_low), df[col].quantile(p_high)
            n = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info('%s: %d outliers tratados [%.2f, %.2f]', col, n, lower, upper)
    return df

def imputar_nulos(df, estrategias=None):
    """Imputa valores nulos según estrategia por columna (Parte 2.3 del notebook)."""
    df = df.copy()
    defaults = {
        'Mto_ingreso_mensual': 'median', 'Nro_dependiente': 'mode',
        'Prct_deuda_vs_ingresos': 'median', 'Prct_uso_tc': 'median',
        'Edad': 'median', 'Nro_prestao_retrasados': 'median',
        'Nro_prod_financieros_deuda': 'median', 'Nro_retraso_60dias': 'median',
        'Nro_creditos_hipotecarios': 'median', 'Nro_retraso_ultm3anios': 'median',
    }
    for col, strategy in (estrategias or defaults).items():
        if col in df.columns and df[col].isnull().sum() > 0:
            n = df[col].isnull().sum()
            val = (df[col].median() if strategy == 'median' else
                   df[col].mean()   if strategy == 'mean' else df[col].mode()[0])
            df.fillna({col:val}, inplace=True)
            logger.info('%s: %d nulos imputados con %s (%.4f)', col, n, strategy, val)
    return df

def dividir_y_balancear(X, y, test_size=0.2, tecnica='smote', random_state=42):
    """División estratificada y balanceo. tecnica: 'smote'|'undersampling'|'oversampling'."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    logger.info('Split: train=%d test=%d', len(y_train), len(y_test))
    sampler_map = {
        'smote':        SMOTE(random_state=random_state),
        'undersampling': RandomUnderSampler(random_state=random_state),
        'oversampling':  RandomOverSampler(random_state=random_state),
    }
    sampler = sampler_map.get(tecnica, SMOTE(random_state=random_state))
    X_bal, y_bal = sampler.fit_resample(X_train, y_train)
    logger.info('Balanceo %s: antes=%d despues=%d', tecnica, len(y_train), len(y_bal))
    return X_bal, X_test, y_bal, y_test