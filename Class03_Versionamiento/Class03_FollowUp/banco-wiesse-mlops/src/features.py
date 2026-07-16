"""
features.py — Feature engineering del proyecto Banco Wiesse.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

PESOS_RETRASOS = {
    'Nro_prestao_retrasados': 3,
    'Nro_retraso_60dias':     5,
    'Nro_retraso_ultm3anios': 2,
}

COLUMNAS_ESCALAR = [
    'Prct_uso_tc', 'Prct_deuda_vs_ingresos', 'Mto_ingreso_mensual',
    'Edad', 'Nro_prestao_retrasados', 'Score_retrasos',
]

def crear_score_retrasos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Score_retrasos'] = sum(
        df[col] * peso for col, peso in PESOS_RETRASOS.items()
        if col in df.columns
    )
    logger.info('Score_retrasos: min=%.1f max=%.1f',
                df['Score_retrasos'].min(), df['Score_retrasos'].max())
    return df

def crear_categorias_edad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Edad_cat'] = pd.cut(
        df['Edad'], bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'],
        include_lowest=True,
    )
    dummies = pd.get_dummies(df['Edad_cat'], prefix='Edad')
    df = pd.concat([df, dummies], axis=1)
    return df

def crear_categorias_dependientes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Deps_cat'] = pd.cut(
        df['Nro_dependiente'], bins=[-1, 0, 2, 4, 10],
        labels=['0', '1-2', '3-4', '5+'], include_lowest=True,
    )
    dummies = pd.get_dummies(df['Deps_cat'], prefix='Deps')
    df = pd.concat([df, dummies], axis=1)
    return df

def estandarizar_variables(df: pd.DataFrame, columnas=None) -> pd.DataFrame:
    df = df.copy()
    cols = columnas or COLUMNAS_ESCALAR
    scaler = StandardScaler()
    for col in cols:
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_std'] = scaler.fit_transform(df[[col]]).flatten()
    return df

def seleccionar_features(X: pd.DataFrame, y: pd.Series, k=15) -> list:
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns)
    top_k = scores.nlargest(k).index.tolist()
    logger.info('Top %d features: %s', k, top_k[:5])
    return top_k
