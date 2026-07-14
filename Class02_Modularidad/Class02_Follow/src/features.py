"""features.py — Feature engineering extraido de la Parte 2 del notebook."""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

# Pesos del notebook original: Parte 2.4.3
PESOS_RETRASOS = {
    'Nro_prestao_retrasados': 3,   # retraso >3 meses
    'Nro_retraso_60dias':     5,   # retraso >60 dias
    'Nro_retraso_ultm3anios': 2,   # retraso >30 dias
}
COLUMNAS_ESCALAR = [
    'Prct_uso_tc', 'Prct_deuda_vs_ingresos', 'Mto_ingreso_mensual',
    'Edad', 'Nro_prestao_retrasados', 'Score_retrasos',
]

def crear_score_retrasos(df: pd.DataFrame) -> pd.DataFrame:
    """Score compuesto de retrasos (Parte 2.4.3 del notebook)."""
    df = df.copy()
    df['Score_retrasos'] = sum(
        df[col] * peso for col, peso in PESOS_RETRASOS.items() if col in df.columns)
    logger.info('Score_retrasos: min=%.1f max=%.1f',
                df['Score_retrasos'].min(), df['Score_retrasos'].max())
    return df

def crear_categorias_edad(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa Edad en 6 rangos y crea dummies (Parte 2.4.1 del notebook)."""
    df = df.copy()
    df['Edad_cat'] = pd.cut(df['Edad'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65'],
        include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df['Edad_cat'], prefix='Edad')], axis=1)
    logger.info('Edad_cat: %s', df['Edad_cat'].value_counts().to_dict())
    return df

def crear_categorias_dependientes(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa Nro_dependiente en 4 rangos y crea dummies (Parte 2.4.2)."""
    df = df.copy()
    df['Deps_cat'] = pd.cut(df['Nro_dependiente'],
        bins=[-1, 0, 2, 4, 10], labels=['0', '1-2', '3-4', '5+'],
        include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df['Deps_cat'], prefix='Deps')], axis=1)
    return df

def estandarizar_variables(df: pd.DataFrame, columnas=None) -> pd.DataFrame:
    """Estandariza con StandardScaler (Parte 2.5 del notebook)."""
    df = df.copy()
    scaler = StandardScaler()
    for col in (columnas or COLUMNAS_ESCALAR):
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_std'] = scaler.fit_transform(df[[col]]).flatten()
    return df

def seleccionar_features(X: pd.DataFrame, y: pd.Series, k: int = 15) -> list:
    """SelectKBest con f_classif (Parte 2.6 del notebook)."""
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    top_k = pd.Series(selector.scores_, index=X.columns).nlargest(k).index.tolist()
    logger.info('Top %d features: %s', k, top_k[:5])
    return top_k