"""
Etapa 2 — Feature engineering y selección de variables.

Entrada : artifacts/data_clean.csv
Salida  : artifacts/features.csv

Ejecutar: python pipeline/features.py
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import config as C

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | FEATURES | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones de feature engineering."""
    df = df.copy()
    df['Score_retrasos'] = sum(
        df[col] * peso for col, peso in C.PESOS_RETRASOS.items() if col in df.columns)
    df['Edad_cat'] = pd.cut(df['Edad'], bins=[0,25,35,45,55,65,100],
        labels=['<25','25-35','35-45','45-55','55-65','>65'], include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df['Edad_cat'], prefix='Edad')], axis=1)
    df['Deps_cat'] = pd.cut(df['Nro_dependiente'], bins=[-1,0,2,4,10],
        labels=['0','1-2','3-4','5+'], include_lowest=True)
    df = pd.concat([df, pd.get_dummies(df['Deps_cat'], prefix='Deps')], axis=1)
    scaler = StandardScaler()
    for col in ['Prct_uso_tc','Prct_deuda_vs_ingresos','Mto_ingreso_mensual',
                'Edad','Nro_prestao_retrasados','Score_retrasos']:
        if col in df.columns and df[col].std() > 0:
            df[f'{col}_std'] = scaler.fit_transform(df[[col]]).flatten()
    log.info('Features creadas: %d columnas totales', len(df.columns))
    return df


def seleccionar_top_k(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona las K mejores features con SelectKBest."""
    excluir = [C.TARGET, C.ID_COL, 'Edad_cat', 'Deps_cat']
    X = df.drop(columns=[c for c in excluir if c in df.columns]).select_dtypes(include=['number'])
    y = df[C.TARGET]
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    top_k = pd.Series(selector.scores_, index=X.columns).nlargest(C.K_FEATURES).index.tolist()
    log.info('Top-%d features: %s...', C.K_FEATURES, top_k[:3])
    cols = [c for c in [C.ID_COL, C.TARGET] + top_k if c in df.columns]
    return df[cols]


def validar_salida(df: pd.DataFrame) -> None:
    """Valida que el dataset de features esté listo para modelado."""
    errores = []
    if df.isnull().sum().sum() > 0:
        errores.append(f'NULOS POST-FEATURES: {df.isnull().sum().sum()}')
    if C.TARGET not in df.columns:
        errores.append('TARGET ausente en features')
    if len(df.columns) < C.K_FEATURES + 1:
        errores.append(f'POCAS COLUMNAS: {len(df.columns)}')
    if errores:
        raise ValueError('VALIDACIÓN ETAPA 2 FALLIDA:\n' + '\n'.join(errores))
    log.info('  ✓ Validación OK: %d filas | %d columnas', *df.shape)


def run() -> pd.DataFrame:
    """Ejecuta la etapa de feature engineering."""
    log.info('=== ETAPA 2: FEATURES ===')
    if not C.CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f'Etapa 1 no completada: {C.CLEAN_DATA_PATH}')
    df = pd.read_csv(C.CLEAN_DATA_PATH)
    log.info('Cargado: %d filas x %d columnas', *df.shape)
    df = crear_features(df)
    df = seleccionar_top_k(df)
    validar_salida(df)
    df.to_csv(C.FEATURES_PATH, index=False)
    log.info('Artefacto guardado: %s', C.FEATURES_PATH)
    log.info('=== ETAPA 2 COMPLETADA ===')
    return df


if __name__ == '__main__':
    run()
