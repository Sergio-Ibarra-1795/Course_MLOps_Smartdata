"""
config.py — Configuración centralizada del pipeline MLOps Banco Wiesse.

ÚNICO lugar donde se definen parámetros.
Todas las etapas del pipeline importan desde aquí.

Uso:
    import config as C
    df = pd.read_csv(C.RAW_DATA_PATH, sep=';')
"""
import os
from pathlib import Path

# ── Directorios ──────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / 'data'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
MODELS_DIR    = ROOT_DIR / 'models'
PIPELINE_DIR  = ROOT_DIR / 'pipeline'

# ── Archivos ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = DATA_DIR      / 'df.csv'
CLEAN_DATA_PATH = ARTIFACTS_DIR / 'data_clean.csv'
FEATURES_PATH   = ARTIFACTS_DIR / 'features.csv'
MODEL_PATH      = ARTIFACTS_DIR / 'modelo.pkl'
METRICS_PATH    = ARTIFACTS_DIR / 'metrics.json'
REPORT_PATH     = ARTIFACTS_DIR / 'reporte_evaluacion.png'
X_TEST_PATH     = ARTIFACTS_DIR / 'X_test.csv'
Y_TEST_PATH     = ARTIFACTS_DIR / 'y_test.csv'

# ── Columnas ─────────────────────────────────────────────────────────────────
TARGET     = 'Default'
ID_COL     = 'ID'
COLUMNAS_REQUERIDAS = [
    'Default', 'Prct_uso_tc', 'Prct_deuda_vs_ingresos',
    'Mto_ingreso_mensual', 'Nro_dependiente', 'Edad',
    'Nro_prestao_retrasados', 'Nro_retraso_60dias',
    'Nro_prod_financieros_deuda', 'Nro_creditos_hipotecarios',
    'Nro_retraso_ultm3anios',
]

# ── Preprocesamiento ──────────────────────────────────────────────────────────
COLUMNAS_NA_STRING  = ['Mto_ingreso_mensual', 'Nro_dependiente']
COLUMNAS_WINSORIZAR = ['Prct_uso_tc', 'Prct_deuda_vs_ingresos', 'Mto_ingreso_mensual']
P_WINSOR_LOW        = 0.05
P_WINSOR_HIGH       = 0.95

# ── Features ──────────────────────────────────────────────────────────────────
K_FEATURES     = 15
PESOS_RETRASOS = {
    'Nro_prestao_retrasados': 3,
    'Nro_retraso_60dias':     5,
    'Nro_retraso_ultm3anios': 2,
}

# ── Entrenamiento ─────────────────────────────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
TECNICA_BALANCEO = 'smote'    # 'smote' | 'undersampling' | 'oversampling'

# ── Evaluación ────────────────────────────────────────────────────────────────
RECALL_MINIMO = 0.480    # pipeline falla si recall < este valor
COSTO_FN      = 10_000  # USD por falso negativo (default no detectado)
COSTO_FP      = 1_000   # USD por falso positivo (cliente bueno rechazado)

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = 'banco-wiesse-pipeline'
MLFLOW_RUN_NAME   = os.getenv('PIPELINE_VERSION', 'run-local')

# ── Crear directorios ─────────────────────────────────────────────────────────
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
