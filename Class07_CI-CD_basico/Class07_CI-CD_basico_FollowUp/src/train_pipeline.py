"""train_pipeline.py — Entrena el modelo y guarda artefactos."""

import json
import logging
import pickle
import sys
from pathlib import Path

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y PATHS DE PYTHON
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

# Aseguramos que 'src/' esté en el path de Python para importaciones limpias
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from generate_data import DATA_PATH, generate  # noqa: E402

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"

# ==============================================================================
# PARÁMETROS Y CONFIGURACIÓN
# ==============================================================================
FEATURES = [
    "Presion",
    "Tonelaje",
    "Velocidad",
    "%Solidos",
    "Potencia",
    "F80",
    "Brazo",
]
TARGET = "picos_intens"
RANDOM_STATE = 123
TEST_SIZE = 0.30
RECALL_MIN = 0.70  # Umbral para el Quality Gate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# FUNCIONES PRINCIPALES
# ==============================================================================
def load_data() -> pd.DataFrame:
    """Carga el dataset CSV o lo genera si no existe."""
    if not DATA_PATH.exists():
        log.info("Dataset no encontrado. Generando dataset sintético...")
        generate()
    return pd.read_csv(DATA_PATH)


def train(df: pd.DataFrame) -> dict:
    """Entrena con GridSearchCV + SMOTE y guarda los artefactos."""
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Balanceo de clases en entrenamiento
    smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
    X_tr_s, y_tr_s = smote.fit_resample(X_train, y_train)

    # Búsqueda de hiperparámetros
    param_grid = {
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid,
        cv=StratifiedKFold(5),
        scoring="f1",
        n_jobs=-1,
    )
    gs.fit(X_tr_s, y_tr_s)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metricas = {
        "f1": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "params": gs.best_params_,
        "recall_minimo": RECALL_MIN,
    }

    log.info(
        "Métricas en Test -> F1: %.4f | Recall: %.4f | Accuracy: %.4f",
        metricas["f1"],
        metricas["recall"],
        metricas["accuracy"],
    )

    # Guardar artefactos
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    with open(METRICS_PATH, "w") as f:
        json.dump(metricas, f, indent=2)

    log.info("Artefactos guardados exitosamente en: %s", ARTIFACTS)
    return metricas


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    df = load_data()
    log.info("Dataset cargado exitosamente (%d filas)", len(df))
    metricas = train(df)
    print(json.dumps(metricas, indent=2))
