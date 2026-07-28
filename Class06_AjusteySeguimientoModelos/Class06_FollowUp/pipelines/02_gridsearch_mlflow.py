"""
Sección 8 — GridSearchCV + MLflow (Run 1)

Búsqueda exhaustiva sobre el espacio de hiperparámetros definido.
120 combinaciones x 5 folds = 600 entrenamientos.
Registra el mejor estimador en MLflow como Run 1.

Ejecución desde la raíz del proyecto:
    python pipelines/02_gridsearch_mlflow.py

Prerrequisito: data/processed/base_data.pkl (generado por 01_base_pipeline.py)
"""

import os
import sys
import pickle
from pathlib import Path

# ── 0. Configuración de Entorno y Rutas ──────────────────────────────────────
# Agregamos la raíz del proyecto al sys.path para importar config.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, classification_report,
)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── 1. Cargar datos congelados de la línea base ──────────────────────────────
BASE_PKL = config.BASE_DATA_PKL

if not BASE_PKL.exists():
    raise FileNotFoundError(
        f"❌ No se encontró '{BASE_PKL}'. "
        "Ejecuta primero: python pipelines/01_base_pipeline.py"
    )

with open(BASE_PKL, "rb") as f:
    X_train_smote, y_train_smote, X_test, y_test = pickle.load(f)

print(f"✅ Datos cargados correctamente — Train: {X_train_smote.shape} | Test: {X_test.shape}")

# ── 2. Configurar MLflow Tracking ─────────────────────────────────────────────
MLFLOW_DIR = (config.BASE_DIR / "mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT = "picos-intensidad-tuning"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

print(f"📍 MLflow tracking URI : {TRACKING_URI}")
print(f"🧪 MLflow experiment   : {EXPERIMENT}")

# ── 3. Definir espacio de búsqueda y ejecutar GridSearchCV ──────────────────
param_grid = {
    "max_depth":         [3, 5, 8, 10, None],   # 5 opciones
    "min_samples_split": [2, 5, 10, 20],         # 4 opciones
    "min_samples_leaf":  [1, 2, 4],              # 3 opciones
    "criterion":         ["gini", "entropy"],    # 2 opciones
}
# 120 combinaciones x 5 folds = 600 entrenamientos

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

gs = GridSearchCV(
    estimator          = DecisionTreeClassifier(random_state=123),
    param_grid         = param_grid,
    cv                 = skf,
    scoring            = "f1",
    n_jobs             = -1,
    verbose            = 1,
    return_train_score = True,
)

print("\n🔍 Ejecutando GridSearchCV (120 combinaciones x 5 folds)...")
gs.fit(X_train_smote, y_train_smote)

print(f"\n✅ Mejores parámetros : {gs.best_params_}")
print(f"🏆 Mejor F1 CV        : {gs.best_score_:.4f}")

# ── 4. Registrar en MLflow ───────────────────────────────────────────────────
n_combinaciones = len(gs.cv_results_["params"])

with mlflow.start_run(run_name="GridSearch-DecisionTree") as run:
    # Parámetros del mejor estimador
    mlflow.log_params(gs.best_params_)
    mlflow.log_param("metodo_busqueda",  "GridSearchCV")
    mlflow.log_param("n_combinaciones",  n_combinaciones)
    mlflow.log_param("tecnica_balanceo", "SMOTE")
    mlflow.log_param("n_folds",          5)
    mlflow.log_param("random_state",     123)

    # Evaluar en test set
    y_pred_gs = gs.best_estimator_.predict(X_test)
    mlflow.log_metrics({
        "accuracy":   round(accuracy_score(y_test, y_pred_gs), 4),
        "f1":         round(f1_score(y_test, y_pred_gs), 4),
        "recall":     round(recall_score(y_test, y_pred_gs), 4),
        "precision":  round(precision_score(y_test, y_pred_gs, zero_division=0), 4),
        "cv_f1_mean": round(gs.best_score_, 4),
    })

    # Registrar modelo
    mlflow.sklearn.log_model(
        sk_model      = gs.best_estimator_,
        artifact_path = "modelo_gridsearch",
    )
    run_id_gs = run.info.run_id
    print(f"\n🚀 Run ID GridSearch registrado: {run_id_gs}")

print("\n=== RESULTADO GRIDSEARCH EN TEST SET ===")
print(classification_report(y_test, y_pred_gs))

# ── 5. Guardar run_id en artifacts/metadata/run_ids.txt ───────────────────────
config.METADATA_DIR.mkdir(parents=True, exist_ok=True)

with open(config.RUN_IDS_TXT, "w") as f:
    f.write(f"gridsearch:{run_id_gs}\n")

print(f"\n✓ Run ID guardado exitosamente en: {config.RUN_IDS_TXT}")
print("  Siguiente paso: python pipelines/03_randomsearch_mlflow.py")