"""
Sección 9 — RandomizedSearchCV + MLflow (Run 2)

Muestrea 30 combinaciones aleatorias de un espacio continuo de hiperparámetros.
Más eficiente que GridSearch para espacios grandes y ayuda a explorar restricciones
de regularización más amplias.

Ejecución desde la raíz del proyecto:
    python pipelines/03_randomsearch_mlflow.py

Prerrequisito: data/processed/base_data.pkl (generado por 01_base_pipeline.py)
"""

import os
import sys
import pickle
from pathlib import Path
from scipy.stats import randint

# ── 0. Configuración de Entorno y Rutas ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import mlflow
import mlflow.sklearn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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

# ── 3. Definir espacio probabilístico y ejecutar RandomizedSearchCV ─────────
param_dist = {
    "max_depth":         randint(2, 25),          # Muestreo continuo entre 2 y 24
    "min_samples_split": randint(2, 50),          # Muestreo continuo entre 2 y 49
    "min_samples_leaf":  randint(1, 20),          # Muestreo continuo entre 1 y 19
    "max_features":      ["sqrt", "log2", None],  # Control de subespacio de variables
    "criterion":         ["gini", "entropy"],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

rs = RandomizedSearchCV(
    estimator           = DecisionTreeClassifier(random_state=123),
    param_distributions = param_dist,
    n_iter              = 30,                     # 30 combinaciones aleatorias
    cv                  = skf,                    # 5 folds
    scoring             = "f1",
    random_state        = 123,
    n_jobs              = -1,
    verbose             = 1,
)

print("\n🎲 Ejecutando RandomizedSearchCV (30 combinaciones x 5 folds = 150 entrenamientos)...")
rs.fit(X_train_smote, y_train_smote)

print(f"\n✅ Mejores parámetros : {rs.best_params_}")
print(f"🏆 Mejor F1 CV        : {rs.best_score_:.4f}")

# ── 4. Registrar en MLflow ───────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomSearch-DecisionTree") as run:
    mlflow.log_params(rs.best_params_)
    mlflow.log_param("metodo_busqueda",  "RandomizedSearchCV")
    mlflow.log_param("n_iter",           30)
    mlflow.log_param("tecnica_balanceo", "SMOTE")
    mlflow.log_param("n_folds",          5)
    mlflow.log_param("random_state",     123)

    # Evaluar en test set
    y_pred_rs = rs.best_estimator_.predict(X_test)
    mlflow.log_metrics({
        "accuracy":   round(accuracy_score(y_test, y_pred_rs), 4),
        "f1":         round(f1_score(y_test, y_pred_rs), 4),
        "recall":     round(recall_score(y_test, y_pred_rs), 4),
        "precision":  round(precision_score(y_test, y_pred_rs, zero_division=0), 4),
        "cv_f1_mean": round(rs.best_score_, 4),
    })

    # Registrar modelo en MLflow
    mlflow.sklearn.log_model(
        sk_model      = rs.best_estimator_,
        artifact_path = "modelo_randomsearch",
    )
    run_id_rs = run.info.run_id
    print(f"\n🚀 Run ID RandomSearch registrado: {run_id_rs}")

print("\n=== RESULTADO RANDOMSEARCH EN TEST SET ===")
print(classification_report(y_test, y_pred_rs))

# ── 5. Agregar run_id a artifacts/metadata/run_ids.txt ────────────────────────
config.METADATA_DIR.mkdir(parents=True, exist_ok=True)

with open(config.RUN_IDS_TXT, "a") as f:
    f.write(f"randomsearch:{run_id_rs}\n")

print(f"\n✓ Run ID adjuntado exitosamente en: {config.RUN_IDS_TXT}")
print("  Siguiente paso: python pipelines/04_comparar_runs.py (o siguiente pipeline)")