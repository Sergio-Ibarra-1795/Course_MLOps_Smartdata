"""
Sección 9 — RandomizedSearchCV + MLflow (Run 2).

Muestrea 30 combinaciones aleatorias de un espacio continuo.
Más eficiente que GridSearch para espacios grandes.

Ejecutar: python src/03_randomsearch_mlflow.py
Prerrequisito: src/base_data.pkl (generado por 01_base_pipeline.py)
"""
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, classification_report,
)

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Cargar datos ──────────────────────────────────────────────────────────────
BASE_PKL = Path("src/base_data.pkl")
if not BASE_PKL.exists():
    raise FileNotFoundError(
        "No se encontró src/base_data.pkl. "
        "Ejecuta primero: python src/01_base_pipeline.py"
    )

with open(BASE_PKL, "rb") as f:
    X_train_smote, y_train_smote, X_test, y_test = pickle.load(f)

print(f"Datos cargados — Train: {X_train_smote.shape} | Test: {X_test.shape}")

# ── Configurar MLflow ─────────────────────────────────────────────────────────
MLFLOW_DIR   = Path("mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT   = "picos-intensidad-tuning"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
print(f"MLflow tracking URI : {TRACKING_URI}")
print(f"MLflow experiment   : {EXPERIMENT}")

# ── Sección 9: RandomizedSearchCV ────────────────────────────────────────────
param_dist = {
    "max_depth":         randint(2, 25),
    "min_samples_split": randint(2, 50),
    "min_samples_leaf":  randint(1, 20),
    "max_features":      ["sqrt", "log2", None],
    "criterion":         ["gini", "entropy"],
}

rs = RandomizedSearchCV(
    estimator           = DecisionTreeClassifier(random_state=123),
    param_distributions = param_dist,
    n_iter              = 30,
    cv                  = StratifiedKFold(5, shuffle=True, random_state=123),
    scoring             = "f1",
    random_state        = 123,
    n_jobs              = -1,
    verbose             = 1,
)

print("\nEjecutando RandomizedSearchCV (30 combinaciones x 5 folds)...")
rs.fit(X_train_smote, y_train_smote)

print(f"\nMejores parámetros : {rs.best_params_}")
print(f"Mejor F1 CV        : {rs.best_score_:.4f}")

# ── Registrar en MLflow ───────────────────────────────────────────────────────
with mlflow.start_run(run_name="RandomSearch-DecisionTree") as run:
    mlflow.log_params(rs.best_params_)
    mlflow.log_param("metodo_busqueda",  "RandomizedSearchCV")
    mlflow.log_param("n_iter",           30)
    mlflow.log_param("tecnica_balanceo", "SMOTE")
    mlflow.log_param("n_folds",          5)
    mlflow.log_param("random_state",     123)

    y_pred_rs = rs.best_estimator_.predict(X_test)
    mlflow.log_metrics({
        "accuracy":   round(accuracy_score(y_test,                    y_pred_rs), 4),
        "f1":         round(f1_score(y_test,                          y_pred_rs), 4),
        "recall":     round(recall_score(y_test,                      y_pred_rs), 4),
        "precision":  round(precision_score(y_test, y_pred_rs, zero_division=0), 4),
        "cv_f1_mean": round(rs.best_score_, 4),
    })

    mlflow.sklearn.log_model(
        sk_model      = rs.best_estimator_,
        artifact_path = "modelo_randomsearch",
    )
    run_id_rs = run.info.run_id
    print(f"\nRun ID RandomSearch: {run_id_rs}")

print("\n=== RESULTADO RANDOMSEARCH ===")
print(classification_report(y_test, y_pred_rs))

with open("src/run_ids.txt", "a") as f:
    f.write(f"randomsearch:{run_id_rs}\n")

print("✓ Run ID guardado en src/run_ids.txt")
print("  Siguiente paso: python src/04_comparar_runs.py")
