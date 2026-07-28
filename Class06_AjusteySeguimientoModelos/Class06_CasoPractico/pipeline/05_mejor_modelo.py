"""
Sección 11 — Selección programática del mejor modelo.

Carga el mejor run por F1 directamente desde MLflow.

Ejecutar: python src/05_mejor_modelo.py
"""
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import classification_report

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Cargar datos de test ──────────────────────────────────────────────────────
BASE_PKL = Path("src/base_data.pkl")
if not BASE_PKL.exists():
    raise FileNotFoundError(
        "No se encontró src/base_data.pkl. "
        "Ejecuta primero: python src/01_base_pipeline.py"
    )

with open(BASE_PKL, "rb") as f:
    _, _, X_test, y_test = pickle.load(f)

# ── Configurar MLflow ─────────────────────────────────────────────────────────
MLFLOW_DIR   = Path("mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT   = "picos-intensidad-tuning"

mlflow.set_tracking_uri(TRACKING_URI)

# ── Buscar y ordenar runs por F1 ──────────────────────────────────────────────
exp = mlflow.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise RuntimeError(
        f"Experiment '{EXPERIMENT}' no encontrado. "
        "Ejecuta primero los scripts 02 y 03."
    )

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.f1 DESC"],
)

# ── Identificar el mejor run ──────────────────────────────────────────────────
mejor_run    = runs.iloc[0]
mejor_id     = mejor_run["run_id"]
mejor_nombre = mejor_run.get("tags.mlflow.runName", "desconocido")

print("=" * 50)
print("SECCIÓN 11 — Selección del mejor modelo")
print("=" * 50)
print(f"Mejor run  : {mejor_nombre}")
print(f"Run ID     : {mejor_id}")
print(f"F1         : {mejor_run['metrics.f1']:.4f}")
print(f"Recall     : {mejor_run['metrics.recall']:.4f}")
print(f"Accuracy   : {mejor_run['metrics.accuracy']:.4f}")
print(f"CV F1 mean : {mejor_run.get('metrics.cv_f1_mean', 'N/A')}")

# ── Cargar el modelo directamente desde MLflow ────────────────────────────────
artifact  = "modelo_gridsearch" if "Grid" in mejor_nombre else "modelo_randomsearch"
model_uri = f"runs:/{mejor_id}/{artifact}"

print(f"\nCargando modelo desde: {model_uri}")
modelo_produccion = mlflow.sklearn.load_model(model_uri)

print(f"Tipo de modelo: {type(modelo_produccion).__name__}")
print(f"Parámetros    : {modelo_produccion.get_params()}")

# ── Evaluación final completa ─────────────────────────────────────────────────
y_pred_final = modelo_produccion.predict(X_test)
print("\n=== EVALUACIÓN FINAL DEL MEJOR MODELO ===")
print(classification_report(y_test, y_pred_final))

# ── Guardar info para el Model Registry ──────────────────────────────────────
with open("src/mejor_run.txt", "w") as f:
    f.write(f"{mejor_id}\n{mejor_nombre}\n{artifact}")

print("✓ Info guardada en src/mejor_run.txt")
print("  Siguiente paso: python src/06_model_registry.py")
