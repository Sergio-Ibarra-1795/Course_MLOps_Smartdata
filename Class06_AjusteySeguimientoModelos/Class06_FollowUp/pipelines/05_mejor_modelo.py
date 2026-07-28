"""
Sección 11 — Selección programática del mejor modelo

Consulta la API de MLflow para identificar la ejecución con mejor F1-Score en el Test Set,
carga dinámicamente su artefacto de modelo deserializado, realiza una verificación final 
y exporta los metadatos a artifacts/metadata/mejor_run.txt.

Ejecución desde la raíz del proyecto:
    python pipelines/05_mejor_modelo.py
"""

import os
import sys
import pickle
from pathlib import Path

# ── 0. Configuración de Entorno y Rutas ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── 1. Cargar datos de test congelados ───────────────────────────────────────
BASE_PKL = config.BASE_DATA_PKL

if not BASE_PKL.exists():
    raise FileNotFoundError(
        f"❌ No se encontró '{BASE_PKL}'. "
        "Ejecuta primero: python pipelines/01_base_pipeline.py"
    )

with open(BASE_PKL, "rb") as f:
    _, _, X_test, y_test = pickle.load(f)

print(f"✅ Datos de test cargados correctamente — Shape: {X_test.shape}")

# ── 2. Configurar MLflow Tracking ─────────────────────────────────────────────
MLFLOW_DIR = (config.BASE_DIR / "mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT = "picos-intensidad-tuning"

mlflow.set_tracking_uri(TRACKING_URI)

# ── 3. Buscar y ordenar runs por F1 en MLflow ───────────────────────────────
exp = mlflow.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise RuntimeError(
        f"❌ Experimento '{EXPERIMENT}' no encontrado en {TRACKING_URI}. "
        "Ejecuta primero los scripts 02 y 03."
    )

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.f1 DESC"],
)

if runs.empty:
    raise RuntimeError("No se encontraron ejecuciones registradas en MLflow.")

# ── 4. Identificar el mejor run ──────────────────────────────────────────────
mejor_run = runs.iloc[0]
mejor_id = mejor_run["run_id"]
mejor_nombre = mejor_run.get("tags.mlflow.runName", "desconocido")

print("\n" + "=" * 55)
print("SECCIÓN 11 — Selección Programática del Mejor Modelo")
print("=" * 55)
print(f"🏆 Mejor Run Identificado : {mejor_nombre}")
print(f"🔑 Run ID                : {mejor_id}")
print(f"🎯 F1-Score (Test)       : {mejor_run['metrics.f1']:.4f}")
print(f"📡 Recall (Test)         : {mejor_run['metrics.recall']:.4f}")
print(f"🎯 Precision (Test)      : {mejor_run.get('metrics.precision', 0):.4f}")
print(f"📊 Accuracy (Test)       : {mejor_run['metrics.accuracy']:.4f}")
print(f"📈 CV F1 Mean            : {mejor_run.get('metrics.cv_f1_mean', 'N/A')}")

# ── 5. Cargar el modelo directamente desde MLflow URI ───────────────────────
if "Grid" in mejor_nombre:
    artifact = "modelo_gridsearch"
elif "Random" in mejor_nombre:
    artifact = "modelo_randomsearch"
else:
    artifact = "modelo_optuna"

model_uri = f"runs:/{mejor_id}/{artifact}"

print(f"\n📦 Cargando modelo directamente desde MLflow URI: {model_uri}")
modelo_produccion = mlflow.sklearn.load_model(model_uri)

print(f"  • Clase del estimador : {type(modelo_produccion).__name__}")
print(f"  • Parámetros óptimos  : {modelo_produccion.get_params()}")

# ── 6. Evaluación final de confirmación ──────────────────────────────────────
y_pred_final = modelo_produccion.predict(X_test)
print("\n=== EVALUACIÓN FINAL DEL MEJOR MODELO EN TEST SET ===")
print(classification_report(y_test, y_pred_final))

# ── 7. Guardar metadatos para el Model Registry ──────────────────────────────
config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
MEJOR_RUN_TXT = config.METADATA_DIR / "mejor_run.txt"

with open(MEJOR_RUN_TXT, "w") as f:
    f.write(f"{mejor_id}\n{mejor_nombre}\n{artifact}\n")

print(f"✓ Metadatos del mejor run guardados exitosamente en: {MEJOR_RUN_TXT}")
print("  Siguiente paso: python pipelines/06_model_registry.py")