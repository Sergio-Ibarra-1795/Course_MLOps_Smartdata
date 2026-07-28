"""
Sección 12 — Registro del mejor modelo en MLflow Model Registry.

Flujo: Run → Registered Model → Staging → (Production)

Ejecutar: python src/06_model_registry.py
Prerrequisito: src/mejor_run.txt (generado por 05_mejor_modelo.py)
"""
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from pathlib import Path

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Configurar MLflow ─────────────────────────────────────────────────────────
MLFLOW_DIR     = Path("mlruns").resolve()
TRACKING_URI   = MLFLOW_DIR.as_uri()
MODEL_REGISTRY = "PicosIntensidad-Clasificador"

mlflow.set_tracking_uri(TRACKING_URI)

# ── Leer el mejor run ─────────────────────────────────────────────────────────
MEJOR_RUN_FILE = Path("src/mejor_run.txt")
if not MEJOR_RUN_FILE.exists():
    raise FileNotFoundError(
        "No se encontró src/mejor_run.txt. "
        "Ejecuta primero: python src/05_mejor_modelo.py"
    )

with open(MEJOR_RUN_FILE) as f:
    lines     = f.read().strip().split("\n")
    mejor_id  = lines[0]
    mejor_nom = lines[1]
    artifact  = lines[2]

print("=" * 50)
print("SECCIÓN 12 — MLflow Model Registry")
print("=" * 50)
print(f"Run a registrar : {mejor_nom}")
print(f"Run ID          : {mejor_id[:12]}...")
print(f"Artifact path   : {artifact}")
print(f"Registry name   : {MODEL_REGISTRY}")

# ── Registrar el mejor modelo ─────────────────────────────────────────────────
model_uri = f"runs:/{mejor_id}/{artifact}"
result    = mlflow.register_model(model_uri, MODEL_REGISTRY)
version   = result.version
print(f"\n✓ Modelo registrado: {MODEL_REGISTRY} v{version}")

# ── Promover a Staging ────────────────────────────────────────────────────────
client = MlflowClient(tracking_uri=TRACKING_URI)
client.transition_model_version_stage(
    name    = MODEL_REGISTRY,
    version = version,
    stage   = "Staging",
)
print(f"✓ Modelo en Staging: {MODEL_REGISTRY} v{version}")

# ── Verificar estado ──────────────────────────────────────────────────────────
model_details = client.get_model_version(MODEL_REGISTRY, version)
print(f"\nEstado del modelo registrado:")
print(f"  Nombre   : {model_details.name}")
print(f"  Versión  : {model_details.version}")
print(f"  Stage    : {model_details.current_stage}")
print(f"  Run ID   : {model_details.run_id[:12]}...")

# ── Verificar carga desde el Registry ────────────────────────────────────────
print(f"\nVerificando carga desde Staging...")
modelo_staging = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY}/Staging")
print(f"✓ Modelo cargado desde Staging: {type(modelo_staging).__name__}")

# ── Opcional: promover a Production ──────────────────────────────────────────
# Descomentar cuando el modelo haya pasado validación en Staging:
#
# client.transition_model_version_stage(
#     name=MODEL_REGISTRY, version=version, stage="Production"
# )
# print(f"✓ Modelo en Production: {MODEL_REGISTRY} v{version}")

print("\n" + "=" * 50)
print("TEMA 6 COMPLETADO")
print("=" * 50)
print(f"  Modelo registrado : {MODEL_REGISTRY} v{version} (Staging)")
print(f"  MLflow UI         : mlflow ui --host 0.0.0.0 --port 5000")
print(f"  Gráfico           : src/comparacion_runs.png")
