"""
Sección 12 — MLflow Model Registry (Staging -> Production)

Lee los metadatos del mejor modelo desde artifacts/metadata/mejor_run.txt,
lo inscribe en el Model Registry centralizado como 'Modelo-Picos-Intensidad' y
promueve la versión a la etapa 'Staging'. La transición a 'Production' queda
comentada hasta completar la validación técnica en Staging.

Ejecución desde la raíz del proyecto:
    python pipelines/06_model_registry.py
"""

import os
import sys
from pathlib import Path

# ── 0. Configuración de Entorno y Rutas ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import mlflow
from mlflow.tracking import MlflowClient

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── 1. Configurar MLflow Tracking y Client ──────────────────────────────────
MLFLOW_DIR = (config.BASE_DIR / "mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
MODEL_NAME = "Modelo-Picos-Intensidad"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print(f"📍 MLflow tracking URI : {TRACKING_URI}")
print(f"📦 Nombre del modelo    : {MODEL_NAME}")

# ── 2. Leer metadatos del mejor run ──────────────────────────────────────────
MEJOR_RUN_TXT = config.METADATA_DIR / "mejor_run.txt"

if not MEJOR_RUN_TXT.exists():
    raise FileNotFoundError(
        f"❌ No se encontró '{MEJOR_RUN_TXT}'. "
        "Ejecuta primero: python pipelines/05_mejor_modelo.py"
    )

with open(MEJOR_RUN_TXT, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

if len(lines) < 3:
    raise ValueError(
        f"❌ El archivo '{MEJOR_RUN_TXT}' debe contener al menos 3 líneas "
        "(run_id, run_name, artifact_path)."
    )

run_id, run_name, artifact_path = lines[0], lines[1], lines[2]

print("\n" + "=" * 55)
print("SECCIÓN 12 — Registro y Promoción en Model Registry")
print("=" * 55)
print(f"🔑 Run ID a registrar : {run_id}")
print(f"🏷️ Run Name          : {run_name}")
print(f"📂 Artifact Path      : {artifact_path}")

# ── 3. Registrar el Modelo en el Model Registry ─────────────────────────────
model_uri = f"runs:/{run_id}/{artifact_path}"
print(f"\n📥 Registrando modelo desde URI: {model_uri}...")

# Inscribir modelo y asignar número de versión
mv = mlflow.register_model(model_uri, MODEL_NAME)

print(f"✅ Modelo inscrito en el Registro con éxito:")
print(f"  • Nombre registrado : {mv.name}")
print(f"  • Versión asignada  : {mv.version}")
print(f"  • Etapa inicial     : {mv.current_stage}")

# ── 4. Promover el modelo a STAGING ─────────────────────────────────────────
print(f"\n🚀 Promoviendo la Versión {mv.version} a la etapa 'Staging'...")

client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="Staging",
    archive_existing_versions=True,  # Archiva versiones anteriores en Staging
)

# Documentar la versión registrada con metadata del experimento
client.update_model_version(
    name=MODEL_NAME,
    version=mv.version,
    description=(
        f"Modelo campeón seleccionado automáticamente ({run_name}). "
        f"Promovido a Staging para pruebas de integración/A-B Testing."
    ),
)

# Confirmación final de la transición
updated_mv = client.get_model_version(MODEL_NAME, mv.version)
print(f"✅ Transición completada con éxito:")
print(f"  • Nueva Etapa (Stage) : {updated_mv.current_stage}")
print(f"  • Descripción         : {updated_mv.description}")

# ── 5. Transición a PRODUCTION (Comentada hasta validación) ─────────────────
print("\n" + "-" * 55)
print("📌 FLUJO A PRODUCCIÓN (PENDIENTE DE VALIDACIÓN EN STAGING)")
print("-" * 55)
print("Una vez validadas las pruebas en Staging, descomenta el bloque siguiente:\n")

"""
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=updated_mv.version,
    stage="Production",
    archive_existing_versions=True  # Mueve la versión de producción actual a Archived
)
print(f"🚀 Versión {updated_mv.version} promovida exitosamente a PRODUCTION.")
"""

print(f"\n✓ Proceso completado. El modelo '{MODEL_NAME}' v{mv.version} está listo en Staging.")