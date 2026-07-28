"""
PASO 8 — Orquestador de Pipeline (run_pipeline.py)

Ejecuta las 6 secciones del pipeline MLOps en secuencia estricta.
Se detiene inmediatamente si cualquiera de los scripts devuelve un código de error.

Ejecución desde la raíz del proyecto:
    python pipelines/run_pipeline.py
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# ── 0. Configuración de Entorno y Rutas ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

# Configurar directorio de logs
LOGS_DIR = config.ARTIFACTS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "pipeline_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | MLOPS | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("mlops_pipeline")

# Lista ordenada de scripts a ejecutar
SCRIPTS = [
    ("Paso 1 : Base Pipeline (Data Prep & Baseline)", "pipelines/01_base_pipeline.py"),
    ("Paso 2 : GridSearchCV + MLflow (Run 1)",        "pipelines/02_gridsearch_mlflow.py"),
    ("Paso 3 : RandomizedSearchCV + MLflow (Run 2)",  "pipelines/03_randomsearch_mlflow.py"),
    ("Paso 4 : Comparar Runs de MLflow",              "pipelines/04_comparar_runs.py"),
    ("Paso 5 : Selección del Mejor Modelo",          "pipelines/05_mejor_modelo.py"),
    ("Paso 6 : Promoción a Model Registry (Staging)", "pipelines/06_model_registry.py"),
]


def ejecutar_script(nombre: str, script_relativo: str) -> bool:
    """Ejecuta un script con subprocess desde la raíz y retorna True si finalizó sin errores."""
    script_path = config.BASE_DIR / script_relativo
    
    if not script_path.exists():
        log.error("XXX FICHERO NO ENCONTRADO: %s", script_path)
        return False

    inicio = time.time()
    log.info(">>> INICIANDO: %s", nombre)
    
    resultado = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,  # Imprime la salida en la consola en tiempo real
        cwd=str(config.BASE_DIR),
    )
    
    duracion = round(time.time() - inicio, 2)
    
    if resultado.returncode == 0:
        log.info("<<< COMPLETADO OK: %s (Tiempo: %.2f s)\n", nombre, duracion)
        return True
    else:
        log.error("XXX FALLÓ LA EJECUCIÓN: %s (Tiempo: %.2f s)\n", nombre, duracion)
        return False


# ── Ejecución Secuencial ─────────────────────────────────────────────────────
log.info("=" * 60)
log.info("🚀 ORQUESTADOR MLOPS — PIPELINE COMPLETO DE ENTRENAMIENTO")
log.info("=" * 60)

resumen = []
exito_total = True

for nombre, script in SCRIPTS:
    ok = ejecutar_script(nombre, script)
    resumen.append((nombre, ok))
    if not ok:
        exito_total = False
        log.error("⛔ Pipeline interrumpido debido a un error en: %s", script)
        log.error("Puedes depurar ejecutando manualmente: python %s", script)
        sys.exit(1)

log.info("=" * 60)
log.info("✅ PIPELINE COMPLETO EJECUTADO CON ÉXITO")
log.info("=" * 60)

for nombre, ok in resumen:
    estado = "OK" if ok else "ERROR"
    log.info("  [%s] %s", estado, nombre)

log.info("\n📊 Artefactos finales generados:")
log.info("  • Gráfico comparativo : %s", config.ARTIFACTS_DIR / "reports" / "figures" / "comparacion_runs.png")
log.info("  • Metadatos mejor run : %s", config.RUN_IDS_TXT)
log.info("  • Log de orquestación : %s", LOG_FILE)
log.info("  • Registro oficial    : MLflow Model Registry ('Modelo-Picos-Intensidad' en Staging)")