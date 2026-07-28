"""
run_tema6.py — Orquestador: ejecuta las 6 secciones del Tema 6 en secuencia.

Uso desde la terminal del Codespace:
    python tema6/run_tema6.py

Equivale a ejecutar uno a uno:
    python tema6/01_base_pipeline.py
    python tema6/02_gridsearch_mlflow.py
    python tema6/03_randomsearch_mlflow.py
    python tema6/04_comparar_runs.py
    python tema6/05_mejor_modelo.py
    python tema6/06_model_registry.py
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TEMA6 | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tema6_run.log"),
    ],
)
log = logging.getLogger("tema6")

SCRIPTS = [
    ("Secciones 1-7 : Base pipeline",             "pipeline/01_base_pipeline.py"),
    ("Sección  8    : GridSearchCV + MLflow",      "pipeline/02_gridsearch_mlflow.py"),
    ("Sección  9    : RandomizedSearchCV + MLflow","pipeline/03_randomsearch_mlflow.py"),
    ("Sección  10   : Comparar runs",              "pipeline/04_comparar_runs.py"),
    ("Sección  11   : Mejor modelo",               "pipeline/05_mejor_modelo.py"),
    ("Sección  12   : Model Registry",             "pipeline/06_model_registry.py"),
]


def ejecutar(nombre: str, script: str) -> bool:
    """Ejecuta un script y retorna True si tuvo éxito."""
    inicio    = time.time()
    log.info(">>> %s", nombre)
    resultado = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # muestra output en tiempo real
    )
    duracion = round(time.time() - inicio, 2)
    if resultado.returncode == 0:
        log.info("<<< OK: %s (%.2f s)", nombre, duracion)
        return True
    else:
        log.error("XXX FALLO: %s (%.2f s)", nombre, duracion)
        return False


log.info("=" * 55)
log.info(" TEMA 6 — Ajuste y Seguimiento de Modelos")
log.info("=" * 55)

resumen = []
for nombre, script in SCRIPTS:
    ok = ejecutar(nombre, script)
    resumen.append((nombre, ok))
    if not ok:
        log.error("Pipeline detenido. Corrige el error y reintenta.")
        log.error("Puedes ejecutar el script individualmente para depurar:")
        log.error("  python %s", script)
        sys.exit(1)

log.info("=" * 55)
log.info(" TEMA 6 COMPLETADO")
log.info("=" * 55)
for nombre, ok in resumen:
    estado = "OK" if ok else "XX"
    log.info("  [%s] %s", estado, nombre)

log.info("")
log.info("  Resultados:")
log.info("    Gráfico comparativo : tema6/comparacion_runs.png")
log.info("    Modelo registrado   : MLflow Model Registry (Staging)")
log.info("    Log de ejecución    : tema6_run.log")
log.info("")
log.info("  Para ver MLflow UI en Codespace:")
log.info("    mlflow ui --host 0.0.0.0 --port 5000")
log.info("    Luego: pestaña PORTS -> clic en globo del puerto 5000")
# MLFLOW_ALLOW_FILE_STORE=true mlflow ui --host 0.0.0.0 --port 5000