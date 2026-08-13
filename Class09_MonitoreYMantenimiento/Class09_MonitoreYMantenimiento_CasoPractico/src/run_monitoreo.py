"""
src/run_monitoreo.py — Orquestador: ejecuta el pipeline completo de monitoreo.

Uso desde la terminal del Codespace:
    python src/run_monitoreo.py              # lote con nombre por defecto
    python src/run_monitoreo.py wine_2024_Q2 # con nombre de lote específico

Equivale a ejecutar en secuencia:
    python src/01_cargar_datos.py
    python src/02_entrenar_modelo.py
    python src/03_reporte_drift.py
    python src/04_pipeline_monitoreo.py
    python src/05_visualizacion_drift.py
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ORQUESTADOR | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPTS = [
    ("Paso 1: Cargar / generar dataset Wine Quality", "src/01_cargar_datos.py"),
    ("Paso 2: Entrenar modelo y medir baseline",       "src/02_entrenar_modelo.py"),
    ("Paso 3: Generar reportes HTML EvidentlyAI",      "src/03_reporte_drift.py"),
    ("Paso 4: Pipeline de monitoreo con alertas",      "src/04_pipeline_monitoreo.py"),
    ("Paso 5: Visualizaciones de distribuciones",      "src/05_visualizacion_drift.py"),
]


def ejecutar(nombre: str, script: str, extra_args: list = None) -> bool:
    inicio = time.time()
    log.info(">>> %s", nombre)
    cmd = [sys.executable, script] + (extra_args or [])
    r   = subprocess.run(cmd, capture_output=False)
    dur = round(time.time() - inicio, 2)
    if r.returncode == 0:
        log.info("<<< OK: %s (%.2f s)", nombre, dur)
        return True
    log.error("XXX FALLO: %s (código: %d)", nombre, r.returncode)
    return False


if __name__ == "__main__":
    nombre_lote = sys.argv[1] if len(sys.argv) > 1 else "lote_actual"

    log.info("=" * 55)
    log.info(" MONITOREO DE DRIFT — Wine Quality | %s", nombre_lote)
    log.info("=" * 55)

    Path("reportes").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    resumen = []
    for nombre, script in SCRIPTS:
        # El paso 4 recibe el nombre del lote como argumento
        extra = [nombre_lote] if "pipeline_monitoreo" in script else []
        ok    = ejecutar(nombre, script, extra)
        resumen.append((nombre, ok))
        if not ok:
            log.error("Pipeline detenido. Corrige el error y reintenta.")
            sys.exit(1)

    log.info("=" * 55)
    log.info(" MONITOREO COMPLETADO")
    log.info("=" * 55)
    for nombre, ok in resumen:
        log.info("  [%s] %s", "OK" if ok else "XX", nombre)

    log.info("")
    log.info("  Reportes HTML  : reportes/01_data_drift.html")
    log.info("                   reportes/02_data_quality.html")
    log.info("                   reportes/03_model_performance.html")
    log.info("  Visualizaciones: reportes/04_distribuciones_comparativas.png")
    log.info("                   reportes/05_psi_barras.png")
    log.info("  Resumen JSON   : reportes/<timestamp>_<lote>_resumen.json")
