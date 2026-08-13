"""
src/run_monitoreo.py — Orquestador general del pipeline de monitoreo de ML.

Ejecuta en secuencia los 5 pasos del pipeline:
 1. Carga / generación del dataset Wine Quality
 2. Entrenamiento del modelo base y registro de métricas
 3. Generación de reportes interactivos de Evidently AI
 4. Pipeline de monitoreo con alertas y Quality Gate (sys.exit)
 5. Visualizaciones de distribuciones estáticas (KS Test + PSI)

Uso desde terminal / WSL:
 python src/run_monitoreo.py
 python src/run_monitoreo.py wine_2026_Q3
"""

import logging
from pathlib import Path
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ORQUESTADOR | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Resolución dinámica de la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reportes"

# Definición ordenada de la secuencia de ejecución
SCRIPTS = [
    ("Paso 1: Cargar / generar dataset Wine Quality", SRC_DIR / "01_cargar_datos.py"),
    ("Paso 2: Entrenar modelo y medir baseline", SRC_DIR / "02_entrenar_modelo.py"),
    ("Paso 3: Generar reportes HTML EvidentlyAI", SRC_DIR / "03_reporte_drift.py"),
    ("Paso 4: Pipeline de monitoreo con alertas", SRC_DIR / "04_pipeline_monitoreo.py"),
    ("Paso 5: Visualizaciones de distribuciones", SRC_DIR / "05_visualizacion_drift.py"),
]


def ejecutar(nombre: str, script_path: Path, extra_args: list = None) -> bool:
    """Ejecuta un script individual mediante subprocess y evalúa su returncode."""
    inicio = time.time()
    log.info(">>> Iniciando %s", nombre)

    if not script_path.exists():
        log.error("XXX Script no encontrado en la ruta: %s", script_path)
        return False

    # Construcción del comando CLI utilizando el mismo intérprete de Python activo
    cmd = [sys.executable, str(script_path)] + (extra_args or [])

    # Ejecución sincrónica en el proceso hijo
    resultado = subprocess.run(cmd, capture_output=False)
    duracion = round(time.time() - inicio, 2)

    if resultado.returncode == 0:
        log.info("<<< OK: %s completado en %.2f s", nombre, duracion)
        return True

    log.error("XXX FALLO: %s detuvo la ejecución (código de salida: %d)", nombre, resultado.returncode)
    return False


def main():
    nombre_lote = sys.argv[1] if len(sys.argv) > 1 else "lote_actual"

    log.info("=" * 60)
    log.info(" INICIANDO PIPELINE DE MONITOREO DE DRIFT | LOTE: %s", nombre_lote)
    log.info("=" * 60)

    # Garantizar la existencia de directorios de trabajo
    for folder in [DATA_DIR, ARTIFACTS_DIR, REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    resumen = []

    for nombre, script_path in SCRIPTS:
        # Pasa el parámetro nombre_lote exclusivamente al script 04
        extra = [nombre_lote] if "04_pipeline_monitoreo" in script_path.name else []
        
        exito = ejecutar(nombre, script_path, extra)
        resumen.append((nombre, exito))

        # Patrón Fail-Fast: Aborta el pipeline si cualquier paso falla
        if not exito:
            log.error("Pipeline interrumpido en: %s. Revisa los logs superiores.", nombre)
            sys.exit(1)

    # Resumen final de ejecución
    log.info("=" * 60)
    log.info(" RESUMEN DE EJECUCIÓN DEL PIPELINE")
    log.info("=" * 60)
    for nombre, exito in resumen:
        estado_str = "OK" if exito else "FAIL"
        log.info(" [%s] %s", estado_str, nombre)

    log.info("-" * 60)
    log.info(" Archivos generados en carpeta /reportes:")
    log.info("  ├── Reportes HTML : Evidently Drift/Quality/Performance")
    log.info("  ├── Visualizaciones: 04_distribuciones_comparativas.png")
    log.info("  ├── Visualizaciones: 05_psi_barras.png")
    log.info("  └── Métricas JSON  : <timestamp>_%s_resumen.json", nombre_lote)
    log.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Ejecución cancelada por el usuario.")
        sys.exit(130)