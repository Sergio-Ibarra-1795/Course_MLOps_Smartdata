"""
run_pipeline.py — Orquestador del pipeline completo Banco Wiesse.

Uso:
    python run_pipeline.py

Ejecuta secuencialmente:
    Etapa 1: Ingestion    →  artifacts/data_clean.csv
    Etapa 2: Features     →  artifacts/features.csv
    Etapa 3: Train        →  artifacts/modelo.pkl + mlruns/
    Etapa 4: Evaluate     →  artifacts/metrics.json + reporte_evaluacion.png

Si alguna etapa falla, el pipeline se detiene con sys.exit(1)
(compatible con CI/CD: GitHub Actions, GitLab CI, etc.)
"""
import sys, logging, time, json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pipeline.ingest   as etapa1
import pipeline.features as etapa2
import pipeline.train    as etapa3
import pipeline.evaluate as etapa4
import config as C

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | PIPELINE | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline_run.log'),
    ],
)
log = logging.getLogger('orquestador')


def ejecutar_etapa(nombre: str, funcion) -> dict:
    """Ejecuta una etapa y captura tiempo y estado."""
    inicio = time.time()
    log.info('>>> Iniciando : %s', nombre)
    try:
        funcion()
        dur = round(time.time() - inicio, 2)
        log.info('<<< Completada: %s  (%.2f s)', nombre, dur)
        return {'etapa': nombre, 'estado': 'OK', 'duracion_s': dur}
    except Exception as e:
        dur = round(time.time() - inicio, 2)
        log.error('XXX FALLO    : %s', nombre)
        log.error('    Motivo   : %s', str(e))
        return {'etapa': nombre, 'estado': 'FALLO', 'duracion_s': dur, 'error': str(e)}


def main():
    inicio_total = time.time()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.info('=' * 58)
    log.info('  PIPELINE BANCO WIESSE — %s', ts)
    log.info('  Versión: %s', C.MLFLOW_RUN_NAME)
    log.info('=' * 58)

    ETAPAS = [
        ('Etapa 1: Ingestion',     etapa1.run),
        ('Etapa 2: Features',      etapa2.run),
        ('Etapa 3: Entrenamiento', etapa3.run),
        ('Etapa 4: Evaluación',    etapa4.run),
    ]

    resumen = []
    for nombre, funcion in ETAPAS:
        resultado = ejecutar_etapa(nombre, funcion)
        resumen.append(resultado)
        if resultado['estado'] == 'FALLO':
            log.error('Pipeline detenido. Etapa fallida: %s', nombre)
            sys.exit(1)

    dur_total = round(time.time() - inicio_total, 2)
    log.info('=' * 58)
    log.info('  PIPELINE COMPLETADO EN %.2f segundos', dur_total)
    log.info('=' * 58)
    for r in resumen:
        icono = 'OK' if r['estado'] == 'OK' else 'XX'
        log.info('  [%s] %-30s (%.2f s)', icono, r['etapa'], r['duracion_s'])

    if C.METRICS_PATH.exists():
        with open(C.METRICS_PATH) as f:
            m = json.load(f)
        log.info('')
        log.info('  MÉTRICAS FINALES:')
        log.info('    Modelo  : %s', m.get('modelo', '?'))
        log.info('    Recall  : %.4f  (umbral: %.2f)', m['recall'], C.RECALL_MINIMO)
        log.info('    F1      : %.4f', m['f1'])
        log.info('    FN      : %d defaults no detectados', m['FN'])
        log.info('    Costo   : $%s', f"{m['costo_estimado']:,}")
        log.info('    Reporte : %s', C.REPORT_PATH)


if __name__ == '__main__':
    main()
