"""
src/04_pipeline_monitoreo.py — Pipeline automatizado de monitoreo con alertas y Quality Gate.

Ejecuta el ciclo completo de monitoreo:
 1. Carga referencia y datos de producción (nuevo lote)
 2. Detecta data drift con EvidentlyAI
 3. Mide degradación de performance (F1 decay)
 4. Genera alertas si supera umbrales
 5. Guarda resumen JSON y HTML con timestamp
 6. Quality gate: sys.exit(1) si el estado es CRITICO

Ejecución:
 python src/04_pipeline_monitoreo.py                  # Lote por defecto
 python src/04_pipeline_monitoreo.py wine_2026_Q3     # Nombre de lote personalizado
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | MONITOR | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Definición de rutas adaptadas a WSL / Linux
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reportes"

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]

# ── Umbrales de alerta ────────────────────────────────────────────────────────
DRIFT_SHARE_UMBRAL = 0.30  # > 30% de features con drift -> ALERTA
F1_DECAY_UMBRAL = 0.10     # > 10% caída en F1 -> ALERTA
F1_DECAY_CRITICO = 0.15    # > 15% caída en F1 -> CRÍTICO (Falla Quality Gate)


def ejecutar_monitoreo(nombre_lote: str = "lote_actual") -> dict:
    """Ejecuta el pipeline completo de monitoreo y retorna el resumen."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    alertas = []
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar datos ───────────────────────────────────────────────────────
    ref_path = DATA_DIR / "wine_ref_con_pred.csv"
    prod_path = DATA_DIR / "wine_prod_con_pred.csv"

    for p in [ref_path, prod_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta pasos 1 y 2 primero.")

    df_ref = pd.read_csv(ref_path)
    df_prod = pd.read_csv(prod_path)
    log.info("Lote '%s' | Ref: %d | Prod: %d muestras", nombre_lote, len(df_ref), len(df_prod))

    # ── 2. Data Drift ─────────────────────────────────────────────────────────
    log.info("Ejecutando análisis de data drift...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=df_ref[FEATURES],
        current_data=df_prod[FEATURES],
    )

    html_path = REPORTS_DIR / f"{timestamp}_{nombre_lote}_drift.html"
    snapshot.save_html(str(html_path))

    resultado = snapshot.dict() if hasattr(snapshot, "dict") else snapshot.as_dict()
    drift_metrics = resultado["metrics"][0]["value"]
    drift_share = drift_metrics.get("share", 0.0)
    drift_n = drift_metrics.get("count", 0)
    drift_total = len(FEATURES)
    drift_detected = drift_share > 0

    if drift_share > DRIFT_SHARE_UMBRAL:
        alertas.append({
            "tipo": "DATA_DRIFT_CRITICO",
            "detalle": f"{drift_share*100:.0f}% features con drift ({drift_n}/{drift_total})",
            "accion": "REENTRENAR modelo con datos nuevos",
        })
        log.warning("ALERTA DRIFT: %.0f%% features con drift", drift_share * 100)
    else:
        log.info("Drift OK: %.0f%% features con drift", drift_share * 100)

    # ── 3. Performance Decay ──────────────────────────────────────────────────
    log.info("Calculando degradación de performance...")
    modelo_path = ARTIFACTS_DIR / "modelo_wine.pkl"

    if modelo_path.exists():
        with open(modelo_path, "rb") as f:
            modelo = pickle.load(f)

        f1_ref = f1_score(df_ref["calidad_bin"], modelo.predict(df_ref[FEATURES]), zero_division=0)
        f1_prod = f1_score(df_prod["calidad_bin"], modelo.predict(df_prod[FEATURES]), zero_division=0)
        decay = (f1_ref - f1_prod) / f1_ref if f1_ref > 0 else 0.0

        if decay > F1_DECAY_CRITICO:
            alertas.append({
                "tipo": "PERFORMANCE_CRITICA",
                "detalle": f"F1 cayó {decay*100:.1f}%: {f1_ref:.3f} → {f1_prod:.3f}",
                "accion": "REENTRENAR URGENTE — degradación supera umbral crítico",
            })
            log.error("CRÍTICO: F1 cayó %.1f%% (%.3f → %.3f)", decay * 100, f1_ref, f1_prod)
        elif decay > F1_DECAY_UMBRAL:
            alertas.append({
                "tipo": "PERFORMANCE_DEGRADADA",
                "detalle": f"F1 cayó {decay*100:.1f}%: {f1_ref:.3f} → {f1_prod:.3f}",
                "accion": "EVALUAR reentrenamiento — monitorear en próximos ciclos",
            })
            log.warning("ALERTA: F1 cayó %.1f%% (%.3f → %.3f)", decay * 100, f1_ref, f1_prod)
        else:
            log.info("Performance OK: F1 decay = %.1f%%", decay * 100)
    else:
        f1_ref = f1_prod = decay = 0.0
        log.warning("modelo_wine.pkl no encontrado — saltando análisis de performance")

    # ── 4. Determinar Estado ──────────────────────────────────────────────────
    estado = "OK"
    if any(a["tipo"].startswith("PERFORMANCE_CRITICA") or a["tipo"].startswith("DATA_DRIFT_CRITICO") for a in alertas):
        estado = "CRITICO"
    elif alertas:
        estado = "ALERTA"

    resumen = {
        "timestamp": timestamp,
        "lote": nombre_lote,
        "estado": estado,
        "drift_detectado": drift_detected,
        "drift_share": drift_share,
        "f1_ref": round(f1_ref, 4),
        "f1_prod": round(f1_prod, 4),
        "f1_decay": round(decay, 4),
        "alertas": alertas,
    }

    return resumen


if __name__ == "__main__":
    nombre_lote = sys.argv[1] if len(sys.argv) > 1 else "lote_actual"
    resumen = ejecutar_monitoreo(nombre_lote)

    # Guardar JSON de resumen del lote
    resumen_path = REPORTS_DIR / f"{resumen['timestamp']}_{nombre_lote}_resumen.json"
    with open(resumen_path, "w") as f:
        json.dump(resumen, f, indent=2)

    print("\n" + "=" * 55)
    print(f" ESTADO FINAL DE LA PIPELINE: {resumen['estado']}")
    print("=" * 55)

    if resumen["alertas"]:
        for alerta in resumen["alertas"]:
            print(f" ⚠ {alerta['tipo']}: {alerta['detalle']}")
            print(f"   → {alerta['accion']}")
    else:
        print(" ✓ Sin alertas — modelo estable en producción")

    # ── 5. Quality Gate (Control de salida de proceso) ──────────────────────
    if resumen["estado"] == "CRITICO":
        log.error("Quality gate FALLIDO — Estado CRÍTICO detectado. Abortando pipeline.")
        sys.exit(1)

    print("\n✓ Paso 4 completado con éxito.")