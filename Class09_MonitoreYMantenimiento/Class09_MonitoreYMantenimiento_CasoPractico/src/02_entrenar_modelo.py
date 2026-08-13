"""
src/02_entrenar_modelo.py — Entrena clasificador de calidad de vino.

Binariza la calidad: bueno (>= 6) vs malo (< 6).
Entrena en datos de REFERENCIA y evalúa en ambos conjuntos
para medir la degradación de performance (concept drift).

Ejecutar: python src/02_entrenar_modelo.py
Prerrequisito: data/wine_referencia.csv y data/wine_produccion.csv
"""
import json
import pickle
import logging
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score, classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR      = Path("data")
ARTIFACTS_DIR = Path("artifacts")
REF_PATH      = DATA_DIR / "wine_referencia.csv"
PROD_PATH     = DATA_DIR / "wine_produccion.csv"
MODEL_PATH    = ARTIFACTS_DIR / "modelo_wine.pkl"
BASELINE_PATH = ARTIFACTS_DIR / "baseline_metrics.json"

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]
TARGET = "quality"


def binarizar(df: pd.DataFrame) -> pd.DataFrame:
    """Crea columna calidad_bin: 1 = bueno (>= 6), 0 = malo (< 6)."""
    df = df.copy()
    df["calidad_bin"] = (df[TARGET] >= 6).astype(int)
    return df


def calcular_metricas(y_true, y_pred, nombre: str) -> dict:
    """Calcula y retorna métricas de clasificación."""
    m = {
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }
    log.info("[%s] F1=%.4f | Recall=%.4f | Acc=%.4f", nombre, m["f1"], m["recall"], m["accuracy"])
    return m


if __name__ == "__main__":
    # Verificar datasets
    for p in [REF_PATH, PROD_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} no encontrado. Ejecuta: python src/01_cargar_datos.py"
            )

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # Cargar y binarizar
    df_ref  = binarizar(pd.read_csv(REF_PATH))
    df_prod = binarizar(pd.read_csv(PROD_PATH))
    log.info("Referencia: %d muestras | Producción: %d muestras", len(df_ref), len(df_prod))

    # Entrenar en referencia
    modelo = GradientBoostingClassifier(n_estimators=100, random_state=42)
    modelo.fit(df_ref[FEATURES], df_ref["calidad_bin"])
    log.info("Modelo entrenado: %s", type(modelo).__name__)

    # Predicciones
    df_ref["prediction"]       = modelo.predict(df_ref[FEATURES])
    df_prod["prediction"]      = modelo.predict(df_prod[FEATURES])
    df_ref["prediction_proba"] = modelo.predict_proba(df_ref[FEATURES])[:, 1]
    df_prod["prediction_proba"] = modelo.predict_proba(df_prod[FEATURES])[:, 1]

    # Métricas
    m_ref  = calcular_metricas(df_ref["calidad_bin"],  df_ref["prediction"],  "REFERENCIA")
    m_prod = calcular_metricas(df_prod["calidad_bin"], df_prod["prediction"], "PRODUCCIÓN")
    decay  = (m_ref["f1"] - m_prod["f1"]) / m_ref["f1"] * 100

    print("\n" + "="*50)
    print(" DEGRADACIÓN DE PERFORMANCE")
    print("="*50)
    print(f" F1 Referencia : {m_ref['f1']:.4f}")
    print(f" F1 Producción : {m_prod['f1']:.4f}")
    print(f" Degradación   : {decay:.1f}%")
    estado = "⚠ ALERTA" if decay > 10 else "✓ OK"
    print(f" Estado        : {estado}")

    # Guardar modelo
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(modelo, f)
    log.info("Modelo guardado: %s", MODEL_PATH)

    # Guardar métricas baseline
    baseline = {
        "referencia":  m_ref,
        "produccion":  m_prod,
        "decay_pct":   round(decay, 1),
        "estado":      "ALERTA" if decay > 10 else "OK",
        "f1_umbral_alerta":   0.10,
        "f1_umbral_reentrenar": 0.15,
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    log.info("Baseline guardado: %s", BASELINE_PATH)

    # Guardar datasets con predicciones para EvidentlyAI
    df_ref.to_csv(DATA_DIR / "wine_ref_con_pred.csv", index=False)
    df_prod.to_csv(DATA_DIR / "wine_prod_con_pred.csv", index=False)

    print("\n✓ Paso 2 completado. Siguiente: python src/03_reporte_drift.py")
