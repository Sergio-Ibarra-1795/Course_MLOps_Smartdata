"""
Sección 10 — Comparación de runs con mlflow.search_runs()

Consulta la API de MLflow para extraer las ejecuciones registradas (GridSearch, RandomSearch),
genera una tabla comparativa en consola y exporta un gráfico de barras comparativo.

Ejecución desde la raíz del proyecto:
    python pipelines/04_comparar_runs.py
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
import pandas as pd
import matplotlib.pyplot as plt

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── 1. Configurar MLflow Tracking ─────────────────────────────────────────────
MLFLOW_DIR = (config.BASE_DIR / "mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT = "picos-intensidad-tuning"

mlflow.set_tracking_uri(TRACKING_URI)

# ── 2. Buscar todos los runs del experimento ─────────────────────────────────────
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

print(f"📊 Total de runs encontrados en MLflow: {len(runs)}")

if runs.empty:
    raise RuntimeError("No se encontraron ejecuciones (runs) registradas dentro del experimento.")

# ── 3. Tabla comparativa ─────────────────────────────────────────────────────────
cols = [
    "tags.mlflow.runName",
    "params.metodo_busqueda",
    "metrics.f1",
    "metrics.recall",
    "metrics.precision",
    "metrics.accuracy",
    "metrics.cv_f1_mean",
]
cols_disp = [c for c in cols if c in runs.columns]

print("\n=== COMPARACIÓN DE RUNS (ordenado por F1 DESC) ===")
print(runs[cols_disp].to_string(index=False))

mejor = runs.iloc[0]
nombre_mejor = mejor.get('tags.mlflow.runName', mejor['run_id'][:8])
print(f"\n★ MEJOR RUN SELECCIONADO: {nombre_mejor}")
print(f"  F1-Score  : {mejor['metrics.f1']:.4f}")
print(f"  Recall    : {mejor['metrics.recall']:.4f}")
print(f"  Precision : {mejor.get('metrics.precision', 0):.4f}")
print(f"  Accuracy  : {mejor['metrics.accuracy']:.4f}")

# ── 4. Generar gráfico comparativo ───────────────────────────────────────────────
metricas_plot = ["metrics.f1", "metrics.recall", "metrics.accuracy"]
titulos_plot  = ["F1-Score", "Recall", "Accuracy"]
labels = [
    str(r.get("tags.mlflow.runName", r["run_id"][:8]))
    for _, r in runs.iterrows()
]
colors = ["#2E75B6", "#C55A11", "#1E6E38", "#7030A0"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Comparación de Experimentos en MLflow — Picos de Intensidad",
    fontsize=13, fontweight="bold",
)

for ax, met, titulo, col in zip(axes, metricas_plot, titulos_plot, colors):
    if met in runs.columns:
        vals = runs[met].tolist()
        bars = ax.bar(labels, vals, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold",
            )
        ax.set_title(titulo, fontweight="bold")
        y_min = max(0, min(vals) - 0.05) if vals else 0
        y_max = min(1.0, max(vals) + 0.08) if vals else 1.0
        ax.set_ylim([y_min, y_max])
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()

# Guardar figura en la estructura de artefactos
OUTPUT_DIR = config.ARTIFACTS_DIR / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_png = OUTPUT_DIR / "comparacion_runs.png"

plt.savefig(output_png, dpi=120, bbox_inches="tight")
plt.close()

print(f"\n✓ Gráfico guardado exitosamente en: {output_png}")
print("  Siguiente paso: python pipelines/05_optuna_mlflow.py (o siguiente pipeline)")