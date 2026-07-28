"""
Sección 10 — Comparación de runs con mlflow.search_runs().

Genera tabla comparativa ordenada por F1 y gráfico de barras.

Ejecutar: python src/04_comparar_runs.py
"""
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# ── Configurar MLflow ─────────────────────────────────────────────────────────
MLFLOW_DIR   = Path("mlruns").resolve()
TRACKING_URI = MLFLOW_DIR.as_uri()
EXPERIMENT   = "picos-intensidad-tuning"

mlflow.set_tracking_uri(TRACKING_URI)

# ── Buscar todos los runs del experimento ─────────────────────────────────────
exp = mlflow.get_experiment_by_name(EXPERIMENT)
if exp is None:
    raise RuntimeError(
        f"Experiment '{EXPERIMENT}' no encontrado. "
        "Ejecuta primero los scripts 02 y 03."
    )

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.f1 DESC"],
)

print(f"Total de runs encontrados: {len(runs)}")

# ── Tabla comparativa ─────────────────────────────────────────────────────────
cols = [
    "tags.mlflow.runName",
    "params.metodo_busqueda",
    "metrics.f1",
    "metrics.recall",
    "metrics.accuracy",
    "metrics.cv_f1_mean",
    "params.n_combinaciones",
]
cols_disp = [c for c in cols if c in runs.columns]

print("\n=== COMPARACIÓN DE RUNS (ordenado por F1 DESC) ===")
print(runs[cols_disp].to_string(index=False))

mejor = runs.iloc[0]
print(f"\n★ MEJOR RUN: {mejor.get('tags.mlflow.runName', mejor['run_id'][:8])}")
print(f"  F1       : {mejor['metrics.f1']:.4f}")
print(f"  Recall   : {mejor['metrics.recall']:.4f}")
print(f"  Accuracy : {mejor['metrics.accuracy']:.4f}")

# ── Gráfico comparativo ───────────────────────────────────────────────────────
metricas_plot = ["metrics.f1", "metrics.recall", "metrics.accuracy"]
titulos_plot  = ["F1-Score", "Recall", "Accuracy"]
labels = [
    str(r.get("tags.mlflow.runName", r["run_id"][:8]))
    for _, r in runs.iterrows()
]
colors = ["#2E75B6", "#C55A11", "#1E6E38", "#7030A0"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Comparación de Runs — Picos Intensidad Tuning",
    fontsize=13, fontweight="bold",
)

for ax, met, titulo, col in zip(axes, metricas_plot, titulos_plot, colors):
    vals = runs[met].tolist()
    bars = ax.bar(labels, vals, color=col, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.4f}", ha="center", fontsize=9, fontweight="bold",
        )
    ax.set_title(titulo, fontweight="bold")
    ax.set_ylim([max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.08)])
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
output_png = "src/comparacion_runs.png"
plt.savefig(output_png, dpi=120, bbox_inches="tight")
plt.close()

print(f"\n✓ Gráfico guardado: {output_png}")
print("  Siguiente paso: python src/05_mejor_modelo.py")
