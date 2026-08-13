"""
src/05_visualizacion_drift.py — Visualización comparativa de distribuciones.

Genera gráficos propios (matplotlib) para identificar las variables
con mayor drift. Complementa los reportes interactivos de EvidentlyAI.

Ejecutar: python src/05_visualizacion_drift.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | VIZ | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reportes")

VARS_MONITOR = [
    "alcohol", "volatile_acidity", "residual_sugar", "total_sulfur_dioxide",
    "fixed_acidity", "density", "ph", "sulphates",
]
COLORS = {"ref": "#2196F3", "prod": "#DC2626"}


def calcular_psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (PSI)."""
    breakpoints = np.linspace(
        min(ref.min(), prod.min()),
        max(ref.max(), prod.max()),
        bins + 1,
    )
    ref_pct  = np.histogram(ref,  bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Evitar log(0)
    ref_pct  = np.where(ref_pct  == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> tuple[str, str]:
    """Clasifica PSI según umbrales estándar."""
    if psi < 0.10:
        return "ESTABLE", "#059669"
    if psi < 0.25:
        return "ALERTA",  "#D97706"
    return "CRÍTICO", "#DC2626"


def grafico_distribuciones(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Genera gráfico de distribuciones comparativas (2x4 subplots)."""
    n_vars = len(VARS_MONITOR)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    fig.suptitle(
        "Referencia vs Producción — Wine Quality Dataset\n"
        "Distribución de variables con drift simulado",
        fontsize=13, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for i, col in enumerate(VARS_MONITOR):
        ax = axes[i]

        # Histogramas
        ax.hist(df_ref[col],  bins=30, alpha=0.6, label="Referencia", color=COLORS["ref"],  density=True)
        ax.hist(df_prod[col], bins=30, alpha=0.6, label="Producción", color=COLORS["prod"], density=True)

        # Métricas de drift
        ks_stat, p_val = ks_2samp(df_ref[col], df_prod[col])
        psi            = calcular_psi(df_ref[col], df_prod[col])
        estado_ks      = "DRIFT" if p_val < 0.05 else "OK"
        estado_psi, c  = clasificar_psi(psi)

        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_xlabel(
            f"KS={ks_stat:.3f} p={p_val:.3f} [{estado_ks}]  |  PSI={psi:.3f} [{estado_psi}]",
            fontsize=8.5, color=c,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        log.info("%-30s KS=%.3f p=%.4f PSI=%.3f [%s]", col, ks_stat, p_val, psi, estado_psi)

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = str(REPORTS_DIR / "04_distribuciones_comparativas.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def grafico_psi_barras(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Gráfico de barras PSI para todas las features."""
    features = [
        "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
        "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
        "density", "ph", "sulphates", "alcohol",
    ]

    psis   = [calcular_psi(df_ref[f], df_prod[f]) for f in features]
    colores = [clasificar_psi(p)[1] for p in psis]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(features)), psis, color=colores, alpha=0.85, edgecolor="white")

    for bar, v in zip(bars, psis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    ax.axhline(y=0.10, color="#D97706", linestyle="--", alpha=0.7, label="PSI=0.10 (alerta)")
    ax.axhline(y=0.25, color="#DC2626", linestyle="--", alpha=0.7, label="PSI=0.25 (crítico)")
    ax.set_title("Population Stability Index (PSI) — Wine Quality Features",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("PSI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = str(REPORTS_DIR / "05_psi_barras.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)

    for p in [DATA_DIR / "wine_ref_con_pred.csv", DATA_DIR / "wine_prod_con_pred.csv"]:
        if not p.exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos 1 y 2 primero.")

    df_ref  = pd.read_csv(DATA_DIR / "wine_ref_con_pred.csv")
    df_prod = pd.read_csv(DATA_DIR / "wine_prod_con_pred.csv")
    log.info("Datos cargados — Ref: %d | Prod: %d", len(df_ref), len(df_prod))

    print("\n[1/2] Generando gráfico de distribuciones comparativas...")
    path1 = grafico_distribuciones(df_ref, df_prod)
    print(f"  Guardado: {path1}")

    print("\n[2/2] Generando gráfico PSI por feature...")
    path2 = grafico_psi_barras(df_ref, df_prod)
    print(f"  Guardado: {path2}")

    print("\n" + "="*55)
    print(" VISUALIZACIONES GENERADAS")
    print("="*55)
    print(f" {path1}")
    print(f" {path2}")
    print("\n✓ Paso 5 completado — Pipeline de monitoreo finalizado")
