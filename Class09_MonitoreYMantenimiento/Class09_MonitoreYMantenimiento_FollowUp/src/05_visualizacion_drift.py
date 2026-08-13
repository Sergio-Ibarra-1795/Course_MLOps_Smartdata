"""
src/05_visualizacion_drift.py — Visualización comparativa de distribuciones (KS + PSI).

Genera gráficos estáticos (PNG) con Matplotlib/Seaborn para complementar los 
reportes interactivos de EvidentlyAI:
  1. 04_distribuciones_comparativas.png — Histogramas con KS-Test y PSI
  2. 05_psi_barras.png — Diagnóstico general de PSI por feature

Ejecución desde la raíz del proyecto:
    python src/05_visualizacion_drift.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo obligatorio para CI/CD / Headless Linux
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | VIZ | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Resolución dinámica de rutas
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reportes"

VARS_MONITOR = [
    "alcohol", "volatile_acidity", "residual_sugar", "total_sulfur_dioxide",
    "fixed_acidity", "density", "ph", "sulphates",
]

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol",
]

COLORS = {"ref": "#2196F3", "prod": "#DC2626"}


def calcular_psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """Calcula el Population Stability Index (PSI) entre dos distribuciones."""
    breakpoints = np.linspace(
        min(ref.min(), prod.min()),
        max(ref.max(), prod.max()),
        bins + 1,
    )
    ref_pct = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Suavizado para evitar división entre cero o logaritmo de cero
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> tuple[str, str]:
    """Clasifica el PSI según umbrales estándar de riesgo."""
    if psi < 0.10:
        return "ESTABLE", "#059669"  # Verde
    if psi < 0.25:
        return "ALERTA", "#D97706"   # Naranja
    return "CRÍTICO", "#DC2626"       # Rojo


def grafico_distribuciones(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Genera histogramas comparativos con métricas estadísticas (KS y PSI)."""
    n_vars = len(VARS_MONITOR)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    fig.suptitle(
        "Referencia vs Producción — Wine Quality Dataset\n"
        "Análisis de Cambio de Distribución (KS Test & PSI)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    axes = axes.flatten()

    for i, col in enumerate(VARS_MONITOR):
        ax = axes[i]

        # Histogramas con densidad de probabilidad normalizada
        ax.hist(df_ref[col], bins=30, alpha=0.5, label="Referencia", color=COLORS["ref"], density=True)
        ax.hist(df_prod[col], bins=30, alpha=0.5, label="Producción", color=COLORS["prod"], density=True)

        # Cálculo de métricas estadísticas
        ks_stat, p_val = ks_2samp(df_ref[col], df_prod[col])
        psi = calcular_psi(df_ref[col], df_prod[col])
        estado_ks = "DRIFT" if p_val < 0.05 else "OK"
        estado_psi, color_estado = clasificar_psi(psi)

        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_xlabel(
            f"KS={ks_stat:.3f} (p={p_val:.3f}) [{estado_ks}] | PSI={psi:.3f} [{estado_psi}]",
            fontsize=8.5, color=color_estado, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        log.info("%-25s | KS=%.3f (p=%.4f) | PSI=%.3f [%s]", col, ks_stat, p_val, psi, estado_psi)

    # Ocultar subplots no utilizados
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = REPORTS_DIR / "04_distribuciones_comparativas.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


def grafico_psi_barras(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Genera gráfico de barras consolidado de PSI para todas las variables."""
    psis = [calcular_psi(df_ref[f], df_prod[f]) for f in FEATURES]
    colores = [clasificar_psi(p)[1] for p in psis]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(FEATURES)), psis, color=colores, alpha=0.85, edgecolor="white")

    # Etiquetas numéricas sobre cada barra
    for bar, v in zip(bars, psis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels([f.replace("_", "\n") for f in FEATURES], fontsize=8.5)
    ax.axhline(y=0.10, color="#D97706", linestyle="--", alpha=0.7, label="PSI = 0.10 (Umbral Alerta)")
    ax.axhline(y=0.25, color="#DC2626", linestyle="--", alpha=0.7, label="PSI = 0.25 (Umbral Crítico)")

    ax.set_title("Population Stability Index (PSI) por Feature", fontweight="bold", fontsize=12)
    ax.set_ylabel("Valor PSI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = REPORTS_DIR / "05_psi_barras.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(out_path)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ref_file = DATA_DIR / "wine_ref_con_pred.csv"
    prod_file = DATA_DIR / "wine_prod_con_pred.csv"

    for p in [ref_file, prod_file]:
        if not p.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {p}. Ejecuta los pasos 1 y 2 primero.")

    df_ref = pd.read_csv(ref_file)
    df_prod = pd.read_csv(prod_file)
    log.info("Datos cargados correctamente — Referencia: %d | Producción: %d", len(df_ref), len(df_prod))

    print("\n[1/2] Generando gráfico de distribuciones comparativas...")
    path1 = grafico_distribuciones(df_ref, df_prod)
    print(f" -> Guardado: {path1}")

    print("\n[2/2] Generando gráfico de barras de PSI...")
    path2 = grafico_psi_barras(df_ref, df_prod)
    print(f" -> Guardado: {path2}")

    print("\n" + "=" * 55)
    print(" VISUALIZACIONES GENERADAS CON ÉXITO")
    print("=" * 55)
    print(f" -> {path1}")
    print(f" -> {path2}")
    print("\n✓ Paso 5 completado — Pipeline de monitoreo finalizado.")