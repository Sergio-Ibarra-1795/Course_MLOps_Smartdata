"""
src/01_cargar_datos.py — Descarga y prepara el dataset Wine Quality (UCI) con simulación de Drift.
Adaptado para ejecución en entorno local Linux / WSL2.

Ejecución desde la raíz del proyecto:
    python src/01_cargar_datos.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Definición de rutas dinámicas referenciadas a la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REF_PATH = DATA_DIR / "wine_referencia.csv"
PROD_PATH = DATA_DIR / "wine_produccion.csv"

FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]
TARGET = "quality"


def descargar_dataset() -> pd.DataFrame:
    """Descarga Wine Quality desde UCI ML Repository o conmuta a generación sintética de respaldo."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("[INFO] Descargando Wine Quality desde UCI ML Repository...")
        wine = fetch_ucirepo(id=186)
        X = wine.data.features
        y = wine.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = df.columns.str.replace(" ", "_").str.lower()
        print(f"[OK] Descarga exitosa: {df.shape[0]} filas x {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"[WARNING] No se pudo descargar desde UCI Repository ({e}).")
        print("[INFO] Generando dataset sintético de respaldo...")
        return _generar_sintetico()


def _generar_sintetico(n: int = 6497, seed: int = 42) -> pd.DataFrame:
    """Dataset sintético con las mismas propiedades estadísticas que Wine Quality UCI."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "fixed_acidity": rng.normal(7.2, 1.8, n).clip(4, 16),
        "volatile_acidity": rng.normal(0.34, 0.16, n).clip(0.08, 1.6),
        "citric_acid": rng.normal(0.32, 0.15, n).clip(0, 1),
        "residual_sugar": rng.exponential(4.5, n).clip(0.9, 65),
        "chlorides": rng.normal(0.055, 0.03, n).clip(0.01, 0.6),
        "free_sulfur_dioxide": rng.normal(30, 17, n).clip(1, 289),
        "total_sulfur_dioxide": rng.normal(115, 56, n).clip(6, 440),
        "density": rng.normal(0.994, 0.003, n).clip(0.987, 1.04),
        "ph": rng.normal(3.22, 0.16, n).clip(2.7, 4.01),
        "sulphates": rng.normal(0.53, 0.15, n).clip(0.22, 2.0),
        "alcohol": rng.normal(10.5, 1.2, n).clip(8, 15),
    })

    score = (
        (df["alcohol"] - 8) / 7 * 3
        - (df["volatile_acidity"] - 0.08) / 1.52 * 2
        + rng.normal(0, 0.5, n)
    )
    df["quality"] = (score * 2 + 4).clip(3, 9).round().astype(int)
    print(f"[OK] Sintético generado: {df.shape}")
    return df


def simular_drift(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide en REFERENCIA (temporada fría) y PRODUCCIÓN (temporada caliente).
    Simula Data Drift en volatile_acidity, alcohol, residual_sugar y total_sulfur_dioxide.
    """
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(42)
    mediana_alc = df["alcohol"].median()

    # REFERENCIA: Vinos con alcohol <= mediana
    df_ref = df[df["alcohol"] <= mediana_alc].copy()
    df_ref, _ = train_test_split(df_ref, test_size=0.3, random_state=42)

    # PRODUCCIÓN: Vinos con alcohol > mediana + desplazamiento artificial (Drift)
    df_prod = df[df["alcohol"] > mediana_alc].copy()
    n = len(df_prod)

    df_prod["volatile_acidity"] += rng.normal(0.08, 0.02, n)
    df_prod["alcohol"] += rng.normal(0.50, 0.15, n)
    df_prod["residual_sugar"] += rng.normal(1.20, 0.40, n)
    df_prod["total_sulfur_dioxide"] += rng.normal(-8.0, 3.00, n)

    # Mantener dentro de rangos físicos reales
    df_prod["volatile_acidity"] = df_prod["volatile_acidity"].clip(0, 2)
    df_prod["alcohol"] = df_prod["alcohol"].clip(8, 15)
    df_prod["residual_sugar"] = df_prod["residual_sugar"].clip(0.9, 65)
    df_prod["total_sulfur_dioxide"] = df_prod["total_sulfur_dioxide"].clip(6, 440)

    return df_ref.reset_index(drop=True), df_prod.reset_index(drop=True)


if __name__ == "__main__":
    # Garantiza la existencia del directorio data/ en la raíz del proyecto
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Obtener dataset
    df = descargar_dataset()

    # 2. Generar ventanas de tiempo con Drift
    df_ref, df_prod = simular_drift(df)

    # 3. Exportar archivos CSV
    df_ref.to_csv(REF_PATH, index=False)
    df_prod.to_csv(PROD_PATH, index=False)

    # 4. Salida informativa
    print(f"\n[DATASETS GUARDADOS]")
    print(f" -> Referencia : {REF_PATH.relative_to(BASE_DIR)} ({len(df_ref)} muestras)")
    print(f" -> Producción : {PROD_PATH.relative_to(BASE_DIR)} ({len(df_prod)} muestras)")

    print(f"\n[VERIFICACIÓN DE DRIFT SIMULADO (Alcohol)]")
    print(f" -> Promedio Referencia : {df_ref['alcohol'].mean():.2f} %vol")
    print(f" -> Promedio Producción : {df_prod['alcohol'].mean():.2f} %vol")
    print(f" -> Delta               : +{df_prod['alcohol'].mean() - df_ref['alcohol'].mean():.2f} %vol")

    print("\n✓ Paso 1 completado con éxito. Puedes continuar con el siguiente script.")