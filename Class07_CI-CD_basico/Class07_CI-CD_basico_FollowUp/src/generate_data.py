"""generate_data.py — Genera dataset sintético de picos de intensidad."""

from pathlib import Path
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ==============================================================================
# Path(__file__) es este archivo (src/generate_data.py)
# .resolve().parents[1] sube 2 niveles para encontrar la raíz del repositorio
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Clasificacion_picos_intensidad.csv"

N = 1500
RANDOM_STATE = 42


# ==============================================================================
# FUNCIÓN GENERADORA
# ==============================================================================
def generate(n: int = N, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera el dataset sintético con la misma estructura que el original."""
    rng = np.random.default_rng(random_state)
    n0, n1 = int(n * 0.92), int(n * 0.08)

    # Clase 0: Operación Normal (92% de los datos)
    df0 = pd.DataFrame(
        {
            "Presion": rng.normal(180, 20, n0),
            "Tonelaje": rng.normal(350, 40, n0),
            "Velocidad": rng.normal(1800, 150, n0),
            "%Solidos": rng.normal(72, 5, n0),
            "Potencia": rng.normal(800, 80, n0),
            "F80": rng.normal(120, 15, n0),
            "Brazo": rng.normal(55, 8, n0),
            "picos_intens": 0,
        }
    )

    # Clase 1: Picos de Intensidad / Anomalia (8% de los datos)
    df1 = pd.DataFrame(
        {
            "Presion": rng.normal(220, 25, n1),
            "Tonelaje": rng.normal(420, 50, n1),
            "Velocidad": rng.normal(2200, 200, n1),
            "%Solidos": rng.normal(78, 6, n1),
            "Potencia": rng.normal(950, 100, n1),
            "F80": rng.normal(145, 18, n1),
            "Brazo": rng.normal(62, 10, n1),
            "picos_intens": 1,
        }
    )

    # Concatenamos, mezclamos y reindexamos
    df = pd.concat([df0, df1], ignore_index=True)
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    # parents=True garantiza crear cualquier carpeta previa que no exista
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(DATA_PATH, index=False)

    rate = df["picos_intens"].mean()
    print(
        f"Dataset generado exitosamente en: {DATA_PATH}\n"
        f"Dimensiones: {df.shape} | Tasa clase 1 (Picos): {rate:.2%}"
    )
