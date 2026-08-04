"""tests/test_data.py — Tests del generador de datos."""

from pathlib import Path
import sys
import pandas as pd

# ==============================================================================
# RESOLUCIÓN DINÁMICA DE RUTAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_data import generate  # noqa: E402

FEATURES = [
    "Presion",
    "Tonelaje",
    "Velocidad",
    "%Solidos",
    "Potencia",
    "F80",
    "Brazo",
]


# ==============================================================================
# CASOS DE PRUEBA (TEST SUITE)
# ==============================================================================
def test_generate_returns_dataframe():
    """generate() debe retornar un DataFrame de Pandas."""
    df = generate(n=200)
    assert isinstance(df, pd.DataFrame)


def test_generate_has_correct_columns():
    """El dataset debe incluir todas las features y la variable objetivo."""
    df = generate(n=200)
    expected_columns = FEATURES + ["picos_intens"]
    for col in expected_columns:
        assert col in df.columns, f"Columna faltante en el dataset: {col}"


def test_generate_target_is_binary():
    """El target 'picos_intens' debe contener exclusivamente valores 0 o 1."""
    df = generate(n=300)
    unique_values = set(df["picos_intens"].unique())
    assert unique_values.issubset({0, 1}), f"Valores inesperados: {unique_values}"


def test_generate_class_imbalance():
    """La proporción de la clase positiva (1) debe estar entre 5% y 20%."""
    df = generate(n=500)
    rate = df["picos_intens"].mean()
    assert 0.05 <= rate <= 0.20, f"Tasa de clase 1 fuera de rango esperado: {rate:.2%}"


def test_generate_no_nulls():
    """El dataset generado no debe contener ningún valor nulo (NaN)."""
    df = generate(n=300)
    assert df.isnull().sum().sum() == 0, "Se encontraron valores nulos en el dataset."


def test_generate_n_rows():
    """El dataset debe generar exactamente el número 'n' de filas solicitado."""
    df = generate(n=400)
    assert len(df) == 400, f"Se esperaban 400 filas pero se obtuvieron {len(df)}."


def test_generate_reproducible():
    """Invocaciones con el mismo random_state deben producir datasets idénticos."""
    df1 = generate(n=200, random_state=99)
    df2 = generate(n=200, random_state=99)
    assert df1.reset_index(drop=True).equals(df2.reset_index(drop=True))
