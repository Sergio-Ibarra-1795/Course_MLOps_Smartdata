"""tests/test_model.py — Tests del modelo serializado."""

import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# RESOLUCIÓN DINÁMICA DE RUTAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_data import generate  # noqa: E402
from train_pipeline import FEATURES, train  # noqa: E402


# ==============================================================================
# FIXTURE A NIVEL DE MÓDULO
# ==============================================================================
@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    """Fixture: entrena el modelo UNA SOLA VEZ para todos los tests del módulo."""
    tmp = tmp_path_factory.mktemp("workdir")
    os.chdir(tmp)
    df = generate(n=400)
    train(df)
    with open(tmp / "artifacts" / "modelo.pkl", "rb") as f:
        return pickle.load(f)


# ==============================================================================
# CASOS DE PRUEBA (TEST SUITE)
# ==============================================================================
def test_model_has_predict(modelo_entrenado):
    """El modelo serializado debe implementar el método predict."""
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    """El modelo serializado debe implementar el método predict_proba."""
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado):
    """Las predicciones del modelo deben ser estrictamente binarias (0 o 1)."""
    # Pasamos el DataFrame directamente sin .values para conservar los nombres de columnas
    X = generate(n=50)[FEATURES]
    y_pred = modelo_entrenado.predict(X)
    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado):
    """predict_proba debe retornar una matriz de forma (len(X), 2) con valores en [0, 1]."""
    X = generate(n=50)[FEATURES]
    proba = modelo_entrenado.predict_proba(X)
    # Comprobación dinámica basada en las filas reales de X
    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado):
    """Las probabilidades predichas por fila deben sumar exactamente 1.0."""
    X = generate(n=50)[FEATURES]
    proba = modelo_entrenado.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_model_is_decision_tree(modelo_entrenado):
    """El objeto cargado debe ser una instancia de DecisionTreeClassifier."""
    assert isinstance(modelo_entrenado, DecisionTreeClassifier)
