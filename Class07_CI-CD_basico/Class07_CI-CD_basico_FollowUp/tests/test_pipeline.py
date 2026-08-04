"""tests/test_pipeline.py — Tests del pipeline de entrenamiento."""

import json
from pathlib import Path
import sys

# ==============================================================================
# RESOLUCIÓN DINÁMICA DE RUTAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_data import generate  # noqa: E402
from train_pipeline import train  # noqa: E402


# ==============================================================================
# CASOS DE PRUEBA (TEST SUITE)
# ==============================================================================
def test_train_returns_metrics(tmp_path, monkeypatch):
    """train() debe retornar un diccionario con las métricas esperadas."""
    monkeypatch.chdir(tmp_path)
    df = generate(n=300)
    metricas = train(df)
    assert "f1" in metricas
    assert "recall" in metricas
    assert "accuracy" in metricas


def test_train_f1_positive(tmp_path, monkeypatch):
    """El F1-score del modelo entrenado debe ser estrictamente mayor que 0."""
    monkeypatch.chdir(tmp_path)
    metricas = train(generate(n=300))
    assert metricas["f1"] > 0.0


def test_train_metrics_in_range(tmp_path, monkeypatch):
    """Todas las métricas generadas deben estar acotadas entre 0.0 y 1.0."""
    monkeypatch.chdir(tmp_path)
    metricas = train(generate(n=300))
    for key in ("f1", "recall", "accuracy"):
        assert 0.0 <= metricas[key] <= 1.0, f"{key} fuera de rango: {metricas[key]}"


def test_train_saves_model(tmp_path, monkeypatch):
    """train() debe serializar y guardar el modelo en artifacts/modelo.pkl."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    assert (tmp_path / "artifacts" / "modelo.pkl").exists()


def test_train_saves_metrics(tmp_path, monkeypatch):
    """train() debe guardar artifacts/metrics.json con la estructura adecuada."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    metrics_file = tmp_path / "artifacts" / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r", encoding="utf-8") as f:
        m = json.load(f)
    assert "f1" in m and "recall" in m and "accuracy" in m


def test_train_saves_best_params(tmp_path, monkeypatch):
    """metrics.json debe registrar los hiperparámetros óptimos de GridSearchCV."""
    monkeypatch.chdir(tmp_path)
    train(generate(n=300))
    with open(tmp_path / "artifacts" / "metrics.json", "r", encoding="utf-8") as f:
        m = json.load(f)
    assert "params" in m
    assert isinstance(m["params"], dict)
