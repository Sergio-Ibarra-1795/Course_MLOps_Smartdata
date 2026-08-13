"""tests/test_monitoreo.py — Tests unitarios del pipeline de monitoreo."""
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC.parent))


def _load(filename, alias):
    """Carga un módulo cuyo nombre empieza con dígito usando importlib."""
    spec = importlib.util.spec_from_file_location(alias, SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Módulos cargados una sola vez al inicio de la sesión de tests
_vis  = _load("05_visualizacion_drift.py", "visualizacion_drift")
calcular_psi  = _vis.calcular_psi
clasificar_psi = _vis.clasificar_psi


# ── Tests del generador de datos ─────────────────────────────────────────────

def test_cargar_datos_genera_referencia_produccion(tmp_path, monkeypatch):
    """simular_drift debe generar dos DataFrames con tamaños razonables."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()

    _m = _load("01_cargar_datos.py", "cargar_datos")
    df = _m._generar_sintetico(n=500)
    df_ref, df_prod = _m.simular_drift(df)

    assert len(df_ref) > 0 and len(df_prod) > 0
    assert "quality" in df_ref.columns


def test_dataset_sin_nulos(tmp_path):
    """El dataset sintético no debe tener nulos."""
    _m = _load("01_cargar_datos.py", "cargar_datos")
    df = _m._generar_sintetico(n=200)
    assert df.isnull().sum().sum() == 0


def test_drift_aumenta_alcohol():
    """El drift simulado debe aumentar el alcohol medio en producción."""
    _m = _load("01_cargar_datos.py", "cargar_datos")
    df = _m._generar_sintetico(n=1000)
    df_ref, df_prod = _m.simular_drift(df)
    assert df_prod["alcohol"].mean() > df_ref["alcohol"].mean()


# ── Tests de métricas PSI ─────────────────────────────────────────────────────

def test_psi_misma_distribucion():
    """PSI de dos distribuciones idénticas debe ser ~0."""
    serie = pd.Series(np.random.normal(10, 1, 500))
    psi   = calcular_psi(serie, serie)
    assert psi < 0.05, f"PSI esperado ~0, obtenido {psi}"


def test_psi_distribuciones_distintas():
    """PSI de distribuciones muy distintas debe ser > 0.25 (crítico)."""
    ref  = pd.Series(np.random.normal(10, 1, 500))
    prod = pd.Series(np.random.normal(14, 1, 500))
    psi  = calcular_psi(ref, prod)
    assert psi > 0.25, f"PSI esperado > 0.25, obtenido {psi}"


def test_psi_clasificacion():
    """clasificar_psi debe retornar el estado correcto."""
    assert clasificar_psi(0.05)[0] == "ESTABLE"
    assert clasificar_psi(0.15)[0] == "ALERTA"
    assert clasificar_psi(0.30)[0] == "CRÍTICO"


# ── Tests del pipeline de monitoreo ──────────────────────────────────────────

def _preparar_datos(tmp_path, n=500):
    """Helper: genera CSVs de referencia y producción con predicciones."""
    _carga  = _load("01_cargar_datos.py",    "cargar_datos")
    _modelo = _load("02_entrenar_modelo.py", "entrenar_modelo")

    df = _carga._generar_sintetico(n=n)
    df_ref, df_prod = _carga.simular_drift(df)
    df_ref  = _modelo.binarizar(df_ref)
    df_prod = _modelo.binarizar(df_prod)
    return df_ref, df_prod


def test_pipeline_monitoreo_genera_json(tmp_path, monkeypatch):
    """ejecutar_monitoreo debe generar un JSON con las claves esperadas."""
    monkeypatch.chdir(tmp_path)
    for d in ("data", "reportes", "artifacts"):
        (tmp_path / d).mkdir()

    df_ref, df_prod = _preparar_datos(tmp_path, n=500)
    df_ref["prediction"]        = 1
    df_prod["prediction"]       = 1
    df_ref["prediction_proba"]  = 0.8
    df_prod["prediction_proba"] = 0.7

    df_ref.to_csv(tmp_path / "data" / "wine_ref_con_pred.csv",  index=False)
    df_prod.to_csv(tmp_path / "data" / "wine_prod_con_pred.csv", index=False)

    _pipeline = _load("04_pipeline_monitoreo.py", "pipeline_monitoreo")
    resumen   = _pipeline.ejecutar_monitoreo("test_lote")

    assert "estado"          in resumen
    assert "drift_detectado" in resumen
    assert "alertas"         in resumen
    assert "timestamp"       in resumen
    assert resumen["estado"] in ("OK", "ALERTA", "CRITICO")


def test_resumen_tiene_metricas_f1(tmp_path, monkeypatch):
    """El resumen debe incluir f1_ref y f1_prod."""
    monkeypatch.chdir(tmp_path)
    for d in ("data", "reportes", "artifacts"):
        (tmp_path / d).mkdir()

    df_ref, df_prod = _preparar_datos(tmp_path, n=400)
    df_ref["prediction"]        = df_ref["calidad_bin"]
    df_prod["prediction"]       = df_prod["calidad_bin"]
    df_ref["prediction_proba"]  = 0.9
    df_prod["prediction_proba"] = 0.8

    df_ref.to_csv(tmp_path / "data" / "wine_ref_con_pred.csv",  index=False)
    df_prod.to_csv(tmp_path / "data" / "wine_prod_con_pred.csv", index=False)

    _pipeline = _load("04_pipeline_monitoreo.py", "pipeline_monitoreo")
    resumen   = _pipeline.ejecutar_monitoreo("test_f1")

    assert "f1_ref"  in resumen
    assert "f1_prod" in resumen