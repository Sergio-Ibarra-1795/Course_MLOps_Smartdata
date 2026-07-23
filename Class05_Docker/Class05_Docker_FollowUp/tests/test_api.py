"""
tests/test_api.py — Tests del endpoint /predecir con TestClient de FastAPI.

No requiere Docker corriendo. Usa TestClient que levanta la app en memoria.
Ejecutar: pytest tests/test_api.py -v

NOTA: Se usa el fixture `client` con scope="session" y context manager
para que el lifespan (startup) se ejecute y el modelo se cargue correctamente.
"""
import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.app import app

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Levanta la app completa con lifespan (carga el modelo en startup)."""
    with TestClient(app) as c:
        yield c

# ── Datos de prueba ───────────────────────────────────────────────────────────

CLIENTE_ALTO_RIESGO = {
    'Prct_uso_tc':                0.92,
    'Prct_deuda_vs_ingresos':     0.78,
    'Mto_ingreso_mensual':        1200.0,
    'Nro_dependiente':            3.0,
    'Edad':                       28,
    'Nro_prestao_retrasados':     5,
    'Nro_prod_financieros_deuda': 10,
    'Nro_retraso_60dias':         3,
    'Nro_creditos_hipotecarios':  0,
    'Nro_retraso_ultm3anios':     4,
}

CLIENTE_BAJO_RIESGO = {
    'Prct_uso_tc':                0.15,
    'Prct_deuda_vs_ingresos':     0.12,
    'Mto_ingreso_mensual':        5500.0,
    'Nro_dependiente':            1.0,
    'Edad':                       45,
    'Nro_prestao_retrasados':     0,
    'Nro_prod_financieros_deuda': 3,
    'Nro_retraso_60dias':         0,
    'Nro_creditos_hipotecarios':  2,
    'Nro_retraso_ultm3anios':     0,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

DECISIONES_VALIDAS    = {'APROBAR', 'REVISAR', 'RECHAZAR'}
NIVELES_RIESGO_VALIDOS = {'BAJO', 'MODERADO', 'ALTO', 'MUY ALTO'}

def assert_respuesta_valida(body: dict):
    """Verifica que la estructura y tipos de la respuesta sean correctos."""
    assert 'score_riesgo'         in body
    assert 'decision'             in body
    assert 'nivel_riesgo'         in body
    assert 'probabilidad_default' in body
    assert 'umbral_usado'         in body
    assert 'modelo'               in body
    assert 0.0 <= body['score_riesgo'] <= 1.0
    assert body['decision']    in DECISIONES_VALIDAS
    assert body['nivel_riesgo'] in NIVELES_RIESGO_VALIDOS
    # Consistencia interna: score y decision deben ser coherentes
    score = body['score_riesgo']
    umbral = body['umbral_usado']
    if score >= umbral:
        assert body['decision'] == 'RECHAZAR'
    elif score >= 0.40:
        assert body['decision'] == 'REVISAR'
    else:
        assert body['decision'] == 'APROBAR'

# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    """El endpoint /health debe retornar status ok."""
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert 'modelo' in data
    assert 'recall_entrenamiento' in data
    assert isinstance(data['recall_entrenamiento'], float)


def test_root_info(client):
    """El endpoint raíz debe retornar info básica de la API."""
    r = client.get('/')
    assert r.status_code == 200
    data = r.json()
    assert 'api' in data
    assert 'docs' in data


def test_predecir_alto_riesgo_estructura(client):
    """POST /predecir debe retornar 200 con estructura correcta."""
    r = client.post('/predecir', json=CLIENTE_ALTO_RIESGO)
    assert r.status_code == 200
    assert_respuesta_valida(r.json())


def test_predecir_bajo_riesgo_estructura(client):
    """POST /predecir debe retornar 200 con estructura correcta."""
    r = client.post('/predecir', json=CLIENTE_BAJO_RIESGO)
    assert r.status_code == 200
    assert_respuesta_valida(r.json())


def test_decision_coherente_con_score(client):
    """La decisión debe ser siempre coherente con el score y el umbral."""
    for payload in [CLIENTE_ALTO_RIESGO, CLIENTE_BAJO_RIESGO]:
        r = client.post('/predecir', json=payload)
        assert r.status_code == 200
        assert_respuesta_valida(r.json())


def test_score_es_determinista(client):
    """El mismo cliente debe recibir siempre el mismo score."""
    r1 = client.post('/predecir', json=CLIENTE_ALTO_RIESGO)
    r2 = client.post('/predecir', json=CLIENTE_ALTO_RIESGO)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()['score_riesgo'] == r2.json()['score_riesgo']


def test_validacion_campos_invalidos(client):
    """Campos inválidos deben retornar HTTP 422 Unprocessable Entity."""
    r = client.post('/predecir', json={'Prct_uso_tc': -999})
    assert r.status_code == 422


def test_validacion_edad_menor_18(client):
    """Edad menor a 18 debe retornar HTTP 422."""
    datos = CLIENTE_BAJO_RIESGO.copy()
    datos['Edad'] = 15
    r = client.post('/predecir', json=datos)
    assert r.status_code == 422


def test_validacion_ingreso_negativo(client):
    """Ingreso mensual negativo debe retornar HTTP 422."""
    datos = CLIENTE_BAJO_RIESGO.copy()
    datos['Mto_ingreso_mensual'] = -500.0
    r = client.post('/predecir', json=datos)
    assert r.status_code == 422


def test_swagger_ui_disponible(client):
    """La documentación Swagger UI debe estar disponible en /docs."""
    r = client.get('/docs')
    assert r.status_code == 200


def test_openapi_schema(client):
    """El schema OpenAPI debe estar disponible en /openapi.json."""
    r = client.get('/openapi.json')
    assert r.status_code == 200
    schema = r.json()
    assert 'paths' in schema
    assert '/predecir' in schema['paths']
    assert '/health'   in schema['paths']