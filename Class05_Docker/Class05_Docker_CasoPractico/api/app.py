"""
api/app.py — API REST del sistema de predicción de default crediticio.

Endpoints:
    POST /predecir  -> score de riesgo del cliente
    GET  /health    -> estado de la API y metadatos del modelo
    GET  /          -> info básica
    GET  /docs      -> Swagger UI (autogenerado por FastAPI)

Ejecutar en Codespace:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Con Docker:
    docker compose up --build
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ClienteInput, PrediccionOutput, HealthResponse
from api.predictor import predictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | API | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la API (startup event)."""
    log.info('Cargando modelo de riesgo crediticio...')
    predictor.cargar()
    log.info('API lista para recibir predicciones')
    yield
    log.info('API cerrando')


app = FastAPI(
    title='API de Riesgo Crediticio — Banco Wiesse',
    description=(
        'Predicción de probabilidad de default crediticio. '
        'Modelo entrenado con el pipeline MLOps del proyecto Banco Wiesse.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)

# Permitir llamadas desde cualquier origen (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', tags=['Info'])
def root():
    """Información básica de la API."""
    return {
        'api':       'Banco Wiesse — Predicción de Default Crediticio',
        'version':   '1.0.0',
        'docs':      '/docs',
        'health':    '/health',
        'prediccion': 'POST /predecir',
    }


@app.get('/health', response_model=HealthResponse, tags=['Salud'])
def health():
    """Estado de la API y metadatos del modelo en producción."""
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail='Modelo no cargado')
    return HealthResponse(
        status='ok',
        modelo=type(predictor.modelo).__name__,
        version='1.0.0',
        recall_entrenamiento=predictor.metricas.get('recall', 0.0),
    )


@app.post('/predecir', response_model=PrediccionOutput, tags=['Prediccion'])
def predecir(cliente: ClienteInput):
    """
    Predice la probabilidad de default de un cliente.

    - **score_riesgo**: probabilidad de default [0.0 - 1.0]
    - **decision**: APROBAR | REVISAR | RECHAZAR
    - **nivel_riesgo**: BAJO | MODERADO | ALTO | MUY ALTO
    """
    try:
        datos     = cliente.model_dump()
        resultado = predictor.predecir(datos)
        log.info(
            'Predicción: score=%.4f decision=%s',
            resultado['score_riesgo'], resultado['decision'],
        )
        return PrediccionOutput(**resultado)
    except Exception as e:
        log.error('Error en predicción: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e))
