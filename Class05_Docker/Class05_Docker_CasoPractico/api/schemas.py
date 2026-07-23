"""
api/schemas.py — Modelos Pydantic para validación de entrada y salida.
Pydantic valida automáticamente tipos y rangos.
Si falla -> HTTP 422 con mensaje claro, el modelo no recibe datos corruptos.
"""
from pydantic import BaseModel, Field
from typing import Literal


class ClienteInput(BaseModel):
    """Datos de un cliente para predecir probabilidad de default."""
    Prct_uso_tc:               float = Field(..., ge=0.0, le=10.0,
                                   description='Porcentaje uso tarjeta crédito')
    Prct_deuda_vs_ingresos:    float = Field(..., ge=0.0, le=10.0,
                                   description='Ratio deuda / ingresos')
    Mto_ingreso_mensual:       float = Field(..., gt=0.0,
                                   description='Ingreso mensual en USD')
    Nro_dependiente:           float = Field(..., ge=0.0, le=20.0,
                                   description='Número de dependientes')
    Edad:                      int   = Field(..., ge=18, le=100,
                                   description='Edad del cliente')
    Nro_prestao_retrasados:    int   = Field(..., ge=0,
                                   description='Préstamos con retraso >3 meses')
    Nro_prod_financieros_deuda: int  = Field(..., ge=0,
                                   description='Número de productos de deuda')
    Nro_retraso_60dias:        int   = Field(..., ge=0,
                                   description='Retrasos >60 días')
    Nro_creditos_hipotecarios: int   = Field(..., ge=0,
                                   description='Créditos hipotecarios')
    Nro_retraso_ultm3anios:    int   = Field(..., ge=0,
                                   description='Retrasos últimos 3 años')

    model_config = {'json_schema_extra': {'example': {
        'Prct_uso_tc':               0.85,
        'Prct_deuda_vs_ingresos':    0.62,
        'Mto_ingreso_mensual':       1800.0,
        'Nro_dependiente':           2.0,
        'Edad':                      32,
        'Nro_prestao_retrasados':    3,
        'Nro_prod_financieros_deuda': 8,
        'Nro_retraso_60dias':        1,
        'Nro_creditos_hipotecarios': 1,
        'Nro_retraso_ultm3anios':    2,
    }}}


class PrediccionOutput(BaseModel):
    """Respuesta de la API con el resultado de la predicción."""
    score_riesgo:          float                               # probabilidad de default [0-1]
    decision:              Literal['APROBAR', 'REVISAR', 'RECHAZAR']
    nivel_riesgo:          Literal['BAJO', 'MODERADO', 'ALTO', 'MUY ALTO']
    probabilidad_default:  float                               # mismo que score_riesgo, más descriptivo
    umbral_usado:          float                               # UMBRAL_RECHAZAR = 0.60
    modelo:                str                                 # tipo de algoritmo usado


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""
    status:                str
    modelo:                str
    version:               str
    recall_entrenamiento:  float
