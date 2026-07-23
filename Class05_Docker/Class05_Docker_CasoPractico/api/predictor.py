"""
api/predictor.py — Carga el modelo.pkl y ejecuta predicciones.
El modelo se carga UNA sola vez al arrancar la API (startup event).
Las predicciones subsecuentes no vuelven a leer el disco.
"""
import pickle, json, logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Rutas relativas al directorio del proyecto
MODEL_PATH   = Path('artifacts/modelo.pkl')
METRICS_PATH = Path('artifacts/metrics.json')

# Umbrales de decisión
UMBRAL_RECHAZAR = 0.60   # score >= 0.60 -> RECHAZAR
UMBRAL_REVISAR  = 0.40   # score >= 0.40 -> REVISAR


class Predictor:
    """Singleton que encapsula la lógica de predicción."""

    def __init__(self):
        self.modelo         = None
        self.metricas       = {}
        self.features_orden = []

    def cargar(self) -> None:
        """Carga modelo y metadatos desde disco. Llamar UNA vez en startup."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f'Modelo no encontrado: {MODEL_PATH}. '
                f'Ejecuta primero: python run_pipeline.py'
            )

        with open(MODEL_PATH, 'rb') as f:
            self.modelo = pickle.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                self.metricas = json.load(f)

        # Orden de features del entrenamiento
        features_path = Path('artifacts/features.csv')
        if features_path.exists():
            df_feat = pd.read_csv(features_path, nrows=0)
            excluir = ['Default', 'ID']
            self.features_orden = [c for c in df_feat.columns if c not in excluir]

        log.info('Modelo cargado: %s', type(self.modelo).__name__)
        log.info('Features en orden: %s...', self.features_orden[:3])

    def predecir(self, datos: dict) -> dict:
        """Recibe dict con los campos del cliente y retorna la predicción."""
        if self.modelo is None:
            raise RuntimeError('Modelo no cargado. Llama a cargar() primero.')

        # Calcular features derivadas
        datos['Score_retrasos'] = (
            datos.get('Nro_prestao_retrasados', 0) * 3 +
            datos.get('Nro_retraso_60dias', 0)     * 5 +
            datos.get('Nro_retraso_ultm3anios', 0) * 2
        )

        # Construir DataFrame en el orden correcto
        if self.features_orden:
            X = pd.DataFrame([{k: datos.get(k, 0.0) for k in self.features_orden}])
        else:
            X = pd.DataFrame([datos])

        # Score de riesgo (probabilidad de default)
        proba = float(self.modelo.predict_proba(X)[0][1])

        # Lógica de decisión basada en umbrales
        if proba >= UMBRAL_RECHAZAR:
            decision     = 'RECHAZAR'
            nivel_riesgo = 'MUY ALTO' if proba >= 0.80 else 'ALTO'
        elif proba >= UMBRAL_REVISAR:
            decision     = 'REVISAR'
            nivel_riesgo = 'MODERADO'
        else:
            decision     = 'APROBAR'
            nivel_riesgo = 'BAJO'

        return {
            'score_riesgo':         round(proba, 4),
            'decision':             decision,
            'nivel_riesgo':         nivel_riesgo,
            'probabilidad_default': round(proba, 4),
            'umbral_usado':         UMBRAL_RECHAZAR,
            'modelo':               type(self.modelo).__name__,
        }


# Instancia global — singleton
predictor = Predictor()
