"""
pipeline/train.py — Etapa 3: Entrenamiento, Evaluación y Registro en MLflow.

Entrada : artifacts/features.csv
Salida  : artifacts/modelo.pkl, artifacts/X_test.csv, artifacts/y_test.csv

Ejecutar desde la raíz: python pipeline/train.py
"""
import sys
import logging
import pickle
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as C
from src.preprocessing import dividir_y_balancear
from src.train import comparar_modelos, analizar_errores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | TRAIN    | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def run():
    log.info('=== ETAPA 3: ENTRENAMIENTO Y MLFLOW ===')
    
    if not C.FEATURES_PATH.exists():
        raise FileNotFoundError(f'Falta el artefacto de la Etapa 2: {C.FEATURES_PATH}')
        
    df = pd.read_csv(C.FEATURES_PATH)
    cols = [c for c in df.columns if c not in [C.TARGET, C.ID_COL]]
    X, y = df[cols], df[C.TARGET]
    
    # 1. Split y Balanceo de datos usando src/preprocessing.py
    X_train_b, X_test, y_train_b, y_test = dividir_y_balancear(
        X, y, test_size=C.TEST_SIZE, tecnica=C.TECNICA_BALANCEO, random_state=C.RANDOM_STATE
    )
    
    # 2. Experimento con MLflow
    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name=C.MLFLOW_RUN_NAME):
        # Registrar hiperparámetros de la ejecución
        mlflow.log_params({
            'k_features': C.K_FEATURES,
            'test_size': C.TEST_SIZE,
            'tecnica_balanceo': C.TECNICA_BALANCEO,
            'random_state': C.RANDOM_STATE
        })
        
        # Entrenar y comparar usando src/train.py
        resultados = comparar_modelos(X_train_b, y_train_b, X_test, y_test)
        
        # Seleccionar al ganador por F1-Score
        mejor_nombre = max(resultados, key=lambda k: resultados[k]['metricas']['f1'])
        mejor_modelo = resultados[mejor_nombre]['modelo']
        metricas_mejor = resultados[mejor_nombre]['metricas']
        
        # Analizar matriz de confusión
        errores = analizar_errores(mejor_modelo, X_test, y_test)
        
        # Registrar métricas en MLflow
        mlflow.log_metrics({
            'f1': metricas_mejor['f1'],
            'recall': metricas_mejor['recall'],
            'accuracy': metricas_mejor['accuracy'],
            'fn': errores['FN'],
            'fp': errores['FP']
        })
        mlflow.log_param('mejor_algoritmo', mejor_nombre)
        mlflow.sklearn.log_model(mejor_modelo, 'modelo')
        
        log.info('Ganador: %s | F1=%.4f | Recall=%.4f', mejor_nombre, metricas_mejor['f1'], metricas_mejor['recall'])

    # 3. Guardar artefactos finales en la carpeta artifacts/
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.MODEL_PATH, 'wb') as f:
        pickle.dump(mejor_modelo, f)
        
    X_test.to_csv(C.X_TEST_PATH, index=False)
    pd.DataFrame({'y_test': y_test.values}).to_csv(C.Y_TEST_PATH, index=False)
    
    log.info('✓ Artefactos guardados: modelo.pkl | X_test.csv | y_test.csv')
    log.info('=== ETAPA 3 COMPLETADA ===')
    return mejor_modelo, X_test, y_test


if __name__ == '__main__':
    run()