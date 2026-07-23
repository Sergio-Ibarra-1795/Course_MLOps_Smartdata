"""
Etapa 3 — Entrenamiento de modelos y registro en MLflow.

Entrada : artifacts/features.csv
Salida  : artifacts/modelo.pkl  +  mlruns/<run_id>/

Ejecutar: python pipeline/train.py
"""
import sys, logging, pickle
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow, mlflow.sklearn
import pandas as pd
from imblearn.over_sampling  import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import config as C

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | TRAIN    | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

MODELOS = {
    'LogisticRegression': LogisticRegression(
        random_state=C.RANDOM_STATE, max_iter=1000, class_weight='balanced'),
    'RandomForest':       RandomForestClassifier(
        random_state=C.RANDOM_STATE, n_estimators=100, max_depth=10,
        class_weight='balanced', n_jobs=-1),
    'GradientBoosting':   GradientBoostingClassifier(
        random_state=C.RANDOM_STATE, n_estimators=100, learning_rate=0.1, max_depth=5),
}


def balancear(X_train, y_train):
    sampler_map = {
        'smote':         SMOTE(random_state=C.RANDOM_STATE),
        'undersampling': RandomUnderSampler(random_state=C.RANDOM_STATE),
        'oversampling':  RandomOverSampler(random_state=C.RANDOM_STATE),
    }
    sampler = sampler_map.get(C.TECNICA_BALANCEO, SMOTE(random_state=C.RANDOM_STATE))
    X_b, y_b = sampler.fit_resample(X_train, y_train)
    log.info('Balanceo %s: %s → %s',
             C.TECNICA_BALANCEO, dict(Counter(y_train)), dict(Counter(y_b)))
    return X_b, y_b


def run():
    """Ejecuta la etapa de entrenamiento."""
    log.info('=== ETAPA 3: ENTRENAMIENTO ===')
    if not C.FEATURES_PATH.exists():
        raise FileNotFoundError(f'Etapa 2 no completada: {C.FEATURES_PATH}')
    df   = pd.read_csv(C.FEATURES_PATH)
    cols = [c for c in df.columns if c not in [C.TARGET, C.ID_COL]]
    X, y = df[cols], df[C.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y)
    X_train_b, y_train_b = balancear(X_train, y_train)

    mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=C.MLFLOW_RUN_NAME):
        mlflow.log_params({'k_features': C.K_FEATURES, 'test_size': C.TEST_SIZE,
                           'tecnica_balanceo': C.TECNICA_BALANCEO,
                           'random_state': C.RANDOM_STATE})
        resultados = {}
        for nombre, modelo in MODELOS.items():
            modelo.fit(X_train_b, y_train_b)
            y_pred = modelo.predict(X_test)
            resultados[nombre] = {
                'modelo': modelo,
                'f1':     round(f1_score(y_test, y_pred), 4),
                'recall': round(recall_score(y_test, y_pred), 4),
                'acc':    round(accuracy_score(y_test, y_pred), 4),
            }
            log.info('  %s: F1=%.4f Recall=%.4f', nombre,
                     resultados[nombre]['f1'], resultados[nombre]['recall'])
        mejor        = max(resultados, key=lambda k: resultados[k]['f1'])
        mejor_modelo = resultados[mejor]['modelo']
        mlflow.log_metrics({'f1': resultados[mejor]['f1'],
                            'recall': resultados[mejor]['recall']})
        mlflow.log_param('mejor_algoritmo', mejor)
        mlflow.sklearn.log_model(mejor_modelo, 'modelo')
        log.info('Mejor: %s | F1=%.4f | Recall=%.4f',
                 mejor, resultados[mejor]['f1'], resultados[mejor]['recall'])

    with open(C.MODEL_PATH, 'wb') as f:
        pickle.dump(mejor_modelo, f)
    pd.DataFrame(X_test).to_csv(C.X_TEST_PATH, index=False)
    pd.DataFrame({'y_test': y_test.values}).to_csv(C.Y_TEST_PATH, index=False)
    log.info('Artefactos guardados: modelo.pkl | X_test.csv | y_test.csv')
    log.info('=== ETAPA 3 COMPLETADA ===')
    return mejor_modelo, X_test, y_test


if __name__ == '__main__':
    run()
