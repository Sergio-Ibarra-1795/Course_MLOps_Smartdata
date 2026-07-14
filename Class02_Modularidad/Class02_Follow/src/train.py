"""train.py — Entrenamiento y evaluacion de modelos (Parte 4 del notebook)."""
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report)

logger = logging.getLogger(__name__)

MODELOS_DEFAULT = {
    'Regresion Logistica': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced', C=1.0),
    'Random Forest': RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=10,
        class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5),
}

def entrenar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    """Entrena y evalua un modelo. Returns (modelo_fit, dict_metricas)."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = {
        'modelo':    nombre,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
    }
    try:
        metricas['roc_auc'] = round(roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1]), 4)
    except AttributeError:
        metricas['roc_auc'] = None
    logger.info('[%s] Recall=%.4f | F1=%.4f | ROC-AUC=%s',
                nombre, metricas['recall'], metricas['f1'], metricas['roc_auc'])
    return modelo, metricas

def comparar_modelos(X_train, y_train, X_test, y_test, modelos=None):
    """Entrena todos los modelos y devuelve resultados ordenados por F1."""
    resultados = {}
    for nombre, modelo in (modelos or MODELOS_DEFAULT).items():
        m_fit, mets = entrenar_modelo(nombre, modelo, X_train, y_train, X_test, y_test)
        resultados[nombre] = {'modelo': m_fit, 'metricas': mets}
    mejor = max(resultados, key=lambda k: resultados[k]['metricas']['f1'])
    logger.info('Mejor modelo: %s (F1=%.4f)', mejor, resultados[mejor]['metricas']['f1'])
    return resultados

def analizar_errores(modelo, X_test, y_test):
    """Matriz de confusion y tasas de error (Parte 4.6 del notebook)."""
    y_pred = modelo.predict(X_test)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    return {
        'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP),
        'fpr':    round(FP/(FP+TN), 4) if (FP+TN) > 0 else 0,
        'fnr':    round(FN/(FN+TP), 4) if (FN+TP) > 0 else 0,
        'recall': round(TP/(TP+FN), 4) if (TP+FN) > 0 else 0,
        'reporte': classification_report(y_test, y_pred,
                       target_names=['No Default', 'Default']),
    }