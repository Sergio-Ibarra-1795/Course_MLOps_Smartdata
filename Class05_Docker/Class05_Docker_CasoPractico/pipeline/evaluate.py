"""
Etapa 4 — Evaluación, quality gate y reporte.

Entrada : artifacts/modelo.pkl + artifacts/X_test.csv + artifacts/y_test.csv
Salida  : artifacts/metrics.json + artifacts/reporte_evaluacion.png

Ejecutar: python pipeline/evaluate.py
"""
import sys, logging, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix)
import config as C

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | EVALUATE | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


def validar_recall(recall: float) -> None:
    """Quality gate: falla si Recall < umbral mínimo."""
    if recall < C.RECALL_MINIMO:
        raise ValueError(
            f'QUALITY GATE FALLIDO: Recall={recall:.4f} < umbral={C.RECALL_MINIMO}. '
            f'El modelo no cumple el estándar mínimo de calidad para producción.')
    log.info('  ✓ Quality gate OK: Recall=%.4f >= %.2f', recall, C.RECALL_MINIMO)


def generar_reporte(metricas: dict, cm) -> None:
    """Genera reporte visual PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Reporte de Evaluación — Pipeline Banco Wiesse',
                 fontsize=13, fontweight='bold')
    mets   = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    vals   = [metricas[m] for m in mets]
    colors = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6']
    bars   = axes[0].bar(mets, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].set_ylim([0, 1.15])
    axes[0].set_title('Métricas del Modelo', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=C.RECALL_MINIMO, color='red', linestyle='--',
                    alpha=0.7, label=f'Recall mínimo ({C.RECALL_MINIMO})')
    axes[0].legend(fontsize=8)
    axes[1].imshow(cm, interpolation='nearest', cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i,j]), ha='center', va='center',
                         fontsize=14, fontweight='bold',
                         color='white' if cm[i,j] > cm.max()/2 else 'black')
    axes[1].set_xticks([0,1]); axes[1].set_yticks([0,1])
    axes[1].set_xticklabels(['No Default','Default'])
    axes[1].set_yticklabels(['No Default','Default'])
    axes[1].set_xlabel('Predicción'); axes[1].set_ylabel('Real')
    axes[1].set_title('Matriz de Confusión', fontweight='bold')
    FN, FP = metricas['FN'], metricas['FP']
    costo  = FN*C.COSTO_FN + FP*C.COSTO_FP
    axes[1].text(0.5, -0.18,
        f'FN={FN}×${C.COSTO_FN:,} + FP={FP}×${C.COSTO_FP:,} = ${costo:,}',
        transform=axes[1].transAxes, ha='center', fontsize=8.5,
        color='#c0392b', fontweight='bold')
    plt.tight_layout()
    plt.savefig(C.REPORT_PATH, dpi=120, bbox_inches='tight')
    plt.close()
    log.info('Reporte guardado: %s', C.REPORT_PATH)


def run() -> dict:
    """Ejecuta la etapa de evaluación."""
    log.info('=== ETAPA 4: EVALUACIÓN ===')
    for f in [C.MODEL_PATH, C.X_TEST_PATH, C.Y_TEST_PATH]:
        if not Path(f).exists():
            raise FileNotFoundError(f'Etapa 3 no completada: {f}')
    with open(C.MODEL_PATH, 'rb') as f:
        modelo = pickle.load(f)
    X_test = pd.read_csv(C.X_TEST_PATH)
    y_test = pd.read_csv(C.Y_TEST_PATH)['y_test']
    y_pred = modelo.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    metricas = {
        'accuracy':       round(accuracy_score(y_test, y_pred), 4),
        'precision':      round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall':         round(recall_score(y_test, y_pred), 4),
        'f1':             round(f1_score(y_test, y_pred), 4),
        'roc_auc':        round(roc_auc_score(y_test, modelo.predict_proba(X_test)[:,1]), 4),
        'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP),
        'costo_estimado': int(FN*C.COSTO_FN + FP*C.COSTO_FP),
        'modelo':         type(modelo).__name__,
        'recall_minimo':  C.RECALL_MINIMO,
    }
    log.info('  Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f',
             metricas['accuracy'], metricas['precision'],
             metricas['recall'],   metricas['f1'])
    log.info('  FN=%d  FP=%d  Costo=$%s', FN, FP, f"{metricas['costo_estimado']:,}")
    validar_recall(metricas['recall'])
    with open(C.METRICS_PATH, 'w') as f:
        json.dump(metricas, f, indent=2)
    log.info('Métricas guardadas: %s', C.METRICS_PATH)
    generar_reporte(metricas, cm)
    log.info('=== ETAPA 4 COMPLETADA ===')
    return metricas


if __name__ == '__main__':
    run()
