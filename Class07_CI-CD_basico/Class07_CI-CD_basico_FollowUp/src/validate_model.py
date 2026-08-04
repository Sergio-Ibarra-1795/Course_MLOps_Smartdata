"""validate_model.py — Valida que las métricas superen el umbral mínimo.

Si falla, el pipeline CI/CD se detiene y no se construye la imagen Docker.
"""

import json
from pathlib import Path
import sys

# ==============================================================================
# CONFIGURACIÓN DE RUTAS DINÁMICAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


# ==============================================================================
# FUNCIÓN DE VALIDACIÓN (QUALITY GATE)
# ==============================================================================
def validate(metrics_path: Path = METRICS_PATH) -> None:
    """Valida métricas. Lanza SystemExit(1) si falla el quality gate."""
    if not metrics_path.exists():
        print(
            f"ERROR: {metrics_path} no existe. Ejecuta train_pipeline.py primero."
        )
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    umbral = m.get("recall_minimo", 0.70)
    recall = m.get("recall", 0.0)
    f1 = m.get("f1", 0.0)

    print("=" * 50)
    print(" QUALITY GATE — VALIDACIÓN DE MÉTRICAS")
    print("=" * 50)
    print(f" Recall    : {recall:.4f}  (umbral: >= {umbral})")
    print(f" F1-Score  : {f1:.4f}")
    print(f" Accuracy  : {m.get('accuracy', 0):.4f}")

    if recall < umbral:
        print(f"\n FALLO: Recall {recall:.4f} < umbral {umbral}")
        print(" El pipeline CI/CD se detiene. No se construye Docker.")
        sys.exit(1)

    print(f"\n APROBADO: Recall {recall:.4f} >= {umbral}")
    print(" Pipeline CI/CD continúa al build de Docker.")


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    validate()