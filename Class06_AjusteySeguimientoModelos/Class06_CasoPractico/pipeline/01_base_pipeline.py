"""
Secciones 1-7 — Pipeline base sin tuning.

Carga (o genera) el dataset, hace el split 70/30, aplica SMOTE y
entrena un DecisionTree con parámetros por defecto.
Exporta base_data.pkl para que los scripts de tuning lo reutilicen.

Ejecutar: python src/01_base_pipeline.py
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, recall_score, precision_score,
)
from imblearn.over_sampling import SMOTE

# ── Sección 2: Cargar o generar dataset ──────────────────────────────────────
DATA_PATH = Path("data/Clasificacion_picos_intensidad.csv")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape}")
else:
    # Generar dataset sintético si no existe
    print("Dataset no encontrado — generando sintético...")
    np.random.seed(123)
    n = 5000
    df = pd.DataFrame({
        "intensidad":  np.random.normal(50, 15, n),
        "duracion":    np.random.exponential(10, n),
        "frecuencia":  np.random.uniform(1, 100, n),
        "amplitud":    np.random.normal(20, 5, n),
        "temperatura": np.random.normal(25, 8, n),
        "presion":     np.random.normal(1013, 20, n),
    })
    prob = 1 / (1 + np.exp(-(df["intensidad"] - 60) / 10))
    df["pico"] = (np.random.uniform(0, 1, n) < prob * 0.3).astype(int)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset sintético generado: {df.shape}")

# ── Sección 3: Exploración ────────────────────────────────────────────────────
print(f"\nShape       : {df.shape}")
print(f"Columnas    : {list(df.columns)}")
print("\nDistribución del target:")
print(df["pico"].value_counts())
print(df["pico"].value_counts(normalize=True).round(4))
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))

# ── Sección 4: Split 70/30 ────────────────────────────────────────────────────
TARGET = "pico"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Asegurar que el target sea entero (puede llegar como float64 desde CSV)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=123, stratify=y
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Train target dist: {y_train.value_counts().to_dict()}")
print(f"Test  target dist: {y_test.value_counts().to_dict()}")

# ── Sección 5: UnderSampling y OverSampling manuales ─────────────────────────
# cast a int explícito para compatibilidad con numpy.random.choice
n_minority   = int(y_train.sum())
n_majority   = int((y_train == 0).sum())
idx_majority = y_train[y_train == 0].index
idx_minority = y_train[y_train == 1].index

np.random.seed(123)

# UnderSampling — reducir la clase mayoritaria al tamaño de la minoritaria
idx_under = np.random.choice(idx_majority, size=n_minority, replace=False)
X_under = pd.concat([X_train.loc[idx_under], X_train.loc[idx_minority]])
y_under = pd.concat([y_train.loc[idx_under], y_train.loc[idx_minority]])

# OverSampling — duplicar la clase minoritaria al tamaño de la mayoritaria
idx_over = np.random.choice(idx_minority, size=n_majority, replace=True)
X_over = pd.concat([X_train, X_train.loc[idx_over]])
y_over = pd.concat([y_train, y_train.loc[idx_over]])

print(f"\nUnderSampling: {X_under.shape} | dist: {y_under.value_counts().to_dict()}")
print(f"OverSampling : {X_over.shape}  | dist: {y_over.value_counts().to_dict()}")

# ── Sección 6: SMOTE ──────────────────────────────────────────────────────────
smote = SMOTE(k_neighbors=5, random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"\nPost-SMOTE: {X_train_smote.shape}")
print(f"Distribución: {pd.Series(y_train_smote).value_counts().to_dict()}")

# ── Sección 7: DecisionTree base (sin tuning) — LÍNEA BASE ───────────────────
print("\n" + "=" * 50)
print("SECCIÓN 7 — DecisionTree base (sin tuning)")
print("=" * 50)

resultados_base = {}

for nombre, X_bal, y_bal in [
    ("Originales",    X_train,       y_train),
    ("UnderSampling", X_under,       y_under),
    ("OverSampling",  X_over,        y_over),
    ("SMOTE",         X_train_smote, y_train_smote),
]:
    dt = DecisionTreeClassifier(random_state=123)
    dt.fit(X_bal, y_bal)
    y_pred = dt.predict(X_test)
    resultados_base[nombre] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    }
    print(f"\n[{nombre}]")
    print(f"  Accuracy : {resultados_base[nombre]['accuracy']:.4f}")
    print(f"  F1       : {resultados_base[nombre]['f1']:.4f}")
    print(f"  Recall   : {resultados_base[nombre]['recall']:.4f}")
    print(f"  Precision: {resultados_base[nombre]['precision']:.4f}")

print("\n=== REPORTE COMPLETO — SMOTE (línea base para tuning) ===")
dt_smote = DecisionTreeClassifier(random_state=123)
dt_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = dt_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote))

# ── Exportar datos para scripts de tuning ────────────────────────────────────
Path("src").mkdir(exist_ok=True)
with open("src/base_data.pkl", "wb") as f:
    pickle.dump((X_train_smote, y_train_smote, X_test, y_test), f)

print("✓ Datos exportados a src/base_data.pkl")
print("  Siguiente paso: python src/02_gridsearch_mlflow.py")
