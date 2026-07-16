"""
main.py — Pipeline completo con MLflow tracking.

Uso en terminal local:
    cd banco-wiesse-mlops
    python src/main.py
"""
import logging
import sys
import os
import pickle
import mlflow
import mlflow.sklearn

# 1. Asegurar que 'src/' esté en el path de Python para importar los módulos locales
DIRECTORIO_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRECTORIO_SRC)

from data_loader   import cargar_datos
from preprocessing import (limpiar_na_strings, winsorizar_columnas,
                            imputar_nulos, dividir_y_balancear)
from features      import (crear_score_retrasos, crear_categorias_edad,
                            crear_categorias_dependientes,
                            estandarizar_variables, seleccionar_features)
from train         import comparar_modelos, analizar_errores

# 2. Resolver rutas relativas al archivo main.py de forma robusta para evitar el CWD trap
RAIZ_PROYECTO = os.path.dirname(DIRECTORIO_SRC)
RUTA_DATASET_LOCAL = os.path.join(RAIZ_PROYECTO, 'data', 'Dataset Endeudamiento Crediticio.csv')
RUTA_MODELS_LOCAL = os.path.join(RAIZ_PROYECTO, 'models')

# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL EXPERIMENTO — cambia estos valores entre versiones
# ══════════════════════════════════════════════════════════════════════
PARAMS = {
    'ruta_datos':        RUTA_DATASET_LOCAL,
    'target':            'Default',
    'k_features':        15,
    'test_size':         0.2,
    'tecnica_balanceo':  'smote',
    'random_state':      42,
    'p_winsor_low':      0.05,
    'p_winsor_high':     0.95,
    'version_pipeline':  'v1-baseline',
}

MLFLOW_EXPERIMENT = 'banco-wiesse-crediticio'
MODELS_DIR        = RUTA_MODELS_LOCAL


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    configurar_logging()
    log = logging.getLogger(__name__)
    log.info('=== PIPELINE BANCO WIESSE — %s ===', PARAMS['version_pipeline'])

    # ── Configurar MLflow ──────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=PARAMS['version_pipeline']) as run:
        run_id = run.info.run_id
        log.info('MLflow Run ID: %s', run_id)

        # 1. Registrar parámetros
        mlflow.log_params(PARAMS)

        # 2. Cargar datos
        df = cargar_datos(PARAMS['ruta_datos'])
        mlflow.log_param('n_filas_dataset', len(df))
        mlflow.log_param('pct_default_dataset',
                         round(df[PARAMS['target']].mean(), 4))

        # 3. Limpieza
        df = limpiar_na_strings(df)
        df = winsorizar_columnas(df,
                                 p_low=PARAMS['p_winsor_low'],
                                 p_high=PARAMS['p_winsor_high'])
        df = imputar_nulos(df)

        # 4. Feature engineering
        df = crear_score_retrasos(df)
        df = crear_categorias_edad(df)
        df = crear_categorias_dependientes(df)
        df = estandarizar_variables(df)

        # 5. Selección de variables y split
        excluir = [PARAMS['target'], 'ID', 'Edad_cat', 'Deps_cat']
        X = df.drop(columns=[c for c in excluir if c in df.columns])
        X = X.select_dtypes(include=['number'])
        y = df[PARAMS['target']]

        top_feats = seleccionar_features(X, y, k=PARAMS['k_features'])
        X_sel = X[top_feats]
        mlflow.log_param('features_seleccionadas', str(top_feats[:5]) + '...')

        X_train, X_test, y_train, y_test = dividir_y_balancear(
            X_sel, y,
            test_size=PARAMS['test_size'],
            tecnica=PARAMS['tecnica_balanceo'],
            random_state=PARAMS['random_state'],
        )
        mlflow.log_param('n_train_balanceado', len(y_train))
        mlflow.log_param('n_test', len(y_test))

        # 6. Entrenar los 3 modelos
        resultados = comparar_modelos(X_train, y_train, X_test, y_test)

        # 7. Identificar mejor modelo
        mejor_nombre = max(resultados, key=lambda k: resultados[k]['metricas']['f1'])
        mejor_modelo  = resultados[mejor_nombre]['modelo']
        mejor_m       = resultados[mejor_nombre]['metricas']
        errores       = analizar_errores(mejor_modelo, X_test, y_test)

        # 8. Registrar MÉTRICAS en MLflow
        mlflow.log_metrics({
            'accuracy':        mejor_m['accuracy'],
            'precision':       mejor_m['precision'],
            'recall':          mejor_m['recall'],
            'f1':              mejor_m['f1'],
            'roc_auc':         mejor_m['roc_auc'] or 0.0,
            'false_negatives': float(errores['FN']),
            'false_positives': float(errores['FP']),
            'true_positives':  float(errores['TP']),
            'costo_estimado':  float(errores['FN'] * 10000 + errores['FP'] * 1000),
        })
        mlflow.log_param('mejor_algoritmo', mejor_nombre)

        # 9. Registrar el MODELO como artefacto
        mlflow.sklearn.log_model(
            mejor_modelo,
            artifact_path='modelo_crediticio',
        )

        # 10. Guardar modelo localmente
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"modelo_{PARAMS['version_pipeline']}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(mejor_modelo, f)
        mlflow.log_artifact(model_path)

        log.info('=== RESULTADO: %s | F1=%.4f | Recall=%.4f | FN=%d | FP=%d ===',
                 mejor_nombre, mejor_m['f1'], mejor_m['recall'],
                 errores['FN'], errores['FP'])
        log.info('Costo estimado: $%s', f"{errores['FN']*10000 + errores['FP']*1000:,}")

        return {
            'run_id':         run_id,
            'mejor_modelo':   mejor_nombre,
            'metricas':       mejor_m,
            'errores':        errores,
            'modelo_objeto':  mejor_modelo,
            'X_test':         X_test,
            'y_test':         y_test,
        }


if __name__ == '__main__':
    main()
