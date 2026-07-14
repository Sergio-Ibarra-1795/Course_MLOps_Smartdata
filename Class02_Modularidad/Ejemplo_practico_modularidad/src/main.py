"""
main.py — Pipeline completo de prediccion de default crediticio.
Ejecutar: cd /workspaces/banco-wiesse-mlops/src && python main.py
"""
import logging, sys
from data_loader import cargar_datos
from preprocessing import (limpiar_na_strings, winsorizar_columnas,
                            imputar_nulos, dividir_y_balancear)
from features import (crear_score_retrasos, crear_categorias_edad,
                      crear_categorias_dependientes,
                      estandarizar_variables, seleccionar_features)
from train import comparar_modelos, analizar_errores

RUTA_DATOS       = 'data/df.csv'
TARGET           = 'Default'
ID_COL           = 'ID'
K_FEATURES       = 15
TECNICA_BALANCEO = 'smote'

def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline_crediticio.log'),
        ],
    )

def main():
    configurar_logging()
    log = logging.getLogger(__name__)
    log.info('=== INICIO PIPELINE RIESGO CREDITICIO BANCO WIESSE ===')

    df = cargar_datos(RUTA_DATOS)
    df = limpiar_na_strings(df)
    df = winsorizar_columnas(df)
    df = imputar_nulos(df)
    df = crear_score_retrasos(df)
    df = crear_categorias_edad(df)
    df = crear_categorias_dependientes(df)
    df = estandarizar_variables(df)

    cols_excluir = [TARGET, ID_COL, 'Edad_cat', 'Deps_cat']
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    X = X.select_dtypes(include=['number'])
    y = df[TARGET]
    X_sel = X[seleccionar_features(X, y, k=K_FEATURES)]

    X_train, X_test, y_train, y_test = dividir_y_balancear(
        X_sel, y, test_size=0.2, tecnica=TECNICA_BALANCEO)

    resultados = comparar_modelos(X_train, y_train, X_test, y_test)
    mejor_nombre = max(resultados, key=lambda k: resultados[k]['metricas']['f1'])
    mejor_modelo = resultados[mejor_nombre]['modelo']
    errores      = analizar_errores(mejor_modelo, X_test, y_test)

    log.info('=== RESULTADOS FINALES ===')
    for nombre, res in resultados.items():
        m = res['metricas']
        log.info('%s: F1=%.4f Recall=%.4f AUC=%s', nombre, m['f1'], m['recall'], m['roc_auc'])
    log.info('Mejor modelo: %s | FN=%d | FP=%d', mejor_nombre, errores['FN'], errores['FP'])
    log.info('=== PIPELINE COMPLETADO ===')

if __name__ == '__main__':
    main()