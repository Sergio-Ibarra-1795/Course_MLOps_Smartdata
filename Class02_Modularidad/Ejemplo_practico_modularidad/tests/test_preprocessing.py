import pytest, pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import limpiar_na_strings, winsorizar_columnas, imputar_nulos

def test_limpiar_na_strings_convierte_strings():
    df = pd.DataFrame({'Mto_ingreso_mensual': ['1000','NA','2000','na'],
                       'Nro_dependiente': ['2','N/A','0','3']})
    r = limpiar_na_strings(df)
    assert r['Mto_ingreso_mensual'].isnull().sum() == 2
    assert r['Nro_dependiente'].isnull().sum() == 1

def test_limpiar_na_strings_no_afecta_numericos():
    df = pd.DataFrame({'Mto_ingreso_mensual': ['500','1200','3000'],
                       'Nro_dependiente': ['0','1','2']})
    r = limpiar_na_strings(df)
    assert r['Mto_ingreso_mensual'].isnull().sum() == 0
    assert r.loc[0, 'Mto_ingreso_mensual'] == 500.0

def test_winsorizar_elimina_outliers():
    df = pd.DataFrame({'Prct_uso_tc': [0.1,0.2,0.3,0.4,0.5,10.0]})
    r = winsorizar_columnas(df, columnas=['Prct_uso_tc'], p_low=0.05, p_high=0.95)
    assert r['Prct_uso_tc'].max() < 10.0

def test_winsorizar_conserva_distribucion_normal():
    df = pd.DataFrame({'Prct_uso_tc': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]})
    r = winsorizar_columnas(df, columnas=['Prct_uso_tc'])
    assert abs(r['Prct_uso_tc'].mean() - df['Prct_uso_tc'].mean()) < 0.1

def test_imputar_nulos_mediana():
    df = pd.DataFrame({'Mto_ingreso_mensual': [1000.0,2000.0,np.nan,4000.0,np.nan]})
    r = imputar_nulos(df, {'Mto_ingreso_mensual': 'median'})
    assert r['Mto_ingreso_mensual'].isnull().sum() == 0
    assert r.loc[2, 'Mto_ingreso_mensual'] == 2000.0