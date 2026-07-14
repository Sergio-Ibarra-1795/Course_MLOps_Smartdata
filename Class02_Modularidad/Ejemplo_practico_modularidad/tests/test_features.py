import pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from features import crear_score_retrasos, crear_categorias_edad

def test_score_retrasos_pesos_correctos():
    df = pd.DataFrame({'Nro_prestao_retrasados': [1,0],
                       'Nro_retraso_60dias': [0,1],
                       'Nro_retraso_ultm3anios': [0,0]})
    r = crear_score_retrasos(df)
    assert r.loc[0,'Score_retrasos'] == 3  # 1*3 + 0*5 + 0*2
    assert r.loc[1,'Score_retrasos'] == 5  # 0*3 + 1*5 + 0*2

def test_score_retrasos_cliente_sin_historial():
    df = pd.DataFrame({'Nro_prestao_retrasados':[0],
                       'Nro_retraso_60dias':[0],'Nro_retraso_ultm3anios':[0]})
    assert crear_score_retrasos(df).loc[0,'Score_retrasos'] == 0

def test_categorias_edad_grupos_correctos():
    df = pd.DataFrame({'Edad': [20,30,40,50,60,70]})
    cats = crear_categorias_edad(df)['Edad_cat'].astype(str).tolist()
    assert cats[0] == '<25'
    assert cats[2] == '35-45'
    assert cats[5] == '>65'

def test_categorias_edad_crea_dummies():
    df = pd.DataFrame({'Edad': [25,35,45,55,65,75]})
    r = crear_categorias_edad(df)
    assert len([c for c in r.columns if c.startswith('Edad_')]) > 0