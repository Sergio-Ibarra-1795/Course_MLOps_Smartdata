# main.py (en la raíz del proyecto)
from pipeline import ingest, features, train, evaluate

def main():
    print("🚀 INICIANDO PIPELINE END-TO-END...")
    ingest.run()
    features.run()
    train.run()
    evaluate.run()
    print("✨ PIPELINE EJECUTADO CON ÉXITO")

if __name__ == '__main__':
    main()
    