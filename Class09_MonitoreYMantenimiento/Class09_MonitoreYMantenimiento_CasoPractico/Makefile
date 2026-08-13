# Makefile — Pipeline de Monitoreo de Drift — Wine Quality
.PHONY: all install lint test monitoreo pasos clean help

# ── Instalación ───────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	@echo "✓ Dependencias instaladas"

# ── Pipeline completo ─────────────────────────────────────────────────────────
all: install lint test monitoreo
	@echo "✓ Pipeline completo OK"

# ── Pasos individuales ────────────────────────────────────────────────────────
paso1:
	python src/01_cargar_datos.py

paso2:
	python src/02_entrenar_modelo.py

paso3:
	python src/03_reporte_drift.py

paso4:
	python src/04_pipeline_monitoreo.py $(LOTE)

paso5:
	python src/05_visualizacion_drift.py

# ── Pipeline completo en un comando ──────────────────────────────────────────
monitoreo:
	@echo "=== Pipeline de Monitoreo de Drift ==="
	python src/run_monitoreo.py $(LOTE)

# ── Calidad de código ─────────────────────────────────────────────────────────
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503,D100,D104

test:
	pytest tests/ -v --tb=short

# ── Limpieza ──────────────────────────────────────────────────────────────────
clean:
	rm -rf data/ artifacts/ reportes/ __pycache__
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

# ── Ayuda ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "=== Pipeline de Monitoreo — Wine Quality ==="
	@echo "  make install       — instalar dependencias"
	@echo "  make monitoreo     — pipeline completo (5 pasos)"
	@echo "  make monitoreo LOTE=wine_2024_Q2 — con nombre de lote"
	@echo "  make paso1         — descargar dataset UCI"
	@echo "  make paso2         — entrenar modelo"
	@echo "  make paso3         — reportes EvidentlyAI"
	@echo "  make paso4         — pipeline con alertas"
	@echo "  make paso5         — visualizaciones PSI/KS"
	@echo "  make lint          — linting con flake8"
	@echo "  make test          — tests unitarios"
	@echo "  make clean         — limpiar artefactos"
	@echo ""
	@echo "Reportes generados en reportes/:"
	@echo "  01_data_drift.html          (EvidentlyAI — interactivo)"
	@echo "  02_data_quality.html        (EvidentlyAI — interactivo)"
	@echo "  03_model_performance.html   (EvidentlyAI — interactivo)"
	@echo "  04_distribuciones_comparativas.png"
	@echo "  05_psi_barras.png"
	@echo ""
