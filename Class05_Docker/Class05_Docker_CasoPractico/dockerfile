# Dockerfile — API de Riesgo Crediticio Banco Wiesse

# ── Imagen base: Python 3.10 slim (liviana, sin herramientas innecesarias) ──
FROM python:3.10-slim

# ── Metadatos ────────────────────────────────────────────────────────────────
LABEL maintainer="MLOps Banco Wiesse"
LABEL description="API de predicción de default crediticio"
LABEL version="1.0.0"

# ── Variables de entorno ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ── Directorio de trabajo dentro del contenedor ───────────────────────────────
WORKDIR /app

# ── Instalar dependencias del sistema (mínimas) ───────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Copiar e instalar dependencias Python ─────────────────────────────────────
# IMPORTANTE: copiar requirements.txt ANTES del código
# Así Docker cachea esta capa y no reinstala si solo cambia el código
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar el código de la API ────────────────────────────────────────────────
COPY api/ ./api/
COPY config.py .

# ── Copiar artefactos del pipeline (modelo + features) ────────────────────────
COPY artifacts/modelo.pkl   ./artifacts/
COPY artifacts/features.csv ./artifacts/
COPY artifacts/metrics.json ./artifacts/

# ── Puerto que expone el contenedor ──────────────────────────────────────────
EXPOSE 8000

# ── Health check para que Docker sepa si el contenedor está sano ──────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')"

# ── Comando de inicio ─────────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]