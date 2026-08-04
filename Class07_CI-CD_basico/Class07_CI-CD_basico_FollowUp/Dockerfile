# Dockerfile — API de Picos de Intensidad
# Usado por GitHub Actions (Job 3: build-docker) y por ejecuciones locales con Docker

FROM python:3.10-slim

LABEL maintainer="MLOps Picos Intensidad"
LABEL description="Imagen CI/CD del modelo de clasificación de picos de intensidad"
LABEL version="1.0.0"

# Previene archivos .pyc innecesarios y asegura logs en tiempo real (unbuffered)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Instalar dependencias del sistema y limpiar caché de apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar e instalar dependencias de Python (aprovecha la capa de caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copiar código fuente y artefactos
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# 4. Verificación opcional durante el build (no rompe si se construye localmente sin artefactos)
RUN python -c "import pickle; \
    m = pickle.load(open('artifacts/modelo.pkl', 'rb')); \
    print('Modelo cargado OK:', type(m).__name__)" || echo "artifacts/ no disponible en build local"

EXPOSE 8000

# 5. Comando por defecto: Smoke test de verificación al iniciar el contenedor
CMD ["python", "-c", \
     "import pickle; m=pickle.load(open('artifacts/modelo.pkl','rb')); \
      print('Modelo listo:', type(m).__name__)"]