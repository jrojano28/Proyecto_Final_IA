FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear carpetas necesarias con permisos para escritura
RUN mkdir -p uploads models static/css \
    && chmod -R 777 uploads models static

# Hugging Face Spaces requiere puerto 7860
EXPOSE 7860

# Usar gunicorn en producción
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
