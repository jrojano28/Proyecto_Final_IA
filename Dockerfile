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

# Puerto por defecto (Koyeb lo sobreescribe con $PORT)
EXPOSE 8000

# Usar variable de entorno PORT (Koyeb la inyecta automáticamente)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 120 app:app"]
