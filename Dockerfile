FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema si son necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear las carpetas de almacenamiento y asignar permisos totales para evitar problemas de escritura
RUN mkdir -p uploads models static && chmod -R 777 uploads models static

# Hugging Face Spaces requiere escuchar en el puerto 7860
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
