# Usa una imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV BACKEND_URL=http://localhost:8000

# Copia el archivo requirements.txt en el contenedor
COPY requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de la aplicación en el contenedor
COPY . .

# Exponer el puerto que utiliza Gradio
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "interface.py"]