# Documentación de la aplicación web con Gradio y FastAPI

## Instrucciones de ejecución

Para ejecutar los contenedores de docker con este proyecto se requiere situarse en una consola en el directorio que contiene este documento (`/app/`) y ademas contar con un modelo previamente entrenado con la ejecución del pipeline de `/airflow`, en la carpeta `/airflow/mlruns/models`, y seguir los siguientes pasos.

1. Buildear la imagen: `docker compose build`

2. Ejecutar los contenedores: `docker compose up`

3. Acceder a la interfaz de Gradio en `http://localhost:7860`