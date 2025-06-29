# Utiliza una imagen base con Python instalado
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /root/airflow

# Establece la variable de entorno AIRFLOW_HOME
ENV AIRFLOW_HOME=/root/airflow

# Desactiva los DAGs de ejemplo
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instala Apache Airflow
RUN pip install "apache-airflow[password,sqlite]==2.8.1" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.1/constraints-3.10.txt"
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install pyarrow
RUN pip install numpy
RUN pip install joblib
RUN pip install gradio
RUN pip install xgboost
RUN pip install lightgbm
RUN pip install optuna
RUN pip install mlflow
RUN pip install optuna-integration[xgboost]
RUN pip install optuna-integration[lightgbm]

# Inicializa la base de datos de Airflow
RUN airflow db migrate

# Expone el puerto 8080 para el servidor web de Airflow
EXPOSE 8080

# Crea el usuario admin de Airflow
RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

# Copia las carpetas necesarias al contenedor
COPY ./dags $AIRFLOW_HOME/dags
COPY ./logs $AIRFLOW_HOME/logs
COPY ./plugins $AIRFLOW_HOME/plugins
COPY ./data $AIRFLOW_HOME/data

# Comando para iniciar el servidor web y el scheduler
CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]
