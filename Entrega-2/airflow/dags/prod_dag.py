from datetime import datetime, date

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from utils.init_functions import create_folders
from utils.preprocess_functions import load_and_preprocess_data
from utils.holdout_functions import split_data
from utils.training_functions import train_model, branch_by_drift, predict_and_save, setup_experiment, select_best_model

dag =  DAG(
    dag_id = "sodai_dag",
    description="A dag to predict next week's sales",
    start_date = datetime.today(),
) 

# Task 1 - Print start statement
start_task = EmptyOperator(task_id="Starting_the_process", retries=2, dag=dag)


# Task 2 - Folder creation operator
folder_task = PythonOperator(
    task_id="Creating_folders",
    python_callable = create_folders,
    dag=dag
)

# Task 3 - Download data (dummy task)
download_task = EmptyOperator(task_id="Downloading_data", dag=dag)

# Task 4 - Loading data and saving in raw folder
load_preprocess_data_task = PythonOperator(
    task_id="Loading_and_preprocessing",
    python_callable=load_and_preprocess_data,
    dag=dag
)

# Task 5 - splitting the data
split_data_task = PythonOperator(
    task_id="Splitting_data",
    python_callable=split_data,
    dag=dag
    )

# Task 6 - branch by drift (evaluates then branches)
branch_by_drift_task = BranchPythonOperator(
    task_id="Branching_by_drift",
    python_callable=branch_by_drift,
    dag=dag
)

# Task n - create experiment

setup_experiment_task = PythonOperator(
    task_id = "Create_experiment_task",
    python_callable=setup_experiment,
    dag=dag
)

# Task 7.a - optimize and train model
train_lr_task = PythonOperator(
    task_id="Training_lr",
    python_callable=train_model,
    op_kwargs = {"model_string":"lr"},
    dag=dag
)

train_xgb_task = PythonOperator(
    task_id = "Training_xgb",
    python_callable=train_model,
    op_kwargs={"model_string":"xgb"},
    dag=dag
)

train_lgbm_task = PythonOperator(
    task_id="Training_lightgbm",
    python_callable=train_model,
    op_kwargs={"model_string": "lgbm"},
    dag=dag
)
# Task n? - select best model
select_best_model_task = PythonOperator(
    task_id="Selecting_best_model",
    python_callable=select_best_model,
    dag=dag,
    trigger_rule='all_success'
)

# Task 8 - prediction task (saves to csv)

predict_task = PythonOperator(
    task_id='Prediction_task',
    python_callable = predict_and_save,
    dag=dag
)

# pipeline definition
start_task >> folder_task >> download_task >> load_preprocess_data_task >> split_data_task >> branch_by_drift_task
branch_by_drift_task >> setup_experiment_task >> [train_lr_task, train_xgb_task, train_lgbm_task] >> select_best_model_task >> predict_task
branch_by_drift_task >> predict_task