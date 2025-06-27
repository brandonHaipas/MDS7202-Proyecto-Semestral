from datetime import datetime, date

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from utils.init_functions import create_folders
from utils.preprocess_functions import load_and_preprocess_data


dag =  DAG(
    dag_id = "hiring_dynamic",
    description="A dynamic dag",
    start_date = datetime(2024,10,1),
    catchup=True,
    schedule_interval='0 15 5 * *'
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

# Task 6 - branch by drift (evaluates then branches)

# Task 7.a - optimize and train model

# Task 7.b - good to go flag

# Task 8 - prediction task (saves to csv)
def branch_by_drift(**kwargs):
    execution_ = kwargs['logical_date'].date()
    threshold_drift = date(2024, 11, 1)

    if execution_date < threshold_date:
        return 'Download_dataset_1'
    else:
        return 'Download_both_datasets'

# Task 3 - Branch operator
date_branching_task = BranchPythonOperator(
    task_id= "Date_branching",
    python_callable=branch_by_date,
    dag=dag
)

# Task 3.a - Download data_1.csv
download_dataset_1_task = BashOperator(
    task_id='Download_dataset_1',
    bash_command='curl -o $AIRFLOW_HOME/{{ ds }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv',
    dag=dag
)

# Task 3.b - Download both datasets
download_dataset_1_and_2_task = BashOperator(
    task_id = 'Download_both_datasets',
    bash_command='curl -o $AIRFLOW_HOME/{{ ds }}/raw/data_1.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv -o $AIRFLOW_HOME/{{ ds }}/raw/data_2.csv https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv',
    dag=dag
)

# Task 4 - Load and merge
load_and_merge_task = PythonOperator(
    task_id = "Load_and_merge",
    python_callable=load_and_merge,
    trigger_rule ='one_success',
    dag=dag
)

# Task 5 - Holdout
holdout_task = PythonOperator(
    task_id="Holdout",
    python_callable= split_data,
    dag=dag
)

# Task 6.a - Train Random Forest
train_rf_task = PythonOperator(
    task_id="Training_rf",
    python_callable = train_model,
    op_kwargs = {"model_name": "rf", "model": RandomForestClassifier(random_state = seed)},
    dag = dag
)

# Task 6.a - Train XGBoost
train_xgb_task = PythonOperator(
    task_id='Training_xgb',
    python_callable = train_model,
    op_kwargs = {"model_name": "xgb", "model": XGBClassifier(random_state =seed)},
    dag= dag
)

# Task 6.a - Train Extra Tree
train_et_task = PythonOperator(
    task_id='Training_et',
    python_callable = train_model,
    op_kwargs = {"model_name": "et", "model": ExtraTreesClassifier(random_state = seed)},
    dag = dag
)

# Task 7 - Evaluate
evaluate_task = PythonOperator(
    task_id='Evaluate_models',
    python_callable = evaluate_models,
    dag=dag,
    trigger_rule='all_success'
)

# pipeline definition
start_task >> folder_task >> date_branching_task
date_branching_task >> [download_dataset_1_task, download_dataset_1_and_2_task]
download_dataset_1_task >> load_and_merge_task
download_dataset_1_and_2_task >> load_and_merge_task
load_and_merge_task >> holdout_task >> [train_rf_task, train_xgb_task, train_et_task] >> evaluate_task