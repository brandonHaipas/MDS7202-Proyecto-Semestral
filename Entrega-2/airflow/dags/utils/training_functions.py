import os
import mlflow.metrics
import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import mlflow
import pickle
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


seed = 999
home_dir = os.getenv("AIRFLOW_HOME")

# Mlflow tracking config
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME   = "best_model" 
MODEL_ALIAS  = "current"
client = MlflowClient()
mlflow.set_tracking_uri(mlflow_tracking_uri)


def register_best_model(run_id):
    # Registro (crea el modelo si no existe previamente)
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    # Promoción con alias
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, mv.version)

def get_metric_from_run(run_id, metric_name="valid_f1"):
    return client.get_run(run_id).data.metrics[metric_name]

def load_current_model_with_metric(metric_name="valid_f1"):
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"   # alias syntax
    model = mlflow.sklearn.load_model(model_uri)

    # metric lives on the original run
    reference_metric = get_metric_from_run(mv.run_id, metric_name)
    return model, reference_metric

def create_objective(X_train, y_train, X_val, y_val, model_string, experiment):
    def objective(trial):
        # Hiperparametros a optimizar
        params = {}
        model = None

        if model_string == "xgb":
            params  = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "max_leaves": trial.suggest_int("max_leaves", 0, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            }

            model = XGBClassifier(seed=seed, **params, eval_metric='pre')
        
        elif model_string == "lgbm":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 0, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            }
            model = LGBMClassifier(random_state=seed, **params)
        
        # Logistic regression
        else:
            params = {

            }

            model = LogisticRegression(random_state=seed, **params)

        preprocessing = ColumnTransformer(
            [
                (
                    "Scale",
                    MinMaxScaler(),
                    [
                        "X", "Y", "size", "num_deliver_per_week", "num_visit_per_week", "mean_past_items", "mean_purchases_per_week", "mean_sales_per_week", "weeks_since_last_purchase"
                    ],
                ),
                (
                    "One Hot Encoding",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    [
                        "brand", "sub_category", "segment", "package", "customer_type",
                    ],
                ),
            ],
        )
        preprocessing.set_output(transform='pandas')

        pipeline = Pipeline(
            steps=[
                ("Preprocessing", preprocessing),
                ("Classification", model)
            ]
        )

        # Evaluación
        with mlflow.start_run(run_name=f"{model_string} with {trial.number}", experiment_id=experiment):
            pipeline.fit(
                X_train,
                y_train,
                )
            pred = pipeline.predict(X_val)
            f1 = f1_score(y_val, pred)
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
        
        return f1

    return objective

def setup_experiment(**kwargs):
    curr_date_str = f"{kwargs.get('ts')}"
    ti = kwargs['ti']

    # Create a common experiment for all trainings
    experiment = mlflow.create_experiment(f"train_{curr_date_str}")
    ti.xcom_push(key="experiment_id", value=experiment)
    pass

# trains pipeline and saves it into MLFlow
def train_model(model_string,**kwargs):
    curr_date_str = f"{kwargs.get('ds')}"
    ti = kwargs['ti']
    train_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/train.parquet")
    val_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/val.parquet")

    X_train, y_train = train_df.drop(columns=['buy']), train_df['buy']
    X_val, y_val = val_df.drop(columns=['buy']), val_df['buy']

    # Read experiment from XCom
    experiment = ti.xcom_pull(key='experiment_id', task_ids='Create_experiment_task')

    objective_fun = create_objective(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_string=model_string, experiment=experiment)

    # Optimización
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective_fun, timeout=30)
    return

def select_best_model(**kwargs):
    # Search in experiment for best model across all trainings
    ti = kwargs['ti']

    experiment = ti.xcom_pull(key='experiment_id', task_ids='Create_experiment_task')
    runs = mlflow.search_runs(experiment)
    best_run_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]

    # Register & promote:
    register_best_model(best_run_id)
    return

def branch_by_drift(**kwargs):
    # Get test data
    curr_date_str = f"{kwargs.get('ds')}"
    val_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/val.parquet")
    X_val, y_val = val_df.drop(columns=['buy']), val_df['buy']

    # Get current best model
    try:
        model, best_f1 = load_current_model_with_metric()
    except MlflowException:
        # first day – force training branch
        return 'Create_experiment_task'

    # Evaluate
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)

    if f1 < best_f1:
        return 'Create_experiment_task'

    return 'Prediction_task'

# evaluates model and saves csv
def predict_and_save(**kwargs):
    curr_date_str = f'{kwargs.get("ds")}'
    
    # charge values for next week
    predict_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/preprocessed/predict.parquet")

    # Get current best model with mlflow
    model, best_f1 = load_current_model_with_metric()

    # Get metrics with new data
    y_pred = model.predict(predict_df)

    predict_df['buy'] = y_pred
    buy_df = predict_df[predict_df['buy'] == 1]
    buy_df[['customer_id', 'product_id']].to_csv(f"/predictions/{curr_date_str}.csv", index=False)