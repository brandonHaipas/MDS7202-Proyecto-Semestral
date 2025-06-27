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
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


seed = 999
home_dir = os.getenv("AIRFLOW_HOME")
MODEL_NAME   = "best_model" 
MODEL_ALIAS  = "current"
client = MlflowClient()


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

def create_custom_vars(df):
    # Se ordenan las filas temporalmente
    df = df.sort_values(by=['customer_id', 'product_id', 'week'])

    # Se agrupa para obtener el promedio de items con un shift de 1 (hasta la semana anterior)
    df['mean_past_items'] = (
        df.groupby(['customer_id', 'product_id'])['items']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Sumar items por semana por cliente
    df['weekly_items'] = df.groupby(['customer_id', 'week'])['items'].transform('sum')

    # Calcular promedio histórico de compras por semana y cliente, excluyendo la semana actual
    df['mean_purchases_per_week'] = (
        df.groupby('customer_id')['weekly_items']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Calcular promedio histórico de ventas por semana y producto, excluyendo la semana actual
    df['mean_sales_per_week'] = (
        df.groupby('product_id')['weekly_items']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Calcular semanas desde la última compra del producto
    df['week_last_purchase'] = df['week'] * (df['items'] > 0)
    df['last_week_seen'] = (
        df.groupby(['customer_id', 'product_id'])['week_last_purchase']
        .transform(lambda x: x.shift(1).cummax())
    )
    df['weeks_since_last_purchase'] = df['week'] - df['last_week_seen']

    # Se completan los valores NaN con -1 (un valor que no puede obtener a través del cálculo)
    df['mean_past_items'] = df['mean_past_items'].fillna(-1)
    df['mean_purchases_per_week'] = df['mean_purchases_per_week'].fillna(-1)
    df['mean_sales_per_week'] = df['mean_sales_per_week'].fillna(-1)
    df['weeks_since_last_purchase'] = df['weeks_since_last_purchase'].fillna(-1)

    return df

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

            # Definición de pipeline con prunning
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, observation_key="validation_1-pre")
            model = XGBClassifier(seed=seed, **params, eval_metric='pre' ,callbacks = [pruning_callback])
        
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

        custom_features = FunctionTransformer(create_custom_vars)
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
                ("Custom Features", custom_features),
                ("Preprocessing", preprocessing),
                ("Classification", model)
            ]
        )

        # Evaluación
        pipeline[:-1].fit(X_train, y_train)
        X_train_transformed = pipeline[:-1].transform(X_train)
        X_val_transformed = pipeline[:-1].transform(X_val)

        with mlflow.start_run(run_name=f"{model_string} with {trial.number}", experiment_id=experiment):
            pipeline.fit(
                X_train,
                y_train,
                Classification__eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
                Classification__verbose=False
                )
            pred = pipeline.predict(X_val)
            f1 = f1_score(y_val, pred)
            mlflow.log_metric("valid_f1", f1)
            mlflow.log_model("model", pipeline)
        
        return f1

    return objective

def setup_experiment(**kwargs):
    # Create a common experiment for all trainings
    pass

# trains pipeline and saves it into MLFlow
def train_model(model_string,**kwargs):
    curr_date_str = f"{kwargs.get('ds')}"
    train_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/train.parquet")
    val_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/val.parquet")

    X_train, y_train = train_df.drop(columns=['buy']), train_df['buy']
    X_val, y_val = val_df.drop(columns=['buy']), val_df['buy']

    # Read experiment from XCom
    experiment = mlflow.create_experiment(f"train_{curr_date_str}")

    objective_fun = create_objective(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, model_string=model_string, experiment=experiment)

    # Optimización
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective_fun, timeout=300)

    # Obtención del mejor modelo y guardado con mlflow
    runs = mlflow.search_runs(experiment)
    best_run_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]

    # Register & promote:
    register_best_model(best_run_id)
    return

def select_best_model(**kwargs):
    # Search in experiment for best model across all trainings
    pass

def branch_by_drift(**kwargs):
    # Get test data
    curr_date_str = f"{kwargs.get('ds')}"
    test_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/test.parquet")
    X_test, y_test = test_df.drop(columns=['buy']), test_df['buy']

    # Get current best model
    try:
        model, best_f1 = load_current_model_with_metric()
    except MlflowException:
        # first day – force training branch
        return 'train'

    # Evaluate
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    if f1 < best_f1:
        return 'train'

    return 'no_train'

# evaluates model and saves csv
def predict_and_save(**kwargs):
    curr_date_str = f'{kwargs.get('sd')}'
    test_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/test.parquet")
    X_test, y_test = test_df.drop(columns=['buy']), test_df['buy']

    # Get current best model with mlflow
    model, best_f1 = load_current_model_with_metric()

    # Get metrics with new data
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    with mlflow.start_run(run_name="evaluation", tags={
            "model_version": mv.version, "alias": MODEL_ALIAS,
            "type": "drift-check"}):
        mlflow.log_metrics({"f1": f1, "baseline_f1": best_f1})
    return