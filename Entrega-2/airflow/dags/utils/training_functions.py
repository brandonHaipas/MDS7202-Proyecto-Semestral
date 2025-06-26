import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder
import os
from math import ceil

seed = 999
home_dir = os.getenv("AIRFLOW_HOME")

def undersample(df, how='imbalanced'):
    # Separar dataframes por clase
    neg_df = df[df['buy'] == 0]
    pos_df = df[df['buy'] == 1]

    # Obtener tamaño de clase minoritaria
    minority_size = min(len(neg_df), len(pos_df))

    if how == 'balanced':
        # Undersample the majority class
        neg_df_undersampled = neg_df.sample(minority_size, random_state=seed)
        pos_df_undersampled = pos_df.sample(minority_size, random_state=seed)

    else:
        if len(neg_df) < len(pos_df):
            # Undersample the majority class
            neg_df_undersampled = neg_df.sample(minority_size, random_state=seed)
            pos_df_undersampled = pos_df.sample(minority_size * 5, random_state=seed)
        else:
            # Undersample the majority class
            neg_df_undersampled = neg_df.sample(minority_size * 5, random_state=seed)
            pos_df_undersampled = pos_df.sample(minority_size, random_state=seed)

    # Combine and preserve temporal order
    df_undersampled = pd.concat([neg_df_undersampled, pos_df_undersampled]).sort_values(by='week')
    return df_undersampled


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

# this split works well with 10 weeks onward, so the dag should start from the 11th week, that way our preprocessed dataframe has 10 weeks
# fifth assumption: the dag will have an execution start date 
def week_split(df):
    """
    split weeks into 70, 20 and 10 percent ratio roughly
    """
    
    min_week = min(df['week'])
    week_number = len(df['week'].value_counts().index) # the exact number of unique weeks
    upper_train_limit = ceil(min_week + 0.7*week_number -1)
    upper_val_limit = ceil(min_week + 0.9*week_number -1)
    lower_test_limit = upper_val_limit + 1
    
    train_df = undersample(df[df['week'] <= upper_train_limit], 'balanced')
    val_df = undersample(df[(df['week'] > upper_train_limit) & (df['week'] <= upper_val_limit)])
    test_df = undersample(df[df['week'] >= lower_test_limit])

    return train_df, val_df, test_df

def split_data(**kwargs):
    curr_date_str = f"{kwargs.get("ds")}"
    prep_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/preprocessed/preprocessed_prev.parquet")

    train_df, val_df, test_df = week_split(prep_df)

    train_df.to_parquet(f"{home_dir}/{curr_date_str}/splits/train.parquet")
    val_df.to_parquet(f"{home_dir}/{curr_date_str}/splits/val.parquet")
    test_df.to_parquet(f"{home_dir}/{curr_date_str}/splits/test.parquet")
    return

# trains pipeline and saves it into MLFlow
def train_model(**kwargs):
    curr_date_str = f"{kwargs.get('ds')}"
    train_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/train.parquet")
    val_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/val.parquet")
    test_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/splits/test.parquet")

    X_train, y_train = train_df.drop(columns=['buy']), train_df['buy']
    X_val, y_val = val_df.drop(columns=['buy']), val_df['buy']
    X_test, y_test = test_df.drop(columns=['buy']), test_df['buy']

    # define pipeline
    
    # adjust parameters, save and train pipeline in mlflow


    return

# evaluates model and return metrics to assess drift afterwards
def evaluate_model():
    # charge best model from mlflow

    # get metrics with new data

    #returns if drift or stay with model
    return

# evaluates model and saves csv
def evaluate_and_save(**kwargs):
    return