# Lectura de datos
from datetime import datetime, timedelta
import pandas as pd
import os
import itertools

home_dir = os.getenv("AIRFLOW_HOME")

# first assumption: clients have certain behaviour in the long run
clientes_df = pd.read_parquet(f'{home_dir}/data/clientes.parquet')
# second assumption: we have product metadata
productos_df = pd.read_parquet(f'{home_dir}/data/productos.parquet')

# since product and client metadata is assumed to be already in the home directory, we can format it however we want
clientes_categories = ['customer_id', 'region_id', 'zone_id', 'customer_type']
productos_categories = ['product_id', 'brand', 'category', 'sub_category' ,'segment', 'package']

clientes_df[clientes_categories] = clientes_df[clientes_categories].astype('category')
productos_df[productos_categories] = productos_df[productos_categories].astype('category')

# random state definition
RANDOM_STATE = 99



def drop_suspicious_purchases(transactions):
    """
    Recieves a transactions dataframe and drops those transactions corresponding to an order id where the resulting number of items is negative
    and drops those transactions where a client returns the same amount of items more than once in a single order
    """
    devoluciones_df = transactions[transactions['items'] < 0]
    suspicious_orders = devoluciones_df.loc[~devoluciones_df.index.isin(devoluciones_df[['customer_id', 'product_id', 'order_id', 'items']].drop_duplicates().index)]['order_id']
    transactions = transactions[~transactions['order_id'].isin(suspicious_orders)]

    # grouped by order id and sum items to get if it's positive or negative

    transactions_by_order = transactions.groupby('order_id')['items'].sum()
    bad_order_ids = transactions_by_order[transactions_by_order < 0].index.to_list()
    transactions = transactions[~transactions['order_id'].isin(bad_order_ids)]

    return transactions

def join_data(transactions):
    """
    joins the transactional data with the product and client metadata after transaction cleaning
    """
    bought_df = transactions.groupby(by=[c for c in transactions.columns if c != 'items'], observed=True).sum().reset_index()
    sodai_df = bought_df.merge(productos_df, on='product_id').merge(clientes_df, on='customer_id')
    return sodai_df



def group_by_week(historic_df, sodai_df):

    sodai_week_df = sodai_df.copy()
    sodai_week_df = sodai_week_df[sodai_week_df['items'] > 0]
    sodai_week_df['week'] = historic_df['week'].max() + 1
    sodai_week_df = sodai_week_df.sort_values('purchase_date')
    df_items = sodai_week_df.groupby(['customer_id', 'product_id', 'week'], as_index=False)['items'].sum()
    sodai_week_df = sodai_week_df.drop(columns=['purchase_date', 'order_id', 'items']).drop_duplicates()
    sodai_week_df = sodai_week_df.merge(df_items, on=['customer_id', 'product_id', 'week'])
    sodai_week_df['buy'] = 1
    return sodai_week_df

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

def add_negative_class_rows(df, how='inbalanced'):
    customer_ids = df['customer_id'].unique()
    product_ids = df['product_id'].unique()
    weeks = list(range(int(df['week'].min()), int(df['week'].max() + 1)))

    # Filas candidatas a ser negativos
    candidates = set(itertools.product(customer_ids, product_ids, weeks))

    # Filas originales del dataset
    originals = set(df[['customer_id', 'product_id', 'week']].itertuples(index=False, name=None))

    # Obtención de negativos
    negative_df = pd.DataFrame(list(candidates - originals), columns=['customer_id', 'product_id', 'week'])

    if how == 'balanced':
        negative_df = negative_df.sample(len(df), random_state=RANDOM_STATE)

    # Añadir filas a dataset
    sodai_negative_df = negative_df.merge(productos_df, on='product_id').merge(clientes_df, on='customer_id')
    sodai_negative_df['buy'] = 0
    sodai_negative_df['items'] = 0

    return pd.concat([df, sodai_negative_df])

def create_predict_dataset(historic_df, week_df):
    """
    arguments
    historic_df without negatives in ready to preprocess format
    week_df in transacciones.parquet format with information of the current week, magically appears in work folder
    """
    
    # Obtain new data grouped by week
    cleaned_week_df = drop_suspicious_purchases(week_df)
    joined_week_df = join_data(cleaned_week_df) # joined with client and product data
    grouped_week_df = group_by_week(historic_df, joined_week_df) # grouped by week number

    # Concat week to historic data to create new historic
    new_historic_df = pd.concat([historic_df, grouped_week_df])
    
    # Create data for prediction using all clients and products in new historic
    unique_clients = new_historic_df['customer_id'].unique()
    unique_products = new_historic_df['product_id'].unique()
    

    prediction_df = (
        pd.MultiIndex
        .from_product([unique_clients, unique_products],
                        names=['customer_id', 'product_id'])
        .to_frame(index=False)
    )

    prediction_df = prediction_df.merge(productos_df, on='product_id').merge(clientes_df, on='customer_id')
    prediction_df['buy'] = -1
    prediction_df['items'] = 0
    prediction_df['week'] = grouped_week_df.iloc[0]['week'] + 1

    # Return grouped week and prediction data   
    return grouped_week_df, prediction_df

def load_and_preprocess_data(**kwargs):
    """
    load dataframes and saves them in 
    """
    curr_date_str = kwargs.get('ds')

    # Leer parquet de esta semana y parquet historico
    week_df = pd.read_parquet(f"{home_dir}/data/week.parquet")
    historic_df = pd.read_parquet(f"{home_dir}/data/historic.parquet")
    
    # Crear nuevo dataframes de semana agrupada y a predecir
    grouped_week_df, predict_df = create_predict_dataset(historic_df, week_df)

    # Crear variables custom en dataframe historico, de semana agrupada y a predecir
    full_hist_df = pd.concat([historic_df, week_df])
    full_hist_neg_df = add_negative_class_rows(full_hist_df)

    full_df = pd.concat([full_hist_neg_df, predict_df])
    full_df = create_custom_vars(full_df)

    historic_df = full_df[full_df['week'] <= historic_df['week'].max()]
    grouped_week_df = full_df[full_df['week'] == grouped_week_df['week'].max()]
    predict_df = full_df[full_df['week'] == predict_df['week'].max()]

    # Guardar dataframes en parquet 
    historic_df.to_parquet(f"{home_dir}/{curr_date_str}/preprocessed/historic.parquet", index=False)
    grouped_week_df.to_parquet(f"{home_dir}/{curr_date_str}/preprocessed/grouped_week.parquet", index=False)
    predict_df.to_parquet(f"{home_dir}/{curr_date_str}/preprocessed/predict.parquet", index=False)
    return