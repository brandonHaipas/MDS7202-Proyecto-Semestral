# Lectura de datos
from datetime import datetime, timedelta
import pandas as pd
import os
import itertools

home_dir = os.getenv("AIRFOW_HOME")

# first assumption: clients have certain behaviour in the long run
clientes_df = pd.read_parquet(f'{home_dir}/data/clientes.parquet')
# second assumption: we have product metadata
productos_df = pd.read_parquet(f'{home_dir}/data/productos.parquet')
# third assumption: we magically have transaction data in our home directory
transacciones_df = pd.read_parquet(f'{home_dir}/data/transacciones.parquet')

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

def group_by_week(sodai_df):

    sodai_week_df = sodai_df.copy()
    sodai_week_df = sodai_week_df[sodai_week_df['items'] > 0]
    sodai_week_df['week'] = sodai_week_df['purchase_date'].apply(lambda x: x.isocalendar().week)
    sodai_week_df = sodai_week_df.sort_values('purchase_date')
    df_items = sodai_week_df.groupby(['customer_id', 'product_id', 'week'], as_index=False)['items'].sum()
    sodai_week_df = sodai_week_df.drop(columns=['purchase_date', 'order_id', 'items']).drop_duplicates()
    sodai_week_df = sodai_week_df.merge(df_items, on=['customer_id', 'product_id', 'week'])
    sodai_week_df['buy'] = 1
    sodai_week_df = add_negative_class_rows(sodai_week_df)
    return sodai_week_df

def add_negative_class_rows(df, how='inbalanced'):
    customer_ids = transacciones_df['customer_id'].unique()
    product_ids = transacciones_df['product_id'].unique()
    weeks = list(range(df['week'].min(), df['week'].max() + 1))

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


def load_and_preprocess_data(**kwargs):
    """
    loads de data based on the date of execution (ds)
    takes all the rows from transacciones.parquet where the dates are below ds, 
    then creates a dataframe and saves it to the /date/raw folder
    and finally joins the datasets saved in /airflow/date/raw with client and product metadata, then saves the final datasets in preprocess 
    """
    curr_date_str = f"{kwargs.get('ds')}"
    date_format = "%Y-%m-%d"
    exec_date = datetime.strptime(curr_date_str, date_format)
    last_date = exec_date + timedelta(days=7)

    # this lines might change because it's not a fact that everyweek there'll be a purchase, but as we have transacciones.parquet we know it's the case
    # fourth assumption: every week there's at least one purchase
    this_week_df = transacciones_df[(transacciones_df['purchase_date'].to_pydatetime() >= exec_date) & (transacciones_df['purchase_date'].to_pydatetime()< last_date)]
    prev_transaccions_df = transacciones_df[transacciones_df['purchase_date'].to_pydatetime() < exec_date]

    prev_transaccions_df.to_parquet(f"{home_dir}/{curr_date_str}/raw/raw_prev_transacc.parquet")
    this_week_df.to_parquet(f"{home_dir}/{curr_date_str}/raw/raw_week_transacc.parquet")

    # first we obtain the dataframes
    prev_transac_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/raw/raw_prev_transacc.parquet")
    this_week_df = pd.read_parquet(f"{home_dir}/{curr_date_str}/raw/raw_week_transacc.parquet")

    # before joining data, there's some type handling to do
    transacciones_categories = ['customer_id', 'product_id', 'order_id']
    prev_transac_df[transacciones_categories] = prev_transac_df[transacciones_categories].astype('category')
    this_week_df = this_week_df[transacciones_categories].astype('category')

    # drop bad data
    prev_transac_df = drop_suspicious_purchases(prev_transac_df)
    this_week_df = drop_suspicious_purchases(this_week_df)

    #join clients, products and transactions parquet datasets for both previous (t) and next week (t+1) instances
    joined_prev = join_data(prev_transac_df)
    joined_week = join_data(this_week_df)

    # group by week and add negative instances
    by_week_prev = group_by_week(joined_prev)
    by_week_this_week = group_by_week(joined_week)

    # save the preprocessed dataframes in the folders
    by_week_prev.to_parquet(f"{home_dir}/{curr_date_str}/preprocessed/preprocessed_prev.parquet")
    by_week_this_week.to_parquet(f"{home_dir}/{curr_date_str}/preprocessed/preprocessed_current.parquet")
    return