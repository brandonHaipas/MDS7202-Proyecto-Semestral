import pandas as pd
import sys

transactions_df = pd.read_parquet("data/transacciones.parquet")
customers_df = pd.read_parquet("data/clientes.parquet")
products_df = pd.read_parquet("data/productos.parquet")

customers_categories = ['customer_id', 'region_id', 'zone_id', 'customer_type']
products_categories = ['product_id', 'brand', 'category', 'sub_category' ,'segment', 'package']

customers_df[customers_categories] = customers_df[customers_categories].astype('category')
products_df[products_categories] = products_df[products_categories].astype('category')

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
    sodai_df = bought_df.merge(products_df, on='product_id').merge(customers_df, on='customer_id')
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
    return sodai_week_df

# arguments passed
args = sys.argv

week_to_use = 52

if len(args) > 1:
    week_to_use = sys.argv[1]

# add intermediate column with iso calendar week
if int(week_to_use) != 52 and (not week_to_use.isnumeric()):
    raise Exception("Not a number")

if int(week_to_use) <= 11:
    raise Exception("number of weeks too low")


transactions_df['week'] = transactions_df['purchase_date'].apply(lambda x: x.isocalendar().week)

# now separate transactions_df in week(s) and historic

hist_transac_df = transactions_df[transactions_df['week']< week_to_use].drop(columns='week').copy()

if week_to_use == 52:

    hist_transac_df = transactions_df[(transactions_df['week'] > 30) & (transactions_df['week'] < week_to_use)].drop(columns='week').copy()
    week_df = transactions_df[transactions_df['week']==week_to_use].drop(columns='week').copy()

    clean_hist_transac_df = drop_suspicious_purchases(hist_transac_df)
    joined_hist_df = join_data(clean_hist_transac_df)
    grouped_hist_df = group_by_week(joined_hist_df)

    grouped_hist_df.to_parquet("historic.parquet")
    week_df.to_parquet("week.parquet")


else:
    quit()

