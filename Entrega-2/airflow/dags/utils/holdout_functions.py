import pandas as pd
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