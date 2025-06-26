import os

# Variable home
home_dir = os.getenv('AIRFLOW_HOME')

def create_folders(**kwargs):
    date = kwargs.get("ds")
    os.mkdir(date)
    os.mkdir(f"{home_dir}/{date}/raw")
    os.mkdir(f"{home_dir}/{date}/splits")
    os.mkdir(f"{home_dir}/{date}/models")
    os.mkdir(f"{home_dir}/{date}/preprocessed")
    return
