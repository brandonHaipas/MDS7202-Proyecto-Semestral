import os

# Variable home
home_dir = os.getenv('AIRFLOW_HOME')

def create_folders(**kwargs):
    date = kwargs.get("ds")
    if not os.path.isdir(date):
        os.mkdir(date)
    if not os.path.isdir(f"{home_dir}/{date}/raw"):
        os.mkdir(f"{home_dir}/{date}/raw")
    if not os.path.isdir(f"{home_dir}/{date}/splits"):
        os.mkdir(f"{home_dir}/{date}/splits")
    if not os.path.isdir(f"{home_dir}/{date}/models"):
        os.mkdir(f"{home_dir}/{date}/models")
    if not os.path.isdir(f"{home_dir}/{date}/preprocessed"):
        os.mkdir(f"{home_dir}/{date}/preprocessed")
    return
