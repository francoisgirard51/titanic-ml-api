# data.py
import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    """
    load CSV in pandas DataFrame.

    :param csv_path: path to load the CSV.
    :return: pandas.DataFrame
    """
    return pd.read_csv(csv_path)
