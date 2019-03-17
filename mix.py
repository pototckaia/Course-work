import pandas as pd

def convert_date(df, name):
    df[name] = pd.to_datetime(df[name])