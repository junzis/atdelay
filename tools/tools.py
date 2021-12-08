import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def missing_values(filename: string):
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    return df


def data_encoding(df: pd.DataFrame):
    enc = OneHotEncoder()
    df.to_numpy()
    enc.fit_transform(df)
    return df


def data_scaling(array: np.array):        
    try:
        scaler = StandardScaler()
        scaler.fit_transform(array)
        return array
    except ValueError:
        print("Encode the dataframe first")