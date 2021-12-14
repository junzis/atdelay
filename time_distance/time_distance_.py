from numpy import radians, sin, arcsin, cos, sqrt
import pandas as pd
import sys


def haversine(P: pd.DataFrame):
    """Calculates the great-circle distance between two decimal
    coordinates using the Haversine formula and applies it to a dataframe.
    The formula was found on https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        P (pd.DataFrame): Dataframe of LRData

    Returns:
        pd.DataFrame: Dataframe of LRData with extra column 'distance'
    """    
    coords_a, coords_b = (P['ADEPLong'], P['ADEPLat']), (P['ADESLong'], P['ADESLat'])
    # Conversion to radians is necessary for the trigonometric functions
    phi_1, phi_2 = radians(coords_a[1]), radians(coords_b[1])
    lambda_1, lambda_2 = radians(coords_a[0]), radians(coords_b[0])

    return 2 * 6371 * arcsin(sqrt(sin((phi_2 - phi_1)/2)**2 + cos(phi_1) * cos(phi_2) * sin((lambda_2 - lambda_1)/2)**2))


def time_distance(P: pd.DataFrame):
    P['distance'] = P.apply(lambda row: haversine(row), axis=1)
    P['flight_time'] = P.apply(lambda row: row['FiledAT'] - row['FiledOBT'], axis=1)
    P = P.drop(['ADEPLong', 'ADEPLat', 'ADESLong', 'ADESLat'], axis=1)

    return P
