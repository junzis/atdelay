from preprocess.extract import generalFilterAirport, filterAirports
from preprocess.airportvalues import airport_dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from regressionModels.tool_box import haversine


def distance_weight_adjacency(airports, threshold=1000):
    """Generates a weight matrix where each entry is filled with a weight representation
    of the distance between two airports

    Args:
        airports (list): List of ICAO airport strings
        threshold (int, optional): Threshold for distance between airports for which the cell will take a value.
        Defaults to 1000.

    Returns:
        np.ndarray: Square numpy array
    """
    D = np.zeros((len(airports), len(airports)))
    threshold = 1000
    for i, airport1 in enumerate(airports):
        for j, airport2 in enumerate(airports):
            coords1 = (
                airport_dict[airport1]["longitude"],
                airport_dict[airport1]["latitude"],
            )
            coords2 = (
                airport_dict[airport2]["longitude"],
                airport_dict[airport2]["latitude"],
            )
            D[i, j] = haversine(coords1, coords2)

    st_dev = np.std(D)
    D = np.where(D < threshold, np.exp(-(D ** 2) / st_dev ** 2), 0)

    diag_filter = np.ones((len(airports), len(airports)))
    np.fill_diagonal(diag_filter, 0)
    return D * diag_filter
