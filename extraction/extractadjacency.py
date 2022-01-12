from .extract import *
from .extractionvalues import *

# from .airportvalues import *
from extraction.extractionvalues import ICAOTOP50
from . import extract
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from regressionModels.tool_box import haversine


def getAdjacencyMatrix(
    airports,
    start=datetime(2018, 3, 1),
    end=datetime(2019, 12, 31),
    timeslotLength=60,
    debug=False,
):

    # Create a list with all times for multiindex later:
    def daterange(start_date, end_date):
        delta = timedelta(minutes=timeslotLength)
        while start_date < end_date:
            yield start_date
            start_date += delta

    dateList = daterange(start, end)

    P = pd.DataFrame()  # start an empty df
    for airport in airports:
        # generate filtered data
        Ptemp = extract.generalFilterAirport(start, end, airport)
        Ptemp = extract.filterAirports(Ptemp, airports)
        Ptemp = Ptemp.drop(
            [
                "ACType",
                "ACOperator",
                "FlightType",
                "ActualDistanceFlown",
                "ECTRLID",
                "ADEPLat",
                "ADEPLong",
                "ADESLat",
                "ADESLong",
                "ActualOBT",
                "ActualAT",
                "ArrivalDelay",
                "DepartureDelay",
            ],
            axis=1,
        )

        P = pd.concat([P, Ptemp])

    # initial step to get the flights between airports
    P = (
        P.groupby([pd.Grouper(key="FiledAT", freq=f"{timeslotLength}min"), "ADES"])[
            "ADEP"
        ]
        .value_counts()
        .unstack(fill_value=0)
    )

    # generate multindex format we want: an adjacency matrix
    adjacencyFormat = pd.MultiIndex.from_product([dateList, airports])

    # apply the multindex format and sort the columns by airports list
    P = P.reindex(adjacencyFormat, fill_value=0)[airports]
    # Generate numpy adjacency matrix in 3d format
    A = P.to_numpy().reshape(-1, 10, 10)

    maximum = np.amax(A, axis=0)
    new_max = np.where(maximum > 0, maximum, 1 )
    final_matrix = np.divide(A, new_max)

    if debug:
        dp = 7
        print(airports)
        print(P.iloc[int(dp*10):int(dp*10)+10])
        print(P.index.get_level_values(0).unique()[dp])
        print(final_matrix[dp])
        print(final_matrix.shape)


    return final_matrix


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
