# from .extract import *
# from .extractionvalues import *
# from .airportvalues import *
from . import extract
from datetime import datetime
import numpy as np


import pandas as pd


def getAdjacencyMatrix(airports, timeslotLength=60):
    P = extract.extractData(start=datetime(2018, 3, 1), end=datetime(2018, 3, 31))
    P = extract.filterAirports(P, airports)
    P = extract.calculateDelays(P)
    P = P.drop(
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
    P.sort_values("FiledOBT")
    arr = (
        P.groupby([pd.Grouper(key="FiledAT", freq=f"{timeslotLength}min"), "ADES"])[
            "ADEP"
        ]
        .value_counts()
        .unstack(fill_value=0)
    )

    mux = pd.MultiIndex.from_product(
        [
            arr.index.get_level_values("FiledAT").unique(),
            arr.index.get_level_values("ADES").unique(),
        ]
    )
    df = arr.reindex(mux)

    a = (
        df.sort_index(level=0, ascending=True)
        .reindex(sorted(df.columns), axis=1)
        .fillna(0)
        .to_numpy()
    )
    a = a
    maximum = np.zeros((10, 10))

    b = [a[i : i + 10] for i in range(0, len(a), 10)]

    ### fix later division by 0 results in nans
    # b = np.array(b)
    # for i in range(len(a[0])):
    #     for j in range(len(a[0])):
    #         maximum[i][j] = max(b[:, i, j])
    # adj_matrices_demand = np.array([x / maximum for x in b])
    print(np.array(b).shape)
    return b
    return adj_matrices_demand


