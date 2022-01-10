# from .extract import *
# from .extractionvalues import *
# from .airportvalues import *
from extraction.extractionvalues import ICAOTOP50
from . import extract
from datetime import datetime
import numpy as np
import pandas as pd


def getAdjacencyMatrix(
    airports, start=datetime(2018, 3, 1), end=datetime(2018, 3, 31), timeslotLength=60, debug=False
):
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
    adjacencyFormat = pd.MultiIndex.from_product(
        [P.index.get_level_values("FiledAT").unique(), airports]
    )

    # apply the multindex format and sort the columns by airports list
    P = P.reindex(adjacencyFormat, fill_value=0)[airports]

    # Generate numpy adjacency matrix in 3d format
    A = P.to_numpy().reshape(-1, 10, 10)

    if debug:
        print(airports)
        print(P.head(10))
        print(A[0])
        print(A.shape)

    return A



def getAdjacencyMatrixOld(airports, timeslotLength=60):
    raise (DeprecationWarning("this version of get adjacency matrix is outdated"))
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


