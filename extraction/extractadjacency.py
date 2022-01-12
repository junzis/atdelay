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
    airports: list,
    start: datetime = datetime(2018, 3, 1),
    end: datetime = datetime(2019, 12, 31),
    timeslotLength: int = 60,
    debug: bool = False,
) -> np.ndarray:
    """Generate adjacency matrix for the spektral dataset based on flights between airports

    Args:
        airports (list): list of airports by ICAO code
        start (datetime, optional): start date to consider (inclusive). Defaults to datetime(2018, 3, 1).
        end (datetime, optional): start date to consider (inclusive). Defaults to datetime(2019, 12, 31).
        timeslotLength (int, optional): length of timeslot to aggregate by in minutes. Defaults to 60.
        debug (bool, optional): print out some lines to troubleshoot the function. Defaults to False.

    Returns:
        np.ndarry: Nairports x Nairports x amount array of adjacency matrices

    """
    # Create a list with all times for multiindex later:
    def daterange(start: datetime, end: datetime):
        """Generator that yields a list of timeslots to conform the index by

        Args:
            start (datetime): start date
            end (datetime): end date

        Yields:
            list: list of timeslots for index
        """
        delta = timedelta(minutes=timeslotLength)
        while start < end:
            yield start
            start += delta

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
    A = P.to_numpy().reshape(-1, len(airports), len(airports))

    # Normalise the matrix
    maximum = np.amax(A, axis=0)
    final_matrix = np.divide(A, maximum)
    np.nan_to_num(final_matrix, copy=False)

    if debug:
        dp = 7
        print(airports)
        print(P.iloc[int(dp * len(airports)) : int(dp * len(airports)) + len(airports)])
        print(P.index.get_level_values(0).unique()[dp])
        print(final_matrix[dp])
        print(final_matrix.shape)

    return final_matrix


def distance_weight_adjacency(airports, threshold=1000):
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
