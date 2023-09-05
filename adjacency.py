from tools.extract import generalFilterAirport, filterAirports
from tools.constants import airport_dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm


def haversine(*P: pd.DataFrame):
    """Calculates the great-circle distance between two decimal

    Args:
        P (pd.DataFrame): Dataframe of LRData or two coordinate tuples

    Returns:
        pd.DataFrame: Dataframe of LRData with extra column 'distance'
    """
    if isinstance(P[0], pd.DataFrame):
        P = P[0]
        coords_a, coords_b = (P["ADEPLong"], P["ADEPLat"]), (
            P["ADESLong"],
            P["ADESLat"],
        )
    else:
        (coords_a, coords_b) = P
    # Conversion to radians is necessary for the trigonometric functions
    phi_1, phi_2 = np.radians(coords_a[1]), np.radians(coords_b[1])
    lambda_1, lambda_2 = np.radians(coords_a[0]), np.radians(coords_b[0])

    return (
        2
        * 6371
        * np.arcsin(
            np.sqrt(
                np.sin((phi_2 - phi_1) / 2) ** 2
                + np.cos(phi_1) * np.cos(phi_2) * np.sin((lambda_2 - lambda_1) / 2) ** 2
            )
        )
    )


def distance_adjacency(airports, threshold=1000):
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


# Create a list with all times for multiindex later:
def daterange(start: datetime, end: datetime, timeinterval, date_filter=None):
    """Create list of timeslots."""

    delta = timedelta(minutes=timeinterval)
    time = start
    date_filter = date_filter.astype(str)
    buffer = []
    while time < end:

        ts = time.strftime("%Y-%m-%d")

        if (date_filter is not None) and (ts not in date_filter):
            time += delta
            continue

        if time.month not in [3, 6, 9, 12]:
            time += delta
            continue

        buffer.append(time)
        time += delta

    return buffer


def throughput_adjacency(df, airports, start, end, timeinterval, date_filter):

    datetime_list = daterange(start, end, timeinterval, date_filter)

    # initial step to get the flights between airports
    grouper = pd.Grouper(key="FiledAT", freq=f"{timeinterval}min")
    df = df.groupby([grouper, "ADES"])["ADEP"].value_counts().unstack(fill_value=0)

    # generate multindex format we want: an adjacency matrix
    adjacencyFormat = pd.MultiIndex.from_product([datetime_list, airports])

    # apply the multindex format and sort the columns by airports list
    df = df.reindex(adjacencyFormat, fill_value=0)[airports]

    # Generate numpy adjacency matrix in 3d format
    mat3d = df.to_numpy().reshape(-1, len(airports), len(airports))

    # Normalise the matrix
    maximum = np.amax(mat3d, axis=0)
    new_maximum = np.where(maximum == 0, 1, maximum)
    final_matrix = np.divide(mat3d, new_maximum)

    return final_matrix


# #%%
# (
#     df_all.groupby(["timeslot", "ap0", "ap1"])
#     .agg(
#         {
#             "flight_id": "count",
#             "duration_planned": "mean",
#             "delay_departure": "mean",
#             "delay_arrival": "mean",

#         }
#     )
#     # .reindex(timeslots, level=0, fill_value=0)
#     .rename(
#         columns={
#             "flight_id": "flights",
#             "duration_planned": "departure_duration",
#         }
#     )
# )
