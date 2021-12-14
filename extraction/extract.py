from numpy.lib.function_base import extract
import pandas as pd
import os
from glob import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from extractionvalues import *
import numpy as np


def extractData(
    start: datetime = None,
    end: datetime = None,
    folderName: str = "data",
    marketSegments: list = marketSegments,
):
    """extract raw data from eurocontrol data and converts it into a pandas dataframe

    Args:
        start (datetime, optional): start time to extract data. Defaults to None.
        end (datetime, optional): final date to extract data. Defaults to None.
        folderName (str, optional): foldername to take data from. Defaults to "data".
        marketSegments ([type], optional): list of market segments to consider default is commercial scheduled. Defaults to marketSegments.

    Raises:
        ValueError: date needs to be between start of 2015 and end of 2019

    Returns:
        pd.DataFrame: complete pandas flights dataframe
    """

    # Basic input validation
    if start is None and end is None:
        start = datetime(2015, 1, 1)
        end = datetime(2019, 12, 31)
    if start.year < 2015 or start.year > 2019:
        raise ValueError(f"Incorrect start date (start between 2015 and 2019) {start}")
    if end.year < 2015 or end.year > 2019:
        raise ValueError(f"Incorrect end date (end between 2015 and 2019) {end}")
    if end.year < start.year:
        raise ValueError(f"Entered end before start ({start} > {end})")

    years = list(range(start.year, end.year + 1))
    listOfFiles = []
    for year in years:
        # Dank file selection https://pynative.com/python-glob/
        listOfFiles.extend(glob(f"{folderName}\\{year}/*/Flights_2*.csv*"))

    finalData = pd.DataFrame()

    for file in tqdm(listOfFiles):
        # read, filter and process csv
        P = pd.read_csv(file)

        # Datetime format
        dform = "%d-%m-%Y %H:%M:%S"

        P = (
            P.query("`ICAO Flight Type` == 'S'")
            .query("`STATFOR Market Segment` in @marketSegments")
            .rename(columns={"FILED OFF BLOCK TIME": "FiledOBT"})
            .rename(columns={"FILED ARRIVAL TIME": "FiledAT"})
            .rename(columns={"ACTUAL OFF BLOCK TIME": "ActualOBT"})
            .rename(columns={"ACTUAL ARRIVAL TIME": "ActualAT"})
            .rename(columns={"STATFOR Market Segment": "FlightType"})
            .rename(columns={"ADEP Latitude": "ADEPLat"})
            .rename(columns={"ADEP Longitude": "ADEPLong"})
            .rename(columns={"ADES Latitude": "ADESLat"})
            .rename(columns={"ADES Longitude": "ADESLong"})
            .rename(columns={"AC Type": "ACType"})
            .rename(columns={"AC Type": "ACType"})
            .rename(columns={"AC Operator": "ACOperator"})
            .rename(columns={"ECTRL ID": "ECTRLID"})
            .rename(columns={"Actual Distance Flown (nm)": "ActualDistanceFlown"})
            .drop(["AC Registration"], axis=1)
            .drop(["Requested FL"], axis=1)
            .drop(["ICAO Flight Type"], axis=1)
            .assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
            .assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))
            .assign(ActualOBT=lambda x: pd.to_datetime(x.ActualOBT, format=dform))
            .assign(ActualAT=lambda x: pd.to_datetime(x.ActualAT, format=dform))
        )
        finalData = finalData.append(P, ignore_index=True)

    # finalData = finalData.
    finalData = (
        finalData.sort_values(by=["ECTRLID"])
        .drop_duplicates("ECTRLID")
        .reset_index(drop=True)
    )

    return finalData


def calculateDelays(P: pd.DataFrame, delayTypes: list = ["arrival", "departure"]):
    """ " calculate delay for both arrival and departure in minutes

    Args:
        P (pd.DataFrame): Pandas flights dataframe
        delayTypes (list, list): arrival and departure times. Defaults to ["arrival", "departure"].

    Returns:
        pd.DataFrame: Pandas flights dataframe with delays
    """
    if "arrival" in delayTypes:
        P = P.assign(
            ArrivalDelay=lambda x: (x.ActualAT - x.FiledAT).astype("timedelta64[m]")
        )
    if "departure" in delayTypes:
        P = P.assign(
            DepartureDelay=lambda x: (x.ActualOBT - x.FiledOBT).astype("timedelta64[m]")
        )
    return P


def filterAirports(P: pd.DataFrame, airports: list):
    """Filter pandas airport arrivals and departures to a list of airports

    Args:
        P (pd.DataFrame): Pandas flights dataframe
        airports (list): list of airports to keep

    Returns:
        pd.DataFrame: filtered flights dataframe
    """

    P = P.query("`ADEP` in @airports & `ADES` in @airports")
    return P


def linearRegressionFormat(P: pd.DataFrame, airports: list = ICAOTOP50):
    """Converts a complete extracted dataframe into the format used for linear regression

    Args:
        P (pd.DataFrame): complete unfiltered pandas dataframe
        airports (list, optional): list of airports. Defaults to ICAOTOP25.

    Returns:
        pd.DataFrame: filtered pandas dataframe in LR format
    """
    columns = [
        "ADEP",
        "ADES",
        "FiledOBT",
        "FiledAT",
        "ACType",
        "ACOperator",
        "ArrivalDelay",
        "DepartureDelay",
    ]
    P = filterAirports(P, airports)
    P = calculateDelays(P)
    P = P.loc[:, columns]
    P["month"] = P["FiledAT"].dt.month
    P["weekday"] = P["FiledAT"].dt.weekday
    P["filedATminutes"] = P["FiledAT"].dt.hour * 60 + P["FiledAT"].dt.minute
    P["filedOBTminutes"] = P["FiledOBT"].dt.hour * 60 + P["FiledOBT"].dt.minute

    # P = P.drop(["FiledOBT", "FiledAT"], axis=1)

    return P


def saveToCSV(P: pd.DataFrame, saveFolder: str = "LRData"):
    """Convert the flights dataframe to a CSV

    Args:
        P (pd.DataFrame): Pandas flights dataframe
        saveFolder (str): name folder to save the CSV file in. Defaults to "LRData".
    """

    if not os.path.exists(saveFolder):
        os.mkdir(os.path.join(saveFolder))
    P.to_csv(f"{saveFolder}/LRDATA.csv")


def readLRDATA(saveFolder: str = "LRData", fileName: str = "LRDATA.csv"):
    """Read data from a flights dataframe in linear regression format

    Args:
        saveFolder (str, optional): folder where data is saved. Defaults to "LRData".
        fileName (str, optional): filename of the dataset. Defaults to "LRDATA.csv".

    Returns:
        pd.Dataframe: flights dataframe in linear regression format
    """
    fullfilename = f"{saveFolder}/{fileName}"
    P = pd.read_csv(fullfilename, header=0, index_col=0)
    return P


def generalFilterAirport(start, end, airport):
    file = f"filteredData/general{airport}.csv"
    dform = "%Y-%m-%d %H:%M:%S"
    if os.path.exists(file):
        P = pd.read_csv(file, header=0, index_col=0)
        P = (
            P.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
            .assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))
            .assign(ActualOBT=lambda x: pd.to_datetime(x.ActualOBT, format=dform))
            .assign(ActualAT=lambda x: pd.to_datetime(x.ActualAT, format=dform))
        )
    else:
        P = extractData(start, end)
        P = P.query("`ADES` == @airport | `ADEP` == @airport")
        P = calculateDelays(P)
        P.to_csv(file)

    return P


def generateNNdata(airport, numRunways=6, numGates=98):
    start = datetime(2018, 1, 1)
    end = datetime(2019, 12, 31)
    P = generalFilterAirport(start, end, airport)
    # P = P.query("`ADES`==@airport")
    # print(P.head())
    P["arriving"] = P.ADES == airport
    P["departing"] = P.ADEP == airport
    P["lowcost"] = P.FlightType != "Traditional Scheduled"
    minss = 5
    plotdays = 0.5
    arrivals = (
        P.groupby(
            [
                pd.Grouper(key="FiledAT", freq=f"{minss}min"),
                # "arriving",
            ]
        )
        .agg(
            {
                "departing": "sum",
                "arriving": "sum",
                # "ADES": "count",
                "DepartureDelay": "mean",
                "ArrivalDelay": "mean",
                "lowcost": "mean",
            }
        )
        .assign(runways=lambda x: numRunways)
        .assign(gates=lambda x: numGates)
        .assign(planes=lambda x: x.arriving - x.departing)
        .assign(weekend=lambda x: x.index.weekday >= 5)
        .assign(winter=lambda x: (x.index.month > 11) | (x.index.month < 3))
        .assign(spring=lambda x: (x.index.month > 2) & (x.index.month < 6))
        .assign(summer=lambda x: (x.index.month > 5) & (x.index.month < 9))
        .assign(autumn=lambda x: (x.index.month > 8) & (x.index.month < 12))
        .fillna(0)
    )

    Y = arrivals.ArrivalDelay
    arrivals = arrivals.drop("ArrivalDelay", axis=1)
    arrivals = arrivals.drop(["runways", "gates"], axis=1)
    arrivals = arrivals * 1
    # arrivals.weekend = arrivals.weekend.astype(int)
    # arrivals.winter = arrivals.winter.astype(int)
    # arrivals.spring = arrivals.spring.astype(int)
    # arrivals.summer = arrivals.summer.astype(int)
    # arrivals.autumn = arrivals.autumn.astype(int)
    # print(arrivals)
    arrivals.reset_index(inplace=True)
    return arrivals.rename(columns={"FiledAT": "timeslot"}), Y


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)


def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())


date_time_key = "timeslot"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    feature_keys = data.columns
    fig, axes = plt.subplots(
        nrows=(len(feature_keys) + 1) // 2,
        ncols=2,
        figsize=(20, 15),
        dpi=70,
        sharex=True,
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = plotcolors[i % (len(plotcolors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(ax=axes[i // 2, i % 2], color=c, title=key, rot=25,)
        # ax.legend(key)
        ax.grid()
    plt.tight_layout()


if __name__ == "__main__":
    # start = datetime(2015, 1, 1)
    # end = datetime(2019, 4, 30)
    # airports = ICAOTOP50
    # print(f"Generating for {len(airports)} Airports")

    # a = extractData(start, end)
    # saveToCSV(a)

    # print(readLRDATA().head(50))
    # print(len(a))

    X, Y = generateNNdata("EHAM")

    X["departing"] = min_max_scaling(X["departing"])
    X["arriving"] = min_max_scaling(X["arriving"])
    X["DepartureDelay"] = min_max_scaling(X["DepartureDelay"])
    X["lowcost"] = min_max_scaling(X["lowcost"])

    print(X)
    show_heatmap(X)
    show_raw_visualization(X)
    plt.show()
