import pandas as pd
import os
from glob import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from extraction.extractionvalues import *
from extraction.airportvalues import *


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
            .query("ADES != ADEP")
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
    P = P.query(
        "ArrivalDelay < 90 & ArrivalDelay > -30 & DepartureDelay < 90 & DepartureDelay > -30 "
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
        "ADEPLat",
        "ADEPLong",
        "ADESLat",
        "ADESLong",
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


def generalFilterAirport(
    start: datetime,
    end: datetime,
    airport: str,
    saveFolder: str = "filteredData",
    forceRegenerateData: bool = False,
):
    """Generate all the flights for a single airport, save and return as dataframe

    Args:
        start (datetime): start date to filter for
        end (datetime): end date to filter for
        airport (str): ICAO code for the airport
        saveFolder (str, optional): target save folder. Defaults to "filteredData".

    Returns:
        pd.DataFrame: Dataframe with all flights for selected filters
    """
    file = f"{saveFolder}/general{airport}.csv"
    dform = "%Y-%m-%d %H:%M:%S"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if not os.path.exists(file) or forceRegenerateData:
        print(f"Generating {airport} airport data from {start} to {end}")
        P = extractData(start, end)
        P = P.query("`ADES` == @airport | `ADEP` == @airport")
        P = calculateDelays(P)
        P.to_csv(file)
    else:
        P = pd.read_csv(file, header=0, index_col=0)
        # Condvert datetime strings to actual datetime objects
        P = (
            P.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
            .assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))
            .assign(ActualOBT=lambda x: pd.to_datetime(x.ActualOBT, format=dform))
            .assign(ActualAT=lambda x: pd.to_datetime(x.ActualAT, format=dform))
        )

    return P


def generateNNdata(
    airport: str,
    timeslotLength: int = 15,
    GNNFormat: bool = False,
    saveFolder: str = "NNData",
    catagoricalFlightDuration: bool = False,
    forceRegenerateData: bool = False,
    start: datetime = datetime(2018, 1, 1),
    end: datetime = datetime(2019, 12, 31),
):
    """Aggregates all flights at a single airport by a certain timeslot.

    Args:
        airport (str): ICAO code for a single airport
        timeslotLength (int, optional): length to aggregate flights for in minutes. Defaults to 15 minutes.
        saveFolder (str, optional): folder to save data in. Defaults to "NNData".
        catagoricalFlightDelay (bool, optional): If false, flight delay is presented as average.\
             If True it is generated as bins from 0-3, 3-6 and >6. Defaults to False.
        forceRegenerateData (bool, optional): force regeneration of data even if it had already been generated. Defaults to False.
        start (datetime, optional): start date to filter for
        end (datetime, optional): end date to filter for
    Returns:
        pd.Dataframe: pandas dataframe with aggregate flight data, unscaled.
    """
    filename = f"{saveFolder}/{airport}_{timeslotLength}m.csv"

    dform = "%Y-%m-%d %H:%M:%S"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if not os.path.exists(filename) or forceRegenerateData:
        print(
            f"Generating NN data for {airport} with a timeslot length of {timeslotLength} minutes"
        )
        P = generalFilterAirport(start, end, airport)

        # Temporary untill weather is added:
        numRunways = 0
        numGates = 0

        ### Data preparation for agg function
        # Are flights arriving or departing?
        P["arriving"] = P.ADES == airport
        P["departing"] = P.ADEP == airport

        # Is it a low cost flight?
        P["lowcost"] = P.FlightType != "Traditional Scheduled"

        # Planned Flight Duration (PFD) in minutes
        P["PFD"] = P["FiledAT"] - P["FiledOBT"]
        P["PFD"] = (
            P["PFD"].dt.components["hours"] * 60 + P["PFD"].dt.components["minutes"]
        )

        # Flight duration for arriving airplanes
        P.loc[(P.arriving == False), "departuresFlightDuration"] = P.PFD
        P.loc[(P.arriving == True), "arrivalsFlightDuration"] = P.PFD

        P["departuresFlightDuration0to3"] = P.departuresFlightDuration < 3 * 60
        P["departuresFlightDuration3to6"] = (P.departuresFlightDuration >= 3 * 60) & (
            P.departuresFlightDuration < 6 * 60
        )
        P["departuresFlightDuration6orMore"] = P.departuresFlightDuration >= 6 * 60

        P["arrivalsFlightDuration0to3"] = P.arrivalsFlightDuration < 3 * 60
        P["arrivalsFlightDuration3to6"] = (P.arrivalsFlightDuration >= 3 * 60) & (
            P.arrivalsFlightDuration < 6 * 60
        )
        P["arrivalsFlightDuration6orMore"] = P.arrivalsFlightDuration >= 6 * 60

        # Delay metrics for arriving and departing airports
        P.loc[(P.arriving == True), "arrivalsDepartureDelay"] = P.DepartureDelay
        P.loc[(P.arriving == True), "arrivalsArrivalDelay"] = P.ArrivalDelay
        P.loc[(P.arriving == False), "departuresDepartureDelay"] = P.DepartureDelay
        P.loc[(P.arriving == False), "departuresArrivalDelay"] = P.ArrivalDelay

        # Collect the time at which the flights are meant to be at the airport
        P.loc[(P.arriving == True), "timeAtAirport"] = P.FiledAT
        P.loc[(P.arriving == False), "timeAtAirport"] = P.FiledOBT

        # P = P.fillna(0)

        ### get aggregate features for rolling window
        Pagg = (
            P.groupby(
                [
                    pd.Grouper(key="timeAtAirport", freq=f"{timeslotLength}min"),
                ]
            )
            .agg(
                {
                    "departing": "sum",
                    "arriving": "sum",
                    "lowcost": "mean",
                    "arrivalsFlightDuration": "mean",
                    "arrivalsDepartureDelay": "mean",
                    "arrivalsArrivalDelay": "mean",
                    "departuresFlightDuration": "mean",
                    "departuresDepartureDelay": "mean",
                    "departuresArrivalDelay": "mean",
                    "departuresFlightDuration0to3": "mean",
                    "departuresFlightDuration3to6": "mean",
                    "departuresFlightDuration6orMore": "mean",
                    "arrivalsFlightDuration0to3": "mean",
                    "arrivalsFlightDuration3to6": "mean",
                    "arrivalsFlightDuration6orMore": "mean",
                }
            )
            .assign(planes=lambda x: x.arriving - x.departing)
            .assign(runways=lambda x: numRunways)
            .assign(gates=lambda x: numGates)
            .assign(
                capacityFilled=lambda x: (x.arriving + x.departing)
                / airport_dict[airport]["capacity"]
            )
            .assign(weekend=lambda x: x.index.weekday >= 5)
            .assign(winter=lambda x: (x.index.month > 11) | (x.index.month < 3))
            .assign(spring=lambda x: (x.index.month > 2) & (x.index.month < 6))
            .assign(summer=lambda x: (x.index.month > 5) & (x.index.month < 9))
            .assign(autumn=lambda x: (x.index.month > 8) & (x.index.month < 12))
            .assign(night=lambda x: (x.index.hour >= 0) & (x.index.hour < 6))
            .assign(morning=lambda x: (x.index.hour >= 6) & (x.index.hour < 12))
            .assign(afternoon=lambda x: (x.index.hour >= 12) & (x.index.hour < 18))
            .assign(evening=lambda x: (x.index.hour >= 18) & (x.index.hour <= 23))
            .drop(["runways", "gates"], axis=1)  # Temp measure until we add weather
            .reset_index()
            .rename(columns={"timeAtAirport": "timeslot"})
            .fillna(0)
        )

        # turn boolean columns into 1 and 0
        boolCols = Pagg.columns[Pagg.dtypes.eq(bool)]
        Pagg.loc[:, boolCols] = Pagg.loc[:, boolCols].astype(int)

        # there are two ways the team wanted the flight
        # duration in bins of 3 hours or as an average,
        #  here the data gets augmented based on the chase
        if catagoricalFlightDuration:
            Pagg = Pagg.drop(
                ["departuresFlightDuration", "arrivalsFlightDuration"], axis=1
            )
        else:
            Pagg = Pagg.drop(
                [
                    "departuresFlightDuration0to3",
                    "departuresFlightDuration3to6",
                    "departuresFlightDuration6orMore",
                    "arrivalsFlightDuration0to3",
                    "arrivalsFlightDuration3to6",
                    "arrivalsFlightDuration6orMore",
                ],
                axis=1,
            )

        Pagg.to_csv(filename)

    else:
        Pagg = pd.read_csv(filename, header=0, index_col=0)
        Pagg = Pagg.assign(timeslot=lambda x: pd.to_datetime(x.timeslot, format=dform))

    if GNNFormat and catagoricalFlightDuration:
        raise ValueError("GNNFormat and catagoricalFlightDuration are not compatible")

    if GNNFormat:
        Y = Pagg.loc[:, ["arrivalsArrivalDelay", "departuresDepartureDelay"]]
        T = Pagg.loc[:, ["timeslot"]]
        Pagg = Pagg.drop(
            [
                "arrivalsArrivalDelay",
                "departuresDepartureDelay",
                "departuresArrivalDelay",
                "timeslot"
            ],
            axis=1,
        )
        return Pagg, Y, T
    else:
        return Pagg


def generateNNdataMultiple(
    airports: list,
    timeslotLength: int = 15,
    GNNFormat: bool = False,
    saveFolder: str = "NNData",
    forceRegenerateData: bool = False,
    start: datetime = datetime(2018, 1, 1),
    end: datetime = datetime(2019, 12, 31),
):
    """Generates NN data for many airports and results all as a dict

    Args:
        airports (list): list of ICAO airport codes
        timeslotLength (int, optional): length to aggregate flights for in minutes. Defaults to 15 minutes.
        saveFolder (str, optional): folder to save data in. Defaults to "NNData".
        forceRegenerateData (bool, optional): force regeneration of data even if it had already been generated. Defaults to False.


    Returns:
        dict: dictionary of NN data dataframes
    """
    dataDict = {}
    for airport in tqdm(airports):
        result = generateNNdata(
            airport,
            timeslotLength,
            GNNFormat,
            saveFolder,
            forceRegenerateData,
            start=start,
            end=end,
        )
        if GNNFormat:
            result = {"X": result[0], "Y": result[1]}

        dataDict[airport] = result

    return dataDict


def show_heatmap(P: pd.DataFrame, dtkey: str = None):
    """Shows a heatmap of correlations for a pandas df

    Args:
        P (pd.DataFrame): pandas data
        dtkey (str, optional): dt column name for removal. Defaults to None.
    """

    if dtkey is not None:
        P = P.drop([dtkey], axis=1)

    plt.matshow(P.corr(), cmap="RdBu_r", vmin=-1, vmax=1)
    plt.xticks(range(P.shape[1]), P.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(P.shape[1]), P.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)


def show_raw_visualization(P: pd.DataFrame, date_time_key="timeslot"):
    """Show features of an NN dataframe over time

    Args:
        data (pd.dataFrame): pandas dataframe in NN format
        date_time_key (str, optional): column that provides datetime. Defaults to "timeslot".
    """
    ncols = 3
    time_data = P[date_time_key]
    feature_keys = P.columns
    fig, axes = plt.subplots(
        nrows=(len(feature_keys) + ncols - 1) // ncols,
        ncols=ncols,
        figsize=(20, 15),
        dpi=70,
        sharex=True,
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = plotcolors[i % (len(plotcolors))]
        t_data = P[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // ncols, i % ncols],
            color=c,
            title=key,
            rot=25,
        )
        # ax.legend(key)
        ax.grid()
    plt.tight_layout()


if __name__ == "__main__":
    pass
