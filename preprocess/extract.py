import pandas as pd
import os
from glob import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from preprocess.common import airports_top50, marketSegments
from preprocess.airportvalues import airport_dict
from preprocess.weather import fetch_weather_data


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
        listOfFiles.extend(glob(f"{folderName}/{year}/*/Flights_2*.csv*"))

    finalData = pd.DataFrame()

    buffer = []

    for file in tqdm(listOfFiles):
        # read, filter and process csv
        df = pd.read_csv(file)

        # Datetime format
        dform = "%d-%m-%Y %H:%M:%S"

        df = (
            df.query("`ICAO Flight Type` == 'S'")
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
        buffer.append(df)

    finalData = (
        pd.concat(buffer, ignore_index=True)
        .sort_values(by=["ECTRLID"])
        .drop_duplicates("ECTRLID")
        .reset_index(drop=True)
    )

    return finalData


def calculateDelays(df: pd.DataFrame, delayTypes: list = ["arrival", "departure"]):
    """ " calculate delay for both arrival and departure in minutes

    Args:
        df (pd.DataFrame): Pandas flights dataframe
        delayTypes (list, list): arrival and departure times. Defaults to ["arrival", "departure"].

    Returns:
        pd.DataFrame: Pandas flights dataframe with delays
    """
    if "arrival" in delayTypes:
        df = df.assign(
            ArrivalDelay=lambda x: (x.ActualAT - x.FiledAT).astype("timedelta64[m]")
        )
    if "departure" in delayTypes:
        df = df.assign(
            DepartureDelay=lambda x: (x.ActualOBT - x.FiledOBT).astype("timedelta64[m]")
        )

    df = df.query(
        "ArrivalDelay < 90 & ArrivalDelay > -30 & DepartureDelay < 90 & DepartureDelay > -30 "
    )

    return df


def filterAirports(df: pd.DataFrame, airports: list):
    """Filter pandas airport arrivals and departures to a list of airports

    Args:
        df (pd.DataFrame): Pandas flights dataframe
        airports (list): list of airports to keep

    Returns:
        pd.DataFrame: filtered flights dataframe
    """

    df = df.query("`ADEP` in @airports | `ADES` in @airports")
    return df


def linearRegressionFormat(df: pd.DataFrame, airports: list = airports_top50):
    """Converts a complete extracted dataframe into the format used for linear regression

    Args:
        df (pd.DataFrame): complete unfiltered pandas dataframe
        airports (list, optional): list of airports. Defaults to airports_top50.

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
    df = filterAirports(df, airports)
    df = calculateDelays(df)
    df = df.loc[:, columns]
    df["month"] = df["FiledAT"].dt.month
    df["weekday"] = df["FiledAT"].dt.weekday
    df["filedATminutes"] = df["FiledAT"].dt.hour * 60 + df["FiledAT"].dt.minute
    df["filedOBTminutes"] = df["FiledOBT"].dt.hour * 60 + df["FiledOBT"].dt.minute

    # df = df.drop(["FiledOBT", "FiledAT"], axis=1)

    return df


def saveToCSV(df: pd.DataFrame, saveFolder: str = "LRData"):
    """Convert the flights dataframe to a CSV

    Args:
        df (pd.DataFrame): Pandas flights dataframe
        saveFolder (str): name folder to save the CSV file in. Defaults to "LRData".
    """

    if not os.path.exists(saveFolder):
        os.mkdir(os.path.join(saveFolder))
    df.to_csv(f"{saveFolder}/LRDATA.csv")


def readLRDATA(saveFolder: str = "LRData", fileName: str = "LRDATA.csv"):
    """Read data from a flights dataframe in linear regression format

    Args:
        saveFolder (str, optional): folder where data is saved. Defaults to "LRData".
        fileName (str, optional): filename of the dataset. Defaults to "LRDATA.csv".

    Returns:
        pd.Dataframe: flights dataframe in linear regression format
    """
    fullfilename = f"{saveFolder}/{fileName}"
    df = pd.read_csv(fullfilename, header=0, index_col=0)
    return df


def generalFilterAirport(
    start: datetime,
    end: datetime,
    airport: str,
    saveFolder: str = "data/filtered",
    forceRegenerateData: bool = False,
):
    """Generate all the flights for a single airport, save and return as dataframe

    Args:
        start (datetime): start date to filter for. Dates are inclusive.
        end (datetime): end date to filter for. Dates are inclusive.
        airport (str): ICAO code for the airport
        saveFolder (str, optional): target save folder. Defaults to "data/filtered".
        forceRegenerateData (bool, optional): force regeneration of data even if it had already been generated. Defaults to False.

    Returns:
        pd.DataFrame: Dataframe with all flights for selected filters
    """
    file = f"{saveFolder}/general{airport}.csv"
    dform = "%Y-%m-%d %H:%M:%S"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    # For the first cold run it generates data for all dates to prevent problems
    if not os.path.exists(file) or forceRegenerateData:
        print(f"Generating {airport} airport data from {start} to {end}")
        df = extractData(start, end)
        df = df.query("`ADES` == @airport | `ADEP` == @airport")
        df = calculateDelays(df)
        df.to_csv(file)
    else:
        df = pd.read_csv(file, header=0, index_col=0)
        # Condvert datetime strings to actual datetime objects
        df = (
            df.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
            .assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))
            .assign(ActualOBT=lambda x: pd.to_datetime(x.ActualOBT, format=dform))
            .assign(ActualAT=lambda x: pd.to_datetime(x.ActualAT, format=dform))
        )

    # Actual date filter.
    # Does NOT include flights that departed the night before but arrived within the filter
    df = df.query("`FiledOBT` >= @start & `FiledAT` <= @end")

    return df


def generateNNdata(
    airport: str,
    timeinterval: int = 30,
    GNNFormat: bool = False,
    disableWeather: bool = True,
    saveFolder: str = "data/nn",
    catagoricalFlightDuration: bool = False,
    forceRegenerateData: bool = False,
    start=datetime(2018, 1, 1),
    end=datetime(2019, 12, 31),
    availableMonths: list = [3, 6, 9, 12],
):
    """Aggregates all flights at a single airport by a certain timeslot.

    Args:
        airport (str): ICAO code for a single airport
        timeinterval (int, optional): length to aggregate flights for in minutes. Defaults to 15 minutes.
        GNNFormat: (bool, optional): returns the data in format used for GNN model (df_agg, Y, T). Defaults to False
        disableWeather: (bool, optional): disables weather features:\
             (["timeslot", "visibility", "windspeed",\
               "temperature", "frozenprecip", \
               "surfaceliftedindex", "cape"]). Defaults to True.

        saveFolder (str, optional): folder to save data in. Defaults to "NNData".
        catagoricalFlightDelay (bool, optional): If false, flight delay is presented as average.\
             If True it is generated as bins from 0-3, 3-6 and >6. Defaults to False.
        forceRegenerateData (bool, optional): force regeneration of data even if it had already been generated. Defaults to False.
        start (datetime, optional): start date to filter for.
        end (datetime, optional): end date to filter for.
        start (datetime, optinoal): start date to generate full data. Defaults to datetime(2019, 1, 31)
        end (datetime, optinoal): end date to generate full data. Defaults to datetime(2019, 12, 31)
        availableMonths (list, optional): list of months available in \
            eurocontrol. Defaults to [March, June, September, December]
    Returns:
        pd.Dataframe: pandas dataframe with aggregate flight data, unscaled.
    """
    filename = f"{saveFolder}/{airport}_{timeinterval}m.csv"

    dform = "%Y-%m-%d %H:%M:%S"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if not os.path.exists(filename) or forceRegenerateData:
        print(
            f"Generating NN data for {airport} with a timeslot length of {timeinterval} minutes"
        )
        df = generalFilterAirport(start, end, airport)

        # Temporary untill weather is added:
        numRunways = 0
        numGates = 0

        ### Data preparation for agg function
        # Are flights arriving or departing?
        df["arriving"] = df.ADES == airport
        df["departing"] = df.ADEP == airport

        # Is it a low cost flight?
        df["lowcost"] = df.FlightType != "Traditional Scheduled"

        # Planned Flight Duration (PFD) in minutes
        df["PFD"] = df["FiledAT"] - df["FiledOBT"]
        df["PFD"] = (
            df["PFD"].dt.components["hours"] * 60 + df["PFD"].dt.components["minutes"]
        )

        # Flight duration for arriving airplanes
        df.loc[(df.arriving == False), "departuresFlightDuration"] = df.PFD
        df.loc[(df.arriving == True), "arrivalsFlightDuration"] = df.PFD

        df["departuresFlightDuration0to3"] = df.departuresFlightDuration < 3 * 60
        df["departuresFlightDuration3to6"] = (df.departuresFlightDuration >= 3 * 60) & (
            df.departuresFlightDuration < 6 * 60
        )
        df["departuresFlightDuration6orMore"] = df.departuresFlightDuration >= 6 * 60

        df["arrivalsFlightDuration0to3"] = df.arrivalsFlightDuration < 3 * 60
        df["arrivalsFlightDuration3to6"] = (df.arrivalsFlightDuration >= 3 * 60) & (
            df.arrivalsFlightDuration < 6 * 60
        )
        df["arrivalsFlightDuration6orMore"] = df.arrivalsFlightDuration >= 6 * 60

        # Delay metrics for arriving and departing airports
        df.loc[(df.arriving == True), "arrivalsDepartureDelay"] = df.DepartureDelay
        df.loc[(df.arriving == True), "arrivalsArrivalDelay"] = df.ArrivalDelay
        df.loc[(df.arriving == False), "departuresDepartureDelay"] = df.DepartureDelay
        df.loc[(df.arriving == False), "departuresArrivalDelay"] = df.ArrivalDelay

        # Collect the time at which the flights are meant to be at the airport
        df.loc[(df.arriving == True), "timeAtAirport"] = df.FiledAT
        df.loc[(df.arriving == False), "timeAtAirport"] = df.FiledOBT

        # This creates a new index to ensure that we have no gaps in the timeslots later
        def daterange(start_date, end_date):
            delta = timedelta(minutes=timeinterval)
            while start_date < end_date:
                if start_date.month in availableMonths:
                    # Only yields the months for which we have
                    # data specified in the argument availableMonths
                    yield start_date
                start_date += delta

        denseDateIndex = daterange(start, end)

        # Functionality for airports outside of the top50
        if airport in list(airport_dict.keys()):
            airportCapacity = airport_dict[airport]["capacity"]
        else:
            airportCapacity = 60  # this is a common value

        # weatherData = fetch_weather_data(airport, timeinterval)

        ### get aggregate features for rolling window
        df_agg = (
            df.groupby(
                [
                    pd.Grouper(key="timeAtAirport", freq=f"{timeinterval}min"),
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
                }
            )
            # This ensure that there are no timeslot gaps
            # at the start and end of the dataframe
            .reindex(denseDateIndex, fill_value=0)
            .assign(planes=lambda x: x.arriving - x.departing)
            .assign(
                capacityFilled=lambda x: (x.arriving + x.departing)
                / airportCapacity
                * 100
            )
            .assign(date=lambda x: x.index.date)
            .assign(weekday=lambda x: x.index.weekday)
            .assign(month=lambda x: x.index.month)
            .assign(hour=lambda x: x.index.hour)
            .reset_index()
            .rename(columns={"timeAtAirport": "timeslot"})
            # Add weather data
            # .merge(weatherData, how="left", on="timeslot", validate="1:m")
            .fillna(0)
        )

        # turn boolean columns into 1 and 0
        boolCols = df_agg.columns[df_agg.dtypes.eq(bool)]
        df_agg.loc[:, boolCols] = df_agg.loc[:, boolCols].astype(int)

        df_agg.to_csv(filename)

    else:
        df_agg = pd.read_csv(filename, header=0, index_col=0)
        df_agg = df_agg.assign(
            timeslot=lambda x: pd.to_datetime(x.timeslot, format=dform)
        )

    df_agg = df_agg.query("`timeslot` >= @start & `timeslot` < @end")

    if disableWeather:
        df_agg = df_agg.drop(
            [
                "visibility",
                "windspeed",
                "temperature",
                "frozenprecip",
                "surfaceliftedindex",
                "cape",
            ],
            axis=1,
            errors="ignore",
        )

    if GNNFormat and catagoricalFlightDuration:
        raise ValueError("GNNFormat and catagoricalFlightDuration are not compatible")

    if GNNFormat:
        Y = df_agg.loc[:, ["arrivalsArrivalDelay", "departuresDepartureDelay"]]
        T = df_agg.loc[:, ["timeslot"]]
        df_agg = df_agg.drop(
            [
                "arrivalsArrivalDelay",
                "departuresDepartureDelay",
                "departuresArrivalDelay",
                # "timeslot",
                "planes",
                "arrivalsDepartureDelay",
            ],
            axis=1,
        )
        return df_agg, Y, T
    else:
        return df_agg


def generateNNdataMultiple(
    airports: list,
    timeinterval: int = 30,
    GNNFormat: bool = False,
    disableWeather: bool = True,
    saveFolder: str = "data/nn",
    forceRegenerateData: bool = False,
    start: datetime = datetime(2018, 1, 1),
    end: datetime = datetime(2019, 12, 31),
):
    """Generates NN data for many airports and results all as a dict

    Args:
        airports (list): list of ICAO airport codes
        timeinterval (int, optional): length to aggregate flights for in minutes. Defaults to 15 minutes.
        GNNFormat: (bool, optional): returns the data in format used for GNN model (df_agg, Y, T). Defaults to False
        disableWeather: (bool, optional): disables weather features:\
                        (["timeslot", "visibility", "windspeed",\
                        "temperature", "frozenprecip", \
                        "surfaceliftedindex", "cape"]). Defaults to True.
        saveFolder (str, optional): folder to save data in. Defaults to "NNData".
        forceRegenerateData (bool, optional): force regeneration of data even if it had already been generated. Defaults to False.
        start (datetime, optional): start date to filter for.
        end (datetime, optional): end date to filter for.

    Returns:
        dict: dictionary of NN data dataframes
    """
    dataDict = {}
    for airport in tqdm(airports):
        result = generateNNdata(
            airport,
            timeinterval,
            GNNFormat,
            disableWeather,
            saveFolder,
            forceRegenerateData,
            start=start,
            end=end,
        )
        if GNNFormat:
            result = {
                "X": result[0],
                "Y": result[1],
                "T": result[2],
            }

        dataDict[airport] = result

    return dataDict
