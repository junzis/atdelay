from numpy.lib.function_base import extract
import pandas as pd
import os
from glob import glob
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

ICAOTOP1 = ["EGLL"]
ICAOTOP5 = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD"]
ICAOTOP10 = [
    "EGLL",
    "LFPG",
    "EHAM",
    "EDDF",
    "LEMD",
    "LEBL",
    "LTFM",
    "UUEE",
    "EDDM",
    "EGKK",
]
ICAOTOP25 = [
    "EGLL",
    "LFPG",
    "EHAM",
    "EDDF",
    "LEMD",
    "LEBL",
    "LTFM",
    "UUEE",
    "EDDM",
    "EGKK",
    "LIRF",
    "EIDW",
    "LFPO",
    "LOWW",
    "LSZH",
    "LPPT",
    "EKCH",
    "LEPA",
    "EGCC",
    "LIMC",
    "ENGM",
    "UUDD",
    "EGSS",
    "EBBR",
    "ESSA",
]

ICAOTOP50 = [
    "EGLL",
    "LFPG",
    "EHAM",
    "EDDF",
    "LEMD",
    "LEBL",
    "LTFM",
    "UUEE",
    "EDDM",
    "EGKK",
    "LIRF",
    "EIDW",
    "LFPO",
    "LOWW",
    "LSZH",
    "LPPT",
    "EKCH",
    "LEPA",
    "EGCC",
    "LIMC",
    "ENGM",
    "UUDD",
    "EGSS",
    "EBBR",
    "ESSA",
    "LGAV",
    "EDDL",
    "EDDT",
    "UUWW",
    "EFHK",
    "LEMG",
    "ULLI",
    "EPWA",
    "EGGW",
    "LSGG",
    "LKPR",
    "EDDH",
    "LHBP",
    "LTBA",
    "UKBB",
    "LEAL",
    "EGPH",
    "LROP",
    "LFMN",
    "LIME",
    "LPPR",
    "EDDS",
    "EGBB",
    "EDDK",
    "LFLL"
]

marketSegments = [
    "Traditional Scheduled",
    "Lowcost",
]


marketSegmentsExtended = [
    "Business Aviation",
    "All-Cargo",
    "Traditional Scheduled",
    "Lowcost",
    "Charter",
]


def extractData(
    start: datetime = None,
    end: datetime = None,
    saveFolder: str = "data/flights",
    folderName: str = "data",
    marketSegments=marketSegments,
):
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

        # Adding format makes it 10x faster (80 seconds vs 8)
        timeCols = [
            "FILED OFF BLOCK TIME",
            "FILED ARRIVAL TIME",
            "ACTUAL OFF BLOCK TIME",
            "ACTUAL ARRIVAL TIME",
        ]
        P[timeCols] = P[timeCols].apply(pd.to_datetime, format=dform)

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
            # .assign(FiledOBT=lambda x: x.FiledOBT.to_datetime(dform))
            # .assign(FiledAT=lambda x: x.FiledAT.to_datetime(dform))
            # .assign(ActualOBT=lambda x: x.ActualOBT.to_datetime(dform))
            # .assign(ActualAT=lambda x: x.ActualAT.to_datetime(dform))
            # .assign(FiledOBT=lambda x: x["FILED OFF BLOCK TIME"])
            # .assign(FiledAT=lambda x: x["FILED ARRIVAL TIME"])
            # .assign(ActualOBT=lambda x: x["ACTUAL OFF BLOCK TIME"])
            # .assign(ActualAT=lambda x: x["ACTUAL ARRIVAL TIME"])
        )
        finalData = finalData.append(P, ignore_index=True)

    # finalData = finalData.
    finalData = (
        finalData.sort_values(by=["ECTRLID"])
        .drop_duplicates("ECTRLID")
        .reset_index(drop=True)
    )

    return finalData


def calculateDelays(P, delayTypes=["arrival", "departure"]):
    if "arrival" in delayTypes:
        P = P.assign(ArrivalDelay=lambda x: (x.ActualAT - x.FiledAT).astype("timedelta64[m]"))
    if "departure" in delayTypes:
        P = P.assign(DepartureDelay=lambda x: (x.ActualOBT - x.FiledOBT).astype("timedelta64[m]"))
    return P


def filterContinental(P, airports):
    P = P.query("`ADEP` in @airports & `ADES` in @airports")
    return P


def linearRegressionFormat(P, airports=ICAOTOP25):
    columns = ["ADEP", "ADES", "FiledOBT", "FiledAT", "ACType", "ACOperator", "ArrivalDelay", "DepartureDelay"]
    P = filterContinental(P, airports)
    P = calculateDelays(P)
    P = P.loc[:,columns]
    P["FiledOBT"] = P["FiledOBT"].apply(lambda x: x.value)
    P["FiledAT"] = P["FiledAT"].apply(lambda x: x.value)
    return P


def saveToCSV(P, saveFolder="LRData"):
    if not os.path.exists(saveFolder):
        os.mkdir(os.path.join(saveFolder))
    P.to_csv(f"{saveFolder}/LRDATA.csv")


def readLRDATA(saveFolder="LRData", fileName="LRDATA.csv"):
    fullfilename = f"{saveFolder}/{fileName}"
    P = pd.read_csv(fullfilename, header=0, index_col=0)
    return P

if __name__ == "__main__":
    start = datetime(2015, 1, 1)
    end = datetime(2019, 12, 31)
    airports = ICAOTOP50
    print(f"Generating for {len(airports)} Airports")

    a = extractData(start, end)
    a = linearRegressionFormat(a, airports)
    saveToCSV(a)

    print(readLRDATA().head(50))