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


def readCSV(file, hello):
    pass


def extractData(
    start: datetime, end: datetime, airports: list = ICAOTOP1, folderName="data"
):

    # Basic input validation
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

    finalData = pd.DataFrame(
        columns=[
            "ECTRL ID",
            "ADEP",
            "ADEP Latitude",
            "ADEP Longitude",
            "ADES",
            "ADES Latitude",
            "ADES Longitude",
            "FILED OFF BLOCK TIME",
            "FILED ARRIVAL TIME",
            "ACTUAL OFF BLOCK TIME",
            "ACTUAL ARRIVAL TIME",
            "AC Type",
            "AC Operator",
            "AC Registration",
            "ICAO Flight Type",
            "STATFOR Market Segment",
            "Requested FL",
            "Actual Distance Flown (nm)",
        ]
    )

    for file in tqdm(listOfFiles):
        # read, filter and process csv
        P = pd.read_csv(file)

        # Adding format makes it 10x faster (80 seconds vs 8)
        P[
            [
                "FILED OFF BLOCK TIME",
                "FILED ARRIVAL TIME",
                "ACTUAL OFF BLOCK TIME",
                "ACTUAL ARRIVAL TIME",
            ]
        ] = P[
            [
                "FILED OFF BLOCK TIME",
                "FILED ARRIVAL TIME",
                "ACTUAL OFF BLOCK TIME",
                "ACTUAL ARRIVAL TIME",
            ]
        ].apply(
            pd.to_datetime, format="%d-%m-%Y %H:%M:%S"
        )

        # P = P[(P["ADEP"].isin(airports)) | (P["ADES"].isin(airports))]
        P = P[(P["ADEP"].isin(airports))]

        # This is kinda wonky atm
        P = P[
            (P["FILED OFF BLOCK TIME"] >= start) | (P["ACTUAL OFF BLOCK TIME"] >= start)
        ]
        P = P[(P["FILED ARRIVAL TIME"] <= end) | (P["ACTUAL ARRIVAL TIME"] <= end)]

        finalData = finalData.append(P, ignore_index=True)

    finalData = finalData.sort_values(by=["ECTRL ID"])

    return finalData
    # print(listOfFiles)


def delayCalc(P):
    P["delay"] = P["ACTUAL ARRIVAL TIME"] - P["FILED ARRIVAL TIME"]
    P["delayMinutes"] = P["delay"].astype("timedelta64[m]")

    return P


if __name__ == "__main__":
    start = datetime(2016, 1, 1)
    end = datetime(2016, 12, 31)
    airports = ICAOTOP25
    print(f"Testing {len(airports)} Airports.")
    a = extractData(start, end, airports)
    # print(a.iloc[0]["FILED OFF BLOCK TIME"])
    a = delayCalc(a)
    # print(a[['FILED ARRIVAL TIME', 'ACTUAL ARRIVAL TIME', "delay", "delayMinutes"]].head(50))
    # print(a[['FILED ARRIVAL TIME', 'ACTUAL ARRIVAL TIME',
    #       "delay", "delayMinutes"]].dtypes)
    # print(a["delayMinutes"].max())
    # print(a["delayMinutes"].min())
    lstofData = []
    for airport in airports:
        lstofData.append(a[a["ADEP"] == airport]["delayMinutes"].to_numpy())
        # plt.boxplot(a[a["ADEP"] == airport]["delayMinutes"].to_numpy(),
        #             showfliers=False, labels=[airport])
    plt.boxplot(lstofData, showfliers=False, labels=airports)
    plt.suptitle("Delay in minutes from departing airports")
    plt.grid()
    plt.show()
