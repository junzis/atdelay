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


def extractData(start: datetime, end: datetime, airports: list = None, folderName:str="data", europeanFlights:bool=False):

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
        if airports is not None:
            P = P[(P["ADEP"].isin(airports))]
            
        if europeanFlights:
            P = P[(P["ADEP"].str[0].isin(["E", "B", "L", "U"])) & (P["ADES"].str[0].isin(["E", "B", "L", "U"]))]

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
    P["arrivalDelay"] = P["ACTUAL ARRIVAL TIME"] - P["FILED ARRIVAL TIME"]
    P["blockoffDelay"] = P["ACTUAL OFF BLOCK TIME"] - P["FILED OFF BLOCK TIME"]
    P["flightTimeDifference"] = (P["ACTUAL ARRIVAL TIME"] - P["ACTUAL OFF BLOCK TIME"]) - (P["FILED ARRIVAL TIME"] - P["FILED OFF BLOCK TIME"])


    P["arrivalDelayMinutes"] = P["arrivalDelay"].astype("timedelta64[m]")
    P["blockoffDelayMinutes"] = P["blockoffDelay"].astype("timedelta64[m]")
    P["flightTimeDelayMinutes"] = P["flightTimeDifference"].astype("timedelta64[m]")

    P["On Time"] =  (P["arrivalDelayMinutes"]).abs() < 15.0

    return P


def airportBoxPlot(P, airports, delayType="blockoffDelayMinutes", showfliers=False):
    lstofData = []
    
    fig, ax = plt.subplots(1,1)

    for airport in airports:
        lstofData.append(P[P["ADEP"] == airport][delayType].to_numpy())

    ax.boxplot(lstofData, showfliers=showfliers, labels=airports)
    fig.suptitle(f"Delay in departure in minutes from departing airports ({delayType})")
    ax.set_xlabel("Airport ICAO code")
    ax.set_ylabel("Delay in Minutes")
    ax.grid()


def airlineOnTime(P, airlines, airlinesNames=None):
    if airlinesNames is None:
        airlinesNames = airlines
    pctList = []
    for airline in airlines:
        factor = len(P[(P["AC Operator"] == airline)  & (P["On Time"] == True)]) / len(P[(P["AC Operator"] == airline)])
        pctList.append(round(factor*100, 4))

    fig, ax = plt.subplots(1, 1)
    chart = ax.bar(airlinesNames, pctList)

    ax.bar_label(chart, label_type='edge', fmt="%d")

    fig.suptitle(f"Percentage of flights arriving within 15 minutes of schedule")
    ax.set_xlabel("Airline ICAO code")
    ax.set_ylabel("On Time %")
    ax.grid()

if __name__ == "__main__":
    start = datetime(2016, 1, 1)
    end = datetime(2016, 12, 31)
    airports = ICAOTOP25
    print(f"Testing {len(airports)} Airports.")
    # a = extractData(start, end, airports, europeanFlights=True)
    # a = delayCalc(a)
    # airportBoxPlot(a, airports)

    b = extractData(start, end, None, europeanFlights=True)
    b = delayCalc(b)
    airlines = ["KLM", "RYR", "AFR", "WZZ",
                "DLH", "BAW", "IBE", "AEE", "SAS", "AFL"]
    airlinesNames = ["KLM", "RyanAir", "AirFrance", "Wizz Air", "Lufthansa", "British Airways", "Iberia", "Aegan", "Scandinavian", "Aeroflot"]
    airlineOnTime(b, airlines, airlinesNames)



    plt.show()
