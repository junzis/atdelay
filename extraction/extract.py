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
    start: datetime,
    end: datetime,
    airports: list = None,
    folderName: str = "data",
    europeanFlights: bool = False,
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
        timeCols = [
            "FILED OFF BLOCK TIME",
            "FILED ARRIVAL TIME",
            "ACTUAL OFF BLOCK TIME",
            "ACTUAL ARRIVAL TIME",
        ]
        P[timeCols] = P[timeCols].apply(pd.to_datetime, format="%d-%m-%Y %H:%M:%S")

        # P = P[(P["ADEP"].isin(airports)) | (P["ADES"].isin(airports))]
        if airports is not None:
            P = P[(P["ADEP"].isin(airports))]

        if europeanFlights:
            P = P[
                (P["ADEP"].str[0].isin(["E", "B", "L", "U"]))
                & (P["ADES"].str[0].isin(["E", "B", "L", "U"]))
            ]

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
    P["flightTimeDifference"] = (
        P["ACTUAL ARRIVAL TIME"] - P["ACTUAL OFF BLOCK TIME"]
    ) - (P["FILED ARRIVAL TIME"] - P["FILED OFF BLOCK TIME"])

    P["arrivalDelayMinutes"] = P["arrivalDelay"].astype("timedelta64[m]")
    P["blockoffDelayMinutes"] = P["blockoffDelay"].astype("timedelta64[m]")
    P["flightTimeDelayMinutes"] = P["flightTimeDifference"].astype("timedelta64[m]")

    P["On Time"] = (P["arrivalDelayMinutes"]).abs() < 15.0

    return P


def airportBoxPlot(P, airports, delayType="blockoffDelayMinutes", showfliers=False):
    lstofData = []

    fig, ax = plt.subplots(1, 1)

    for airport in airports:
        lstofData.append(P[P["ADEP"] == airport][delayType].to_numpy())

    ax.boxplot(lstofData, showfliers=showfliers, labels=airports)
    fig.suptitle(f"Delay in departure in minutes from departing airports ({delayType})")
    ax.set_xlabel("Airport ICAO code")
    ax.set_ylabel("Delay in Minutes")
    ax.grid()

def segmentBoxPlot(P, showfliers=False):
    lstofData = []

    categories = ['Business Aviation', 'All-Cargo', 'Traditional Scheduled', 'Lowcost', 'Charter']

    fig, ax = plt.subplots(1, 1)

    for category in categories:
        lstofData.append(P[P["STATFOR Market Segment"] == category]["blockoffDelayMinutes"].to_numpy())
    lstofData.append(P[b["ICAO Flight Type"] == "S"]["blockoffDelayMinutes"].to_numpy())
    lstofData.append(P[b["ICAO Flight Type"] == "N"]["blockoffDelayMinutes"].to_numpy())
    categories.append("S")
    categories.append("N")

    ax.boxplot(lstofData, showfliers=showfliers, labels=categories)
    fig.suptitle(f"Delay in departure in minutes from departing airports")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Delay in Minutes")
    ax.grid()


def airlineOnTime(P, airlines, airlinesNames=None):
    if airlinesNames is None:
        airlinesNames = airlines
    pctList = []
    for airline in airlines:
        factor = len(P[(P["AC Operator"] == airline) & (P["On Time"] == True)]) / len(
            P[(P["AC Operator"] == airline)]
        )
        pctList.append(round(factor * 100, 4))

    fig, ax = plt.subplots(1, 1)
    chart = ax.bar(airlinesNames, pctList)

    ax.bar_label(chart, label_type="edge", fmt="%d")

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
    # airportBoxPlot(a, airports, showfliers=True)

    b = extractData(start, end, None, europeanFlights=True)
    b = delayCalc(b)

    segmentBoxPlot(b, False)
    # print(b[b["ICAO Flight Type"] == "N"]["arrivalDelayMinutes"].abs().mean())
    # print(b[b["ICAO Flight Type"] == "S"]["arrivalDelayMinutes"].abs().mean())
    # print(b["STATFOR Market Segment"].unique()) # ['Business Aviation' 'All-Cargo' 'Traditional Scheduled' 'Lowcost' 'Charter']
    total = len(b)
    print(f"Total N = {total}")
    print('Business Aviation =', round(len(b[b["STATFOR Market Segment"] == 'Business Aviation'])/total*100,2), "%")
    print('All-Cargo =', round(len(b[b["STATFOR Market Segment"] == 'All-Cargo'])/total*100,2), "%")
    print('Traditional Scheduled =', round(len(b[b["STATFOR Market Segment"] == 'Traditional Scheduled'])/total*100,2), "%")
    print('Lowcost =', round(len(b[b["STATFOR Market Segment"] == 'Lowcost'])/total*100,2), "%")
    print('Charter =', round(len(b[b["STATFOR Market Segment"] == 'Charter'])/total*100,2), "%")

    # print('Business Aviation AVG DELAY=', (b[b["STATFOR Market Segment"] == 'Business Aviation']["arrivalDelayMinutes"].abs().mean()))
    # print('All-Cargo AVG DELAY=', (b[b["STATFOR Market Segment"] == 'All-Cargo']["arrivalDelayMinutes"].abs().mean()))
    # print('Traditional Scheduled AVG DELAY=', (b[b["STATFOR Market Segment"] == 'Traditional Scheduled']["arrivalDelayMinutes"].abs().mean()))
    # print('Lowcost AVG DELAY=', (b[b["STATFOR Market Segment"] == 'Lowcost']["arrivalDelayMinutes"].abs().mean()))
    # print('Charter AVG DELAY=', (b[b["STATFOR Market Segment"] == 'Charter']["arrivalDelayMinutes"].abs().mean()))

    # print('Business Aviation MAX DELAY=', (b[b["STATFOR Market Segment"] == 'Business Aviation']["arrivalDelayMinutes"].abs().max()))
    # print('All-Cargo MAX DELAY=', (b[b["STATFOR Market Segment"] == 'All-Cargo']["arrivalDelayMinutes"].abs().max()))
    # print('Traditional Scheduled MAX DELAY=', (b[b["STATFOR Market Segment"] == 'Traditional Scheduled']["arrivalDelayMinutes"].abs().max()))
    # print('Lowcost MAX DELAY=', (b[b["STATFOR Market Segment"] == 'Lowcost']["arrivalDelayMinutes"].abs().max()))
    # print('Charter MAX DELAY=', (b[b["STATFOR Market Segment"] == 'Charter']["arrivalDelayMinutes"].abs().max()))
    # airlines = ["KLM", "RYR", "AFR", "WZZ", "DLH", "BAW", "IBE", "AEE", "SAS", "AFL"]
    # airlinesNames = [
    #     "KLM",
    #     "RyanAir",
    #     "AirFrance",
    #     "Wizz Air",
    #     "Lufthansa",
    #     "British Airways",
    #     "Iberia",
    #     "Aegan",
    #     "Scandinavian",
    #     "Aeroflot",
    # ]
    # airlineOnTime(b, airlines, airlinesNames)

    plt.show()
