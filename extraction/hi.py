import pandas as pd
from datetime import datetime
from extract import *

def capacity_calc(P: pd.DataFrame, airport: str = "EGLL", airport_capacity: int = 80):
    # times = pd.DatetimeIndex(P.time)
    dform = "%Y-%m-%d %H:%M:%S"
    P = P.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))
    P = P.assign(FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform))
    print(P.dtypes)
    dep = P.query("ADEP == @airport")
    des = P.query("ADES == @airport")
    dep = dep.assign(Date=lambda x: x.FiledOBT.dt.date)
    des = des.assign(Date=lambda x: x.FiledAT.dt.date)

    dep = (
        dep.assign(Hour=lambda x: x.FiledOBT.dt.hour)
        .assign(Minutes=lambda x: x.FiledOBT.dt.minute // 15 * 15)
        .assign(Time=lambda x: x.FiledOBT)
    )
    des = (
        des.assign(Hour=lambda x: x.FiledAT.dt.hour)
        .assign(Minutes=lambda x: x.FiledAT.dt.minute // 15 * 15)
        .assign(Time=lambda x: x.FiledAT)
    )

    new_df = pd.concat([dep, des], axis=0)
    new_df.Time = new_df.Time.apply(
        lambda x: pd.datetime(x.year, x.month, x.day, x.hour, x.minute // 15 * 15, 0)
    )
    times = pd.DatetimeIndex(new_df.Time)
    K = new_df.groupby([times.date, times.hour, times.minute])["ADES"].count()
    cap_dict = K.to_dict()

    new_df["Time_tuple"] = list(zip(new_df.Date, new_df.Hour, new_df.Minutes))
    new_df["capacity"] = new_df["Time_tuple"].map(cap_dict) / airport_capacity / 4
    new_df = new_df.drop(["Time_tuple", "Date", "Hour", "Minutes", "Time"], axis=1)
    new_df = new_df.sort_values("FiledAT")

    return new_df

if __name__ == "__main__":
    a = readCSV("filteredData", "generalEHAM.csv")
    a = capacity_calc(a, "EHAM")
    dtthing = datetime(2019,1,1)
    a = a.query("`FiledOBT` >= @dtthing")
    saveToCSV(a, "generalEHAMCap", "filteredData")



    print(a)