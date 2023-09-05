# %%
import os, sys
import argparse
import glob
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


# %%
parser = argparse.ArgumentParser()
parser.add_argument("start", type=int, help="start year")
parser.add_argument("end", type=int, help="end year")
parser.add_argument("-n", "--n_airports", type=int)
parser.add_argument("-o", "--overwrite", action="store_true", default=False)
args = parser.parse_args()

start_year = args.start
end_year = args.end
overwrite = args.overwrite
n_airports = args.n_airports

# start_year, end_year = 2018, 2019
# n_airports = 50
# overwrite = True

start = datetime(start_year, 1, 1)
end = datetime(end_year, 12, 31)


agg_interval = 30  # minutes


# %%
def gen_timeslot_list(start: datetime, end: datetime, interval: int):
    results = []
    delta = timedelta(minutes=interval)
    now = start

    while now <= end:
        if now.month in [3, 6, 9, 12]:
            results.append(now)
        now += delta

    return results


# %%
flights = pd.read_parquet("data/all_flights.parquet")

weather = pd.concat([pd.read_parquet(f) for f in glob.glob("data/weather/*.parquet")])


# %%
top_airports = (
    flights.query("fobt>='2018-01-01'")
    .groupby("ap0")
    .agg({"flight_id": "count"})
    .sort_values(by="flight_id", ascending=False)
    .head(n_airports)
    .index.tolist()
)


# %%
flights = (
    flights.query("ap0.isin(@top_airports) or ap1.isin(@top_airports)")
    .query("`fobt` >= @start & `fat` <= @end")
    .assign(duration_planned=lambda x: (x.fat - x.fobt).dt.total_seconds() // 60)
    .assign(timeslot=lambda x: x.fobt.dt.floor(f"{agg_interval}T"))
)

# %%

weather = (
    weather.query("icao.isin(@top_airports)")
    .query("`time` >= @start & `time` <= @end")
    .rename(columns={"icao": "airport"})
    .sort_values("time")
)

# %%
timeslots = gen_timeslot_list(start, end, agg_interval)


# %%

for airport in tqdm(top_airports, ncols=60, desc="airports"):
    fout_airport = f"data/airport/{airport}.parquet"

    if os.path.exists(fout_airport) and not overwrite:
        continue

    departures = (
        flights.query("ap0==@airport")
        .groupby("timeslot")
        .agg(
            {
                "flight_id": "count",
                "duration_planned": "mean",
                "delay_departure": "mean",
            }
        )
        # ensure no timeslot gaps
        .reindex(timeslots, fill_value=0)
        .rename(
            columns={
                "flight_id": "departures",
                "duration_planned": "departure_duration",
            }
        )
        .reset_index()
    )

    arrivals = (
        flights.query("ap1==@airport")
        .groupby("timeslot")
        .agg(
            {
                "flight_id": "count",
                "duration_planned": "mean",
                "delay_arrival": "mean",
            }
        )
        # ensure no timeslot gaps
        .reindex(timeslots, fill_value=0)
        .rename(
            columns={"flight_id": "arrivals", "duration_planned": "arrival_duration"}
        )
        .reset_index()
    )

    stats = (
        pd.merge(arrivals, departures)
        .eval("capacity = arrivals + departures")
        .eval("buffer = arrivals - departures")
        .eval("weekday = timeslot.dt.weekday")
        .eval("month = timeslot.dt.month")
        .eval("hour = timeslot.dt.hour")
        .assign(date=lambda x: pd.to_datetime(x.timeslot.dt.date))
    )

    stats = pd.merge_asof(
        stats,
        weather.query("airport==@airport"),
        left_on="timeslot",
        right_on="time",
    ).drop(["airport", "time"], axis=1)

    stats.to_parquet(fout_airport, index=False)
