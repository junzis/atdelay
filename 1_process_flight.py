#%%
import os, sys
import glob
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


#%%
start_year = int(sys.argv[1])
end_year = int(sys.argv[2])

start = datetime(start_year, 1, 1)
end = datetime(end_year, 12, 31)


#%%
files = []
for year in range(start_year, end_year + 1):
    files.extend(sorted(glob.glob(f"data/{year}/*/Flights_2*.csv*")))

dfs = []

for file in tqdm(files, ncols=60, desc="months"):
    # read, filter and process csv
    df = pd.read_csv(file)

    # Datetime format
    date_format = "%d-%m-%Y %H:%M:%S"

    df = (
        df.rename(
            columns={
                "ICAO Flight Type": "scheduled",
                "FILED OFF BLOCK TIME": "fobt",
                "FILED ARRIVAL TIME": "fat",
                "ACTUAL OFF BLOCK TIME": "aobt",
                "ACTUAL ARRIVAL TIME": "aat",
                "STATFOR Market Segment": "market",
                "ADEP": "ap0",
                "ADES": "ap1",
                "ADEP Latitude": "lat0",
                "ADEP Longitude": "lon0",
                "ADES Latitude": "lat1",
                "ADES Longitude": "lon1",
                "AC Type": "typecode",
                "AC Operator": "operator",
                "ECTRL ID": "flight_id",
                "Actual Distance Flown (nm)": "distance_nm",
            }
        )
        .query("ap0 != ap1")
        .drop(["AC Registration", "Requested FL"], axis=1)
        .assign(fobt=lambda x: pd.to_datetime(x.fobt, format=date_format))
        .assign(fat=lambda x: pd.to_datetime(x.fat, format=date_format))
        .assign(aobt=lambda x: pd.to_datetime(x.aobt, format=date_format))
        .assign(aat=lambda x: pd.to_datetime(x.aat, format=date_format))
        .assign(
            delay_arrival=lambda x: (x.aat - x.fat).astype("timedelta64[m]"),
            delay_departure=lambda x: (x.aobt - x.fobt).astype("timedelta64[m]"),
        )
    )

    dfs.append(df)


df_all = (
    pd.concat(dfs, ignore_index=True)
    .sort_values(by=["flight_id"])
    .drop_duplicates("flight_id")
)

# flights within delay quantile [0.001, 0.999]
qtl = df_all[["delay_arrival", "delay_departure"]].quantile([0.001, 0.999])

df_all = df_all.query(
    f'{qtl["delay_arrival"][0.001]}<delay_arrival<{qtl["delay_arrival"][0.999]}'
).query(
    f'{qtl["delay_departure"][0.001]}<delay_departure<{qtl["delay_departure"][0.999]}'
)

#%%
df_all.to_parquet("data/all_flights.parquet", index=False)
