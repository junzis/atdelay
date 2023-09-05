# %%
import glob
import pandas as pd
import cfgrib
from pathlib import Path
from tqdm import tqdm
from openap import nav


# %%
df = pd.read_parquet("data/all_flights.parquet")

airport_names = (
    df.groupby("ap0")
    .agg({"flight_id": "count"})
    .sort_values(by="flight_id", ascending=False)
    .index.tolist()
)

airport = (
    nav._read_airport()
    .query("icao.isin(@airport_names)")
    .assign(lat_=lambda d: round(d.lat * 4) / 4)
    .assign(lon_=lambda d: round(d.lon * 4) / 4)
    .assign(latlon=lambda d: d.lat_.astype(str) + "_" + d.lon_.astype(str))
)

latlons = airport.latlon.tolist()


# %%

grib_files = sorted(glob.glob("data/era5/*.grib"))

for gf in tqdm(grib_files, ncols=60):
    stem = Path(gf).stem
    year, month, day = [int(x) for x in stem.split("-")]

    fout = f"data/weather/{stem}.parquet"

    if Path(fout).exists():
        continue

    # %%
    data = cfgrib.open_datasets(gf)

    # %%
    d0 = (
        data[0]
        .merge(data[2])
        .assign(longitude=lambda x: (x.longitude + 180) % 360 - 180)
        .assign(
            latlon=lambda x: x.latitude.astype(str).str.cat(
                "_", x.longitude.astype(str)
            )
        )
        .to_dataframe()
        .query(f"latlon.isin({latlons})")
        .reset_index()
    )

    # %%
    d1 = (
        data[1]
        .assign(longitude=lambda x: (x.longitude + 180) % 360 - 180)
        .assign(
            latlon=lambda x: x.latitude.astype(str).str.cat(
                "_", x.longitude.astype(str)
            )
        )
        .to_dataframe()
        .query(f"latlon.isin({latlons})")
        .reset_index()
    )

    # %%
    df = (
        (
            d0.eval("time=time+step")
            .eval("day=time.dt.day")
            .query("day==@day")[["time", "latlon", "fg10", "sf", "tp", "ptype"]]
        )
        .merge(d1.drop(["number", "step", "surface", "valid_time"], axis=1))
        .merge(airport[["latlon", "icao"]])
        .drop(["latlon"], axis=1)
    )

    # %%
    df.to_parquet(fout, index=False)
