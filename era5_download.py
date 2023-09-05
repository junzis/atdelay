# %%
import os
import cdsapi
import pandas as pd

# %%
c = cdsapi.Client()


# %%
dates = []
for year in [2016, 2017]:
    for month in [3, 6, 9, 12]:
        month_start = pd.Timestamp(year, month, 1)
        month_end = month_start + pd.offsets.MonthEnd()
        dates.append(pd.date_range(month_start, month_end).to_series())

dates = pd.concat(dates)

# %%
for date in dates:
    print(date)

    year = date.year
    month = date.month
    day = date.day

    file_out = f"data/era5/{year}-{month:02d}-{day:02d}.grib"

    if os.path.exists(file_out):
        continue

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": [
                "100m_u_component_of_wind",
                "100m_v_component_of_wind",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "10m_wind_gust_since_previous_post_processing",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "precipitation_type",
                "snowfall",
                "total_column_cloud_ice_water",
                "total_column_cloud_liquid_water",
                "total_column_rain_water",
                "total_column_snow_water",
                "total_precipitation",
            ],
            "year": f"{year}",
            "month": f"{month:02d}",
            "day": f"{day:02d}",
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
        },
        file_out,
    )
