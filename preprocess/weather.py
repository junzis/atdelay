from datetime import datetime
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from preprocess.airportvalues import airport_dict
from glob import glob


def basic_data_reader(
    fileloc: str, data: str, max_scale: float = None, min_scale: float = None
):
    """Plots the weather data from grib file into an image that 'moves'

    Args:
        fileloc (str): file location of grib file
        data (str): which variable should be plotted, such as 'gust'
        max_scale (float, optional): maximum value for colour scale. Defaults to None.
        min_scale (float, optional): minimum value for colour scale. Defaults to None.

    Returns:
        [type]: [description]
    """
    plt.ion()
    if fileloc == "./data/Schiphol_Weather_Data.grib":
        ds = xr.open_dataset(
            fileloc,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            decode_coords=True,
        )
        weather_data = ds.variables[data].data
        day_counter = 0
        for day in weather_data:
            day_counter += 1
            hour_counter = -1
            for hour in day:
                hour_counter += 1
                plt.imshow(hour, cmap="hot", interpolation="nearest")
                plt.title(f"day {day_counter}, hour = {hour_counter}")
                plt.draw()
                plt.pause(0.0001)
                plt.clf()
    else:
        try:
            weather_data = np.loadtxt(fileloc)
        except OSError:
            return 0
        plt.imshow(
            weather_data,
            cmap="rainbow",
            interpolation="nearest",
            vmax=max_scale,
            vmin=min_scale,
        )
        plt.title(fileloc)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


def fetch_grb(year, month, day, hour, pred=0, plot_data: bool = False):
    """grabs weather data from internet and turns it into numpy arrays for each variable and saves them in corresponding folder.

    Args:
        year (int): year to grab data from
        month (int): month to grab data from
        day (int): day to grab data from
        hour (int): hour to grab data from (only data at 0, 6, 12 and 18)
        pred (int, optional): Fixed end of url. Only change from 0 if grabbing from different dataset. Defaults to 0.
        plot_data (bool, optional): Plots each variable after downloading data. Defaults to False.

    Returns:
        [type]: [description]
    """
    datadir = "./data/grib/"
    if not os.path.exists(datadir):
        os.mkdir(os.path.join(datadir))

    windgfs_url = "https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/"
    ym = "%04d%02d" % (year, month)
    ymd = "%04d%02d%02d" % (year, month, day)
    hm = "%02d00" % hour
    pred = "%03d" % pred

    fname = "gfsanl_3_%s_%s_%s.grb2" % (ymd, hm, pred)
    fpath = datadir + fname

    if not os.path.exists(
        f"./data/Weather_Data_Filtered/cape/{year}/cape_{year}_{month}_{day}_{hour}.npy"
    ):
        remote_loc = "/%s/%s/gfsanl_3_%s_%s_%s.grb2" % (ym, ymd, ymd, hm, pred)
        remote_url = windgfs_url + remote_loc
        # print("\n Downloading %s" % remote_url)
        response = requests.get(remote_url, stream=True)
        if response.status_code != 200:
            # print("Error. remote data not found")
            return None
        # else:
        #     print('Download succesfull!')

        with open(fpath, "wb") as f:
            # print('Writing to grib...')
            total_length = response.headers.get("content-length")

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(100 * dl / total_length)
                    # sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (100-done)) )
                    # sys.stdout.flush()

        ds = xr.open_dataset(
            fpath,
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}},
            decode_coords=True,
        )
        ds2 = ds.where(
            (ds.longitude < 60) & (ds.latitude > 30) & (ds.latitude < 70), drop=True
        )
        ds3 = ds.where(
            (ds.longitude > 350) & (ds.latitude > 30) & (ds.latitude < 70), drop=True
        )

        for weather_type in ["vis", "gust", "t", "cpofp", "lftx", "cape"]:
            weather_data_numpy = np.hstack(
                [ds3.variables[weather_type].data, ds2.variables[weather_type].data]
            )

            if not os.path.exists(f"./data/Weather_Data_Filtered/{weather_type}/"):
                os.makedirs(f"./data/Weather_Data_Filtered/{weather_type}/")
            if not os.path.exists(
                f"./data/Weather_Data_Filtered/{weather_type}/{year}/"
            ):
                os.makedirs(f"./data/Weather_Data_Filtered/{weather_type}/{year}/")
            np.savetxt(
                f"./data/Weather_Data_Filtered/{weather_type}/{year}/{weather_type}_{year}_{month}_{day}_{hour}.npy",
                weather_data_numpy,
                fmt="%d",
            )
            if plot_data:
                plt.imshow(weather_data_numpy, cmap="hot", interpolation="nearest")
                plt.show()
        os.remove(fpath)
        os.remove(fpath + ".923a8.idx")
    # else:
    # print('File already exists!')


def npy_to_df(year: int, interval: int):
    """grabs numpy files and saves a df for each airport with interpolated data.

    Args:
        year (int, optional): year to grab data from.
        interval (bool, optional): minute intervals to have data points at (should be between 60 - 1).
    """

    if 0 >= interval >= 61:
        raise ValueError("Interval value should be between 1 and 60 minutes")
    if 60 % interval != 0:
        raise ValueError("Interval should always contain each full hour")

    if not os.path.exists(
        f"./data/Weather_Data_Filtered/gust/{year}/gust_{year}_3_1_0.npy"
    ):
        if year == 2018 or year == 2019:
            raise FileNotFoundError(
                f"COULD NOT FIND {year} DATA! Make sure you have the 2019 and 2018 data downloaded correctly"
            )
        else:
            raise FileNotFoundError(
                f"COULD NOT FIND {year} DATA! If you want data for a year other than 2019 and 2018, run fetch_grb() in a for loop (see weather.py __main__). Warning: this function takes ~hour to run"
            )

    if not os.path.exists(
        f"./data/Weather_Data_Filtered/Airports/{interval}_interval/{year}/"
    ):
        os.makedirs(
            f"./data/Weather_Data_Filtered/Airports/{interval}_interval/{year}/"
        )

    minute_list = []
    overloaded = False
    minute = 0
    while not overloaded:
        minute_list.append(minute)
        minute += interval
        if minute >= 60:
            overloaded = True

    for airport in tqdm(airport_dict):
        long = int(airport_dict[airport]["longitude"])
        lat = int(airport_dict[airport]["latitude"])
        airport_data = {
            "time": [],
            "vis": [],
            "gust": [],
            "t": [],
            "cpofp": [],
            "lftx": [],
            "cape": [],
        }
        for month in [3, 6, 9, 12]:
            for day in range(1, 31):
                for hour in range(0, 24):
                    for minute in minute_list:
                        airport_data["time"].append(
                            datetime(year, month, day, hour, minute)
                        )
                        for variable in ["vis", "gust", "t", "cpofp", "lftx", "cape"]:
                            if hour in [0, 6, 12, 18] and minute == 0:
                                try:
                                    weather_array = np.loadtxt(
                                        f"./data/Weather_Data_Filtered/{variable}/{year}/{variable}_{year}_{month}_{day}_{hour}.npy"
                                    )
                                    airport_data[variable].append(
                                        weather_array[69 - lat, long + 9]
                                    )
                                except OSError:
                                    airport_data[variable].append(np.NaN)
                            else:
                                airport_data[variable].append(np.NaN)
        df = pd.DataFrame(airport_data)
        for variable in ["vis", "gust", "t", "cpofp", "lftx", "cape"]:
            df[[variable]] = df[[variable]].interpolate()

        pd.DataFrame((df)).to_csv(
            f"./data/Weather_Data_Filtered/Airports/{interval}_interval/{year}/{airport}_{year}_{interval}.csv",
            header=True,
            index=False,
        )


def fetch_weather_data(airport: str, interval: int, years: list = [2019, 2018]):
    """reads a weather data file, and if it does not exist, it will generate it and read it afterwards.

    Args:
        airport (str): airport code (FE: 'EBBR').
        interval (int): minute intervals to have data points at (should be between 60 - 1).
        years (list): list of years to have in dataframe. Defaults to [2019, 2018].

    Returns:
        pd.DataFrame: dataframe with weather data for years that were assigned
        Features are as follows:
        visibility         =  vis         =  Visibility [m]
        windspeed          =  gust        =  Wind speed [m/s]
        temperature        =  t           =  Temperature [K]
        frozenprecip       =  cpofp       =  Prob. of frozen precipitation [%]
        surfaceliftedindex =  lftx        =  Surface lifted index [K]
        (The lifted index (LI) is the temperature difference between the environment Te(p) and an air parcel lifted adiabatically Tp(p) at a given pressure height in the troposphere (lowest layer where most weather occurs) of the atmosphere, usually 500 hPa (mb). The temperature is measured in Celsius. When the value is positive, the atmosphere (at the respective height) is stable and when the value is negative, the atmosphere is unstable.)
        cape               =  cape        =  Convective available potential energy [J/kg]
        (In meteorology, convective available potential energy (commonly abbreviated as CAPE), is the integrated amount of work that the upward (positive) buoyancy force would perform on a given mass of air (called an air parcel) if it rose vertically through the entire atmosphere. At high values of cape (1000+)  there is a high probability of heavy thunderstorms)

    """

    dform = "%Y-%m-%d %H:%M:%S"

    if airport not in airport_dict:
        raise ValueError("INCORRECT AIRPORT REQUEST")
    if 0 >= interval >= 61:
        raise ValueError("Interval value should be between 1 and 60 minutes")
    if 60 % interval != 0:
        raise ValueError("Interval should always contain each full hour")

    for year in years:
        if not os.path.exists(
            f"./data/Weather_Data_Filtered/Airports/{interval}_interval/{year}/{airport}_{year}_{interval}.csv"
        ):
            print(f"generating {year} weather data for {interval} minute interval")
            npy_to_df(year, interval)

    listOfFiles = []  # list with location of airport files of all years
    for year in years:
        listOfFiles.extend(
            glob(
                f"./data/Weather_Data_Filtered/Airports/{interval}_interval/{year}/{airport}_*_{interval}.csv"
            )
        )
    final_df = pd.DataFrame()
    for fileloc in listOfFiles:
        df = pd.read_csv(fileloc, header=0)
        final_df = final_df.append(df, ignore_index=True)

    final_df = (
        final_df.assign(timeslot=lambda x: pd.to_datetime(x.time, format=dform))
        .drop("time", axis=1)
        .rename(
            columns={
                "vis": "visibility",
                "gust": "windspeed",
                "t": "temperature",
                "cpofp": "frozenprecip",
                "lftx": "surfaceliftedindex",
                "cape": "cape",
            }
        )
        .set_index("timeslot")
    )

    return final_df


if __name__ == "__main__":
    pass
