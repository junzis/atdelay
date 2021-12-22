import os
import sys
import numpy as np
import pandas as pd
import cfgrib
from cfgrib import xarray_to_grib
import xarray as xr
import matplotlib.pyplot as plt
import bluesky as bs
import requests
from tqdm import tqdm

def basic_data_reader(fileloc: str, data: str):
    ds = xr.open_dataset(fileloc, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}, decode_coords= True)
    weather_data = ds.variables[data].data

    day_counter = 0
    plt.ion()
    if fileloc == './data/Schiphol_Weather_Data.grib':
        for day in weather_data:
            day_counter += 1
            hour_counter = -1
            for hour in day:
                hour_counter += 1
                plt.imshow(hour, cmap='hot', interpolation='nearest')
                plt.title(f'day {day_counter}, hour = {hour_counter}')
                plt.draw()
                plt.pause(0.0001)
                plt.clf()
    else:
        print(weather_data)
        print(weather_data.shape)
        plt.imshow(weather_data, cmap='rainbow', interpolation='nearest')
        plt.title(fileloc)
        plt.draw()
        plt.pause(0.1)
        plt.clf()

def fetch_grb(year, month, day, hour, pred=0, plot_data : bool = False):
    datadir = './data/grib/'
    if not os.path.exists(datadir):
        print('COULD NOT FIND GRIB FOLDER')
        return -1

    windgfs_url="https://www.ncei.noaa.gov/data/global-forecast-system/access/historical/analysis/"
    ym = "%04d%02d" % (year, month)
    ymd = "%04d%02d%02d" % (year, month, day)
    hm = "%02d00" % hour
    pred = "%03d" % pred

    fname = "gfsanl_3_%s_%s_%s.grb2" % (ymd, hm, pred)
    fpath = datadir + fname
    
    if not os.path.exists(f'./data/Weather_Data_Filtered/cape/{year}/cape_{year}_{month}_{day}_{hour}.npy'):
        remote_loc = "/%s/%s/gfsanl_3_%s_%s_%s.grb2" % (ym, ymd, ymd, hm, pred)
        remote_url = windgfs_url + remote_loc
        print("\n Downloading %s" % remote_url)
        response = requests.get(remote_url, stream=True)
        if response.status_code != 200:
                    print("Error. remote data not found")
                    return None
        else:
            print('Download succesfull!')

        
        with open(fpath, "wb") as f:
            print('Writing to grib...')
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in tqdm(response.iter_content(chunk_size=4096)):
                    dl += len(data)
                    f.write(data)
                    done = int(100 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (100-done)) )
                    sys.stdout.flush()
        
        ds = xr.open_dataset(fpath, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}}, decode_coords= True)
        ds2 = ds.where((ds.longitude < 60) & (ds.latitude > 30) & (ds.latitude < 70), drop=True)
        ds3 = ds.where((ds.longitude > 350) & (ds.latitude > 30) & (ds.latitude < 70), drop=True)

        for weather_type in ['vis', 'gust', 't', 'cpofp', 'lftx', 'cape']:
            weather_data_numpy = np.hstack([ds3.variables[weather_type].data, ds2.variables[weather_type].data])

            if not os.path.exists(f'./data/Weather_Data_Filtered/{weather_type}/'):
                os.makedirs(f'./data/Weather_Data_Filtered/{weather_type}/')
            if not os.path.exists(f'./data/Weather_Data_Filtered/{weather_type}/{year}/'):
                os.makedirs(f'./data/Weather_Data_Filtered/{weather_type}/{year}/')
            np.savetxt(f'./data/Weather_Data_Filtered/{weather_type}/{year}/{weather_type}_{year}_{month}_{day}_{hour}.npy', weather_data_numpy, fmt='%d')
            if plot_data:
                plt.imshow(weather_data_numpy, cmap='hot', interpolation='nearest')
                plt.show()
        os.remove(fpath)
        os.remove(fpath + '.923a8.idx')
    else:
        print('File already exists!')
    
    
    

if __name__ == "__main__":
    for month in [3, 6, 9, 12]:
        for day in range(1, 31):
            for hour in [0, 6, 12, 18]:
                fetch_grb(2019, month, day, hour)

    # basic_data_reader('./data/Schiphol_Weather_Data.grib')

    # for day in range(1, 10):
    #     for hour in [0, 6, 12, 18]:
    #         print('hour = ', hour)
    #         if hour == 6 or hour == 0:
    #             if day < 10:
    #                 basic_data_reader(f'./data/grib/gfsanl_3_2019060{day}_0{hour}00_000.grb2', 'gust')
    #             else:
    #                 basic_data_reader(f'./data/grib/gfsanl_3_201906{day}_0{hour}00_000.grb2', 'gust')
    #         else:
    #             if day < 10:
    #                 basic_data_reader(f'./data/grib/gfsanl_3_2019060{day}_{hour}00_000.grb2', 'gust')
    #             else:
    #                 basic_data_reader(f'./data/grib/gfsanl_3_201906{day}_{hour}00_000.grb2', 'gust')
    pass
