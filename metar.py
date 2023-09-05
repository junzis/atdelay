#%%
import numpy as np
import pandas as pd
from mivek import metar
from mivek.exceptions import InvalidWindException, InvalidVisibilityException


def decode_metar(metar_list):
    """
    Decode a list of METAR codes and return a pandas DataFrame containing the different components.

    Args:
        metar_list (list): A list of strings representing METAR codes.

    Returns:
        pandas.DataFrame: A DataFrame containing the different components of the METAR codes.
    """
    # Create a dictionary to store the decoded components
    metar_dict = {
        "icao": [],
        "datetime": [],
        "wind_dir": [],
        "wind_speed": [],
        "visibility": [],
        "cloud_cover": [],
        "temperature": [],
        "dew_point": [],
        "altimeter": [],
    }

    # Loop through each METAR code in the input list and decode it
    for metar_code in metar_list:
        try:
            obs = metar.Metar(metar_code)

            # Extract the ICAO code and date/time
            icao_code = obs.station
            datetime_str = obs.time.ctime()
            metar_dict["icao"].append(icao_code)
            metar_dict["datetime"].append(datetime_str)

            # Extract the wind direction and speed
            wind_dir = 0 if obs.wind_direction is None else obs.wind_direction.value()
            wind_speed = 0 if obs.wind_speed is None else obs.wind_speed.value("KT")
            metar_dict["wind_dir"].append(wind_dir)
            metar_dict["wind_speed"].append(wind_speed)

            # Extract the visibility
            visibility = np.nan if obs.visibility is None else obs.visibility.value("m")
            metar_dict["visibility"].append(visibility)

            # Extract the cloud cover
            cloud_layers = []
            for layer in obs.clouds:
                cloud_dict = {}
                if layer.altitude is None:
                    # If the altitude is None, the cloud layer is at the surface
                    cloud_dict["type"] = layer.type
                    cloud_dict["altitude"] = 0
                else:
                    cloud_dict["type"] = layer.type
                    cloud_dict["altitude"] = layer.altitude.value("ft") * 0.3048
                cloud_layers.append(cloud_dict)
            metar_dict["cloud_cover"].append(cloud_layers)

            # Extract the temperature and dew point
            temperature = (
                np.nan if obs.temperature is None else obs.temperature.value("C")
            )
            dew_point = np.nan if obs.dew_point is None else obs.dew_point.value("C")
            metar_dict["temperature"].append(temperature)
            metar_dict["dew_point"].append(dew_point)

            # Extract the altimeter setting
            altimeter = np.nan if obs.altimeter is None else obs.altimeter.value("hPa")
            metar_dict["altimeter"].append(altimeter)

        except (InvalidWindException, InvalidVisibilityException):
            # Skip METAR codes with invalid wind or visibility components
            pass

    # Convert the dictionary to a pandas DataFrame and return it
    return pd.DataFrame.from_dict(metar_dict)


metar_list = [
    "KORD 051751Z 00000KT 10SM FEW200 SCT250 09/M02 A3026 RMK AO2 SLP250 T00941017 10094 21006 50003",
    "KLAX 051753Z VRB03KT 10SM FEW020 SCT035 BKN050 21/10 A3002 RMK AO2 SLP166 T02060100 10211 20167 58007",
    "KJFK 051751Z 36006KT 10SM SCT150 SCT250 06/M07 A3037 RMK AO2 SLP288 T00611072 10061 21006 50004",
    "KDEN 051753Z 00000KT 10SM FEW220 11/M02 A3033 RMK AO2 SLP223 T01061022 10106 20056 58009",
    "KORD 051801Z 00000KT 10SM FEW200 SCT250 10/M02 A3025 RMK AO2 T01001017",
    "KLAX 051758Z VRB03KT 10SM FEW020 SCT035 BKN050 21/10 A3002 RMK AO2 SLP166 T02110100 10211 20167 58007",
]

decoded_metar = decode_metar(metar_list)
print(decoded_metar)

#%%
import requests
import pandas as pd
from datetime import datetime, timedelta


def get_metar_data(airport_list, start_date, end_date):
    """
    Retrieve METAR data from the aviationweather.gov API for a list of airports and a range of dates.

    Args:
        airport_list (list): A list of strings representing airport codes.
        start_date (str): A string representing the start date in YYYY-MM-DD format.
        end_date (str): A string representing the end date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: A DataFrame containing the decoded components of the METAR codes.
    """
    # Initialize an empty DataFrame to store the METAR data
    metar_df = pd.DataFrame()

    # Loop through each date in the date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    while start_dt <= end_dt:
        date_str = start_dt.strftime("%Y/%m/%d")

        # Loop through each airport code and retrieve the METAR data
        for airport_code in airport_list:
            url = f"https://aviationweather.gov/metar/data?ids={airport_code}&format=raw&date={date_str}&hours=00&mostRecent=true"
            response = requests.get(url)
            metar_list = response.text.split("\n")

            # Decode the METAR data and add it to the DataFrame
            # metar_df = pd.concat([metar_df, decode_metar_vectorized(metar_list)])
            metar_df = decode_metar_vectorized(metar_list)
            print(metar_df)

        # Increment the date by one day
        start_dt += delta

    return metar_df


airport_list = ["KORD", "KLAX", "KJFK", "KDEN"]
start_date = "2022-01-01"
end_date = "2022-01-03"

metar_df = get_metar_data(airport_list, start_date, end_date)
print(metar_df)
