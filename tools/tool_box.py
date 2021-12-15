from re import L
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import get_scorer
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from numpy import radians, sin, arcsin, cos, sqrt


def filtering_data_onehot(
    filename: str = "LRData/LRDATA.csv",
    start: datetime = datetime(2018, 1, 1),
    end: datetime = datetime(2019, 12, 31),
    airport: str = "EGLL",
    airport_capacity: int = 88,
):
    """Takes all the data points in a filename for a given interval of time, encodes it using it get_dummies, and focuses prediction efforts on a single airport of choosing. 

    Args:
        filename (str, optional): Filename of the .csv file to extract data from. Defaults to "LRData/LRDATA.csv".
        start (datetime, optional): Starting point of the time interval. Defaults to datetime(2018, 1, 1).
        end (datetime, optional): Ending point of the time interval. Defaults to datetime(2019, 12, 31).
        airport (str, optional): Airport codename. Defaults to "EGLL" (Heathrow Airport).
        airport_capacity (int, optional): Capacity of airport per hour defined as maximum movements per hour possible. Defaults to 88.


    Returns:
        tuple: Array with all relevant features of the dataset and another array of target variables.
    """
    df = pd.read_csv(filename, header=0, index_col=0)

    dform = "%Y-%m-%d %H:%M:%S"
    df = df.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform)).assign(
        FiledAT=lambda x: pd.to_datetime(x.FiledAT, format=dform)
    )
    df_2 = data_filter_outliers(df)
    if airport != None:
        print(f"-------Selecting airport {airport}-------")
        df_capacity = capacity_calc(df_2, airport, airport_capacity)
        df_capacity = df_capacity.query("ADES == @airport")
    else:
        df_capacity = df_2
    df_time_distance = time_distance(df_capacity)
    df_3 = dummies_encode(df_time_distance, airport)
    X_final = scaler(df_3)
    y = df_capacity["ArrivalDelay"].to_numpy()

    pd.DataFrame((df_3)).to_csv("data/finaldf.csv", header=False, index=False)
    pd.DataFrame((X_final)).to_csv("data/xdata.csv", header=False, index=False)
    print("-------Regression model DataFrame to .csv: DONE-------")
    pd.DataFrame((y)).to_csv("data/ydata.csv", header=False, index=False)
    print("-------Regression model target variables to .csv: DONE-------")

    return X_final, y


def data_filter_outliers(
    P: pd.DataFrame,
    start: datetime = datetime(2018, 1, 1),
    end: datetime = datetime(2019, 12, 31),
):
    """Filtering outliers from the data, an outlier is determined to be either 90 min or more late, 30 min early or departing and arriving from/on the same airport

    Args:
        P (pd.DataFrame): pandas dataframe with all flight data
        start (datetime, optional): Starting point of the time interval. Defaults to datetime(2018, 1, 1).
        end (datetime, optional): Ending point of the time interval. Defaults to datetime(2019, 12, 31).

    Returns:
        pd.DataFrame: dataframe with outliers removed
    """

    P = P.query(
        "FiledOBT <= @end & FiledOBT >= @start & ArrivalDelay < 90 & ArrivalDelay > -30 & ADES != ADEP"
    )
    return P


def capacity_calc(P: pd.DataFrame, airport: str = "EGLL", airport_capacity: int = 88):
    """Calculating the part capacity of airport used per 15, dependent on number of arrivals and departures and adding this to the data

    Args:
        P (pd.DataFrame): dataframe with all flight data for certain airport
        airport (str, optional): airport for which capacity will be calculated. Defaults to "EGLL".
        airport_capacity (int, optional): Capacity of airport per hour defined as maximum take-offs per hour possible. Defaults to 88.

    Returns:
        pd.DataFrame: dataframe with capacity of airport at time of flight 
    """
    dform = "%Y-%m-%d %H:%M:%S"
    P = P.assign(FiledOBT=lambda x: pd.to_datetime(x.FiledOBT, format=dform))

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


def time_distance(P: pd.DataFrame):
    """Calculates the filed flight time of each flight and distance between 2 airports.

    Args:
        P (pd.DataFrame): DataFrame containing flights with filed arrival time and filed off block time.

    Returns:
        pd.DataFrame: Same dataframe but with a new column 'flight_time'.
    """
    P = P.assign(distance=lambda row: haversine(row), axis=1)
    P["flight_time"] = P.apply(
        lambda row: (row["FiledAT"] - row["FiledOBT"]).seconds / 60, axis=1
    )
    P = P.drop(["ADEPLong", "ADEPLat", "ADESLong", "ADESLat"], axis=1)

    return P


def haversine(P: pd.DataFrame):
    """Calculates the great-circle distance between two decimal
    coordinates using the Haversine formula and applies it to a dataframe.
    The formula was found on https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        P (pd.DataFrame): Dataframe of LRData

    Returns:
        pd.DataFrame: Dataframe of LRData with extra column 'distance'
    """
    coords_a, coords_b = (P["ADEPLong"], P["ADEPLat"]), (P["ADESLong"], P["ADESLat"])
    # Conversion to radians is necessary for the trigonometric functions
    phi_1, phi_2 = radians(coords_a[1]), radians(coords_b[1])
    lambda_1, lambda_2 = radians(coords_a[0]), radians(coords_b[0])

    return (
        2
        * 6371
        * arcsin(
            sqrt(
                sin((phi_2 - phi_1) / 2) ** 2
                + cos(phi_1) * cos(phi_2) * sin((lambda_2 - lambda_1) / 2) ** 2
            )
        )
    )


def dummies_encode(P: pd.DataFrame, airport: str = None):
    """Encoding the categorial features of the data. Similar to OneHotEncoder

    Args:
        P (pd.DataFrame): Pandas DataFrame with all  flight data
        airport ([type]): airport for which calculations will be done

    Returns:
        pd.DataFrame: Pandas DataFrame with all categorial features encoded
    """
    if airport == None:
        new_df = P.drop(["FiledOBT", "FiledAT", "ACType", "ArrivalDelay"], axis=1)
        new_df3 = pd.get_dummies(
            new_df, columns=["ADEP", "ADES", "ACOperator", "month", "weekday"]
        )

    else:
        new_df = P.drop(
            ["ADES", "FiledOBT", "FiledAT", "ACType", "ArrivalDelay"], axis=1,
        )
        new_df3 = pd.get_dummies(
            new_df, columns=["ADEP", "ACOperator", "month", "weekday"]
        )

    return new_df3


def scaler(P: pd.DataFrame):
    """Scaling the data

    Args:
        P (pd.DataFrame): pandas dataframe to be scaled

    Returns:
        pd.DataFrame: Scaled dataframe
    """
    encoded_array = P.to_numpy()
    scaler = MinMaxScaler()
    X_scaled_array = scaler.fit_transform(encoded_array)

    return X_scaled_array


def get_data(
    folderName: str = "data",
    fileName_x: str = "xdata.csv",
    fileName_y: str = "ydata.csv",
):
    """Extracting data from csv files

    Args:
        folderName (str, optional): Name of the folder which the data is in. Defaults to "data".
        fileName_x (str, optional): Name of the file containing the data of the features(X). Defaults to "xdata.csv".
        fileName_y (str, optional): name of the file containing the labels(y). Defaults to "ydata.csv".

    Returns:
        [type]: [description]
    """
    X = np.genfromtxt(f"{folderName}/{fileName_x}", delimiter=",")
    print(f"Data points saved under filename {fileName_x} ---- EXTRACTED.")
    y = np.genfromtxt(f"{folderName}/{fileName_y}", delimiter=",")
    print(f"Target variables saved under filename {fileName_y} ---- EXTRACTED.")

    return X, y


def parameter_search(
    model,
    parameters: dict,
    X_train: np.array,
    y_train: np.array,
    score_string: str,
    n_folds: int = 5,
):
    """Optimizes parameters of model using cross-validation.

    Args:
        model (sklearnModel): sklearn regression model
        parameters (dict): dictionary containing the to tune parameters
        X_train (np.array): training data
        y_train (np.array): training labels
        score_string (str, optional): defines the to use score function. Get strings from https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter. Defaults to "neg_mean_squared_error".
        n_folds (int, optional): number of folds to use for cros-validation. Defaults to 5.

    Returns:
        dict: optimal parameters for model
    """

    cv = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        model, parameters, cv=cv, n_jobs=-1, verbose=4, scoring=score_string,
    ).fit(X_train, y_train)

    print("grid search = ", grid_search)
    print("best params = ", grid_search.best_params_)
    print("refit time = ", grid_search.refit_time_)
    df = pd.DataFrame(grid_search.cv_results_)
    y_values = df.to_numpy()[:, -3]
    ymin = np.max(y_values)
    print("best score = ", ymin)

    if model == KNeighborsRegressor:
        plt.plot(parameters["n_neighbors"], y_values * -1)
        plt.scatter(grid_search.best_params_["n_neighbors"], ymin * -1, color="red")
        plt.xlabel("K")
        plt.ylabel("MSE")
        plt.show()

    filedOBT = pd.read_csv("./tools/finaldf.csv", header=0).to_numpy()[:, 1]
    best_parameters = grid_search.best_params_

    if type(model) == KNeighborsRegressor:
        best_model = KNeighborsRegressor(
            n_neighbors=best_parameters["n_neighbors"],
            weights=best_parameters["weights"],
        )  #'n_neighbors' : range(1, 100, 10), 'weights' : ["uniform"]

    X_train_2, X_test, y_train_2, y_test = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42
    )
    best_model.fit(X_train_2, y_train_2)
    prediction = grid_search.predict(X_test)

    return best_parameters, prediction, y_test


def plot(df: pd.DataFrame, x_name: str, y_name: str):
    """Plots predictions of a model vs. the real target values.

    Args:
        df (pd.DataFrame): DataFrame containing predictions of a model and real target values.
        x_name (str): Name of the column to be used on x-axis
        y_name (str): Name of the column to be used on y-axis
    """
    sns.set(style="darkgrid")
    g = sns.jointplot(
        x=x_name,
        y=y_name,
        data=df,
        kind="kde",
        xlim=[-30, 90],
        ylim=[-30, 90],
        fill=True,
    )
    sns.regplot(x=x_name, y=y_name, data=df, scatter=False, ax=g.ax_joint)
    plt.show()


if __name__ == "__main__":
    X, y = filtering_data_onehot(
        filename="LRData/LRDATA.csv",
        start=datetime(2018, 1, 1),
        end=datetime(2019, 12, 31),
        airport="LFPG",
        airport_capacity=120,
    )

