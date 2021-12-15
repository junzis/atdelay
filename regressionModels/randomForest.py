from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(".")
from regressionModels.tool_box import get_data, plot


def grid_search_forest():
    """ Performs grid search on a RandomForest estimator. DO NOT USE.

    Returns:
        dict: A dictionary containing results after GridSearch.
    """
    parameters = {
        "n_estimators": [100, 300, 600],
        "max_features": [None, "log2", "auto"],
        "max_depth": [50, 250, 500],
        "min_samples_split": [2, 5, 10, 20, 50, 100, 1000],
        "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100, 1000],
        "bootstrap": [True, False],
    }

    regr = RandomForestRegressor(n_jobs=-1)

    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        regr, parameters, cv=cv, n_jobs=-1, verbose=4, scoring="neg_mean_absolute_error"
    ).fit(X_train, y_train)

    return grid_search.cv_results_


def top50_calculator():
    airport_dict = {
        "LEAL": 36,
        "EDDS": 48,
        "LFLL": 48,
        "ESSA": 84,
        "EGPH": 42,
        "UKBB": 36,
        "EIDW": 48,
        "LIME": 26,
        "EDDK": 36,
        "LFMN": 50,
        "EGBB": 40,
        "LPPR": 24,
    }
    acc_dict = {}
    for airport in airport_dict:
        print(f"Doing calculations for aiport: {airport}")
        X, y = filtering_data_onehot(
            filename="LRData/LRDATA.csv",
            start=datetime(2018, 1, 1),
            end=datetime(2019, 12, 31),
            airport=airport,
            airport_capacity=airport_dict[airport],
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        forest = RandomForestRegressor(
            n_estimators=300,
            max_features="auto",
            max_depth=50,
            min_samples_split=10,
            min_samples_leaf=2,
            bootstrap=True,
            n_jobs=-1,
        )
        forest.fit(X_train, y_train)
        prediction = forest.predict(X_test)
        score = mean_absolute_error(y_test, prediction)
        print(f"Accuracy of {airport} = {score}")
        acc_dict[airport] = score
        print(f"accuracy dictionary until now = {acc_dict}")

    print(acc_dict)
    overall_error = sum(acc_dict.values()) / len(acc_dict)
    return acc_dict, overall_error 

