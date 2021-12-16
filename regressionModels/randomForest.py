from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(".")
from regressionModels.tool_box import get_preprocessed_data, plot


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

