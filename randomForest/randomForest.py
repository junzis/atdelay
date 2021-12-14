from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append(".")
from tools.tool_box import get_data, plot


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


if __name__ == "__main__":
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    predictions = {}
    forest = RandomForestRegressor(
        n_estimators=300,
        max_features="auto",
        max_depth=50,
        min_samples_split=10,
        min_samples_leaf=2,
        bootstrap=True,
        n_jobs=-1
    )
    forest.fit(X_train, y_train)
    prediction = forest.predict(X_test)

    predictions["real"] = y_test
    predictions["predicted"] = prediction
    predictions["errors"] = prediction - y_test
    score = mean_absolute_error(y_test, prediction)
    print(f"The mean absolute error = {score}")
    predictions_df = pd.DataFrame.from_dict(predictions)
    plot(predictions_df, "real", "predicted")

