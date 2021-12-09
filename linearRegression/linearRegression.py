from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sys
from datetime import datetime


import matplotlib.pyplot as plt

sys.path.append(".")
from tools.tools_thing import filtering_data

parameters = {"polynomial__degree": [x for x in range(10, 30)]}

pipe = Pipeline(
    steps=[("polynomial", PolynomialFeatures), ("Linearreg", LinearRegression)]
)

y, X = filtering_data("LRData/LRDATA.csv")


def plot_errors():
    """Plots errors vs. the number of degrees of each regressor
    """
    plt.plot(
        parameters["polynomial__degree"], grid_search.cv_results_["mean_test_score"]
    )
    plt.title("MSE vs degree of polynomial for Linear Regression")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.show()


def best_model():
    """Returns the parameters of the best model found using linear and polynomial regression

    Returns:
        int: The degree of the polynomial with the best score
    """
    return int(grid_search.best_params_["degree"])

