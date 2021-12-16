from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(".")
from regressionModels.tool_box import filtering_data_onehot
from regressionModels.tool_box import parameter_search, get_data


def linear_regression_model(start_degree: int = 1, stop_degree: int = 2):
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    scores = []
    for i in range(start_degree, stop_degree):

        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X_train)
        poly.fit(X_poly, y_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        X_test_trans = poly.fit_transform((X_test))
        prediction = model.predict(X_test_trans)
        score = mean_squared_error(y_test, prediction)

        print(f"MSE for degree {i} = {score}")

        scores.append(score)
    return scores


def plot_errors(start_degree: int = 1, stop_degree: int = 2):
    """Plots errors vs. the number of degrees of each regressor
    """

    scores = linear_regression_model(start_degree, stop_degree)
    plt.plot([x for x in range(start_degree, stop_degree)], scores)
    plt.title("MSE vs degree of polynomial for Linear Regression")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.show()

