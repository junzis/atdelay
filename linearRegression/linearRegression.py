from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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
from tools.tools_thing import filtering_data
from tools.tools_thing import parameter_search


def linear_regression_model(degree):
    X = np.genfromtxt("linearRegression/xdata.csv", delimiter=",")
    y = np.genfromtxt("linearRegression/ydata.csv", delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    scores = []
    for i in range(1, degree + 1):

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


def plot_errors(degree):
    """Plots errors vs. the number of degrees of each regressor
    """

    scores = linear_regression_model(degree)
    plt.plot([x for x in range(1, degree + 1)], scores)
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

