from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def parameter_search(start: int = 1, stop: int = 10):
    
    parameters = {"polynomial__degree": [x for x in range(start, stop)]}

    pipe = Pipeline(
        steps=[("polynomial", PolynomialFeatures), ("Linearreg", LinearRegression)]
    )
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    grid_search = GridSearchCV(
        pipe, parameters, cv=cv, n_jobs=-1, verbose=4, scoring="neg_mean_squared_error"
    ).fit(X_train, y_train)

    print(grid_search.cv_results_)


def plot_errors():
    plt.plot(
        parameters["polynomial__degree"], grid_search.cv_results_["mean_test_score"]
    )
    plt.title("MSE vs degree of polynomial for Linear Regression")
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE")
    plt.show()


def best_model():
    return grid_search.best_params_["degree"]

