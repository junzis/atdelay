from re import L
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


# Define type of sklearn models for type hints
sklearnModel = np.Union[SVR, RandomForestRegressor, KNeighborsRegressor, LinearRegression]


def missing_values(filename: str):
    df = pd.read_csv(filename)
    df.dropna(inplace=True)
    return df


def data_encoding(df: pd.DataFrame):
    enc = OneHotEncoder()
    df.to_numpy()
    enc.fit_transform(df)
    return df


def data_scaling(array: np.array):        
    try:
        scaler = StandardScaler()
        scaler.fit_transform(array)
        return array
    except ValueError:
        print("Encode the dataframe first")


def parameter_search(model: sklearnModel, parameters: dict, X_train: np.array, y_train: np.array, score_string: str = "neg_mean_squared_error", n_folds: int=5):
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
        model, parameters, cv=cv, n_jobs=-1, verbose=4, scoring=score_string
    ).fit(X_train, y_train)

    best_parameters = grid_search.best_params_
    return best_parameters


def split_into_folds(X: np.array, y: np.array, n_folds: int):
    """Splits the data into n amount of folds.
       Used in double_cross_validation.

    Args:
        X (np.array): Training data
        y (np.array): labels corresponding to training data
        n_folds (int): amount of folds

    Returns:
        tuple(list, list): [description]
    """    
    interval = len(y) // n_folds
    return ([X[i * interval:(i + 1) * interval] for i in range(n_folds)],
            [y[i * interval:(i + 1) * interval] for i in range(n_folds)])


def double_cross_validation(model: sklearnModel, parameters: dict, X_train: np.array, y_train: np.array, score_string: str="neg_mean_squared_error", inner_folds: int=3,
                            outer_folds: int=10, print_performance: bool=True):
    """Tunes and evaluates the performance of a model using double (nested) cross-validation.

    Args:
        model (sklearnModel): sklearn model
        parameters (dict): dictionary with model name and dict as key with in that all hyper-parameters
        X_train (np.array): Training + test data
        y_train (np.array): labels corresponding to training data
        score_func (function): which score function should be used to calculate the score of the model
        inner_folds (int, optional): amount of inner folds used for nested cross validation. Defaults to 3.
        outer_folds (int, optional): amount of outer folds used for nested cross validation. Defaults to 10.
        print_performance (bool, optional): Choose if python should print best performing perameters for each model. Defaults to True.

    Returns:
        performance_score, st_dev, best_parameters [type]: [description]
    """    

    scores = []
    best_parameters = {}

    X_folds, y_folds = split_into_folds(X_train, y_train, n_folds=outer_folds)
    for i, (X_fold_test, y_fold_test) in enumerate(zip(X_folds, y_folds)):
        # Split the data up into test fold and training data
        X_fold_train = np.concatenate([X_folds[k] for k in range(outer_folds) if k != i])
        y_fold_train = np.concatenate([y_folds[k] for k in range(outer_folds) if k != i])

        # Perform inner cross validation to tune the model
        grid_search = GridSearchCV(model, parameters, cv=KFold(n_splits=inner_folds), shuffle=True, scoring=score_string, n_jobs=-1)
        grid_search.fit(X_fold_train, y_fold_train)
        best_model, hyperparameters = grid_search.best_estimator_, grid_search.best_params_

        # Evaluate model
        predictions = best_model.predict(X_fold_test)
        score_func = get_scorer(score_string)._score_func
        score = score_func(y_fold_test, predictions)
        scores.append(score)

        # If best_model performs better than any previously found model, we update best_parameters.
        # The if/else statements are necessary, because we need to be able to know whether a high outcome or a low
        # outcome of score_func is desirable. For accuracy_score, a high value is desired, while for log_loss a low
        # value is more ideal.
        if score_func([1, 2], [1, 2]) > score_func([1, 2], [2, 1]):
            if score == max(scores):
                best_parameters = hyperparameters
        else:
            if score == min(scores):
                best_parameters = hyperparameters

    performance_score = np.mean(scores)
    st_dev = np.std(scores)

    if print_performance:
        print(
            f'{score_func.__name__.replace("_", " ")}: {round(performance_score, 3)}, st_dev: {round(st_dev, 5)}')
        print(f'Tuned parameters: {best_parameters}')

    return best_parameters, performance_score, st_dev
