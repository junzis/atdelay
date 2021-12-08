import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def split_into_folds(X: np.array, y: np.array, n_folds: int):
    """splits the data into n amount of folds

    Args:
        X (np.array): Training + test data
        y (np.array): labels corresponding to training data
        n_folds (int): amount of folds

    Returns:
        tuple(list, list): [description]
    """    
    interval = len(y) // n_folds
    return ([X[i * interval:(i + 1) * interval] for i in range(n_folds)],
            [y[i * interval:(i + 1) * interval] for i in range(n_folds)])


def double_cross_validation(model, parameters: dict, X_train: np.array, y_train: np.array, score_func, inner_folds: int=3,
                            outer_folds: int=10, print_performance: bool=True):
    """[summary]

    Args:
        model ([type]): [description]
        parameters (dict): [description]
        X_train (np.array): [description]
        y_train (np.array): [description]
        score_func ([type]): [description]
        inner_folds (int, optional): [description]. Defaults to 3.
        outer_folds (int, optional): [description]. Defaults to 10.
        print_performance (bool, optional): [description]. Defaults to True.

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
        clf = GridSearchCV(model, parameters, cv=KFold(n_splits=inner_folds))
        clf.fit(X_fold_train, y_fold_train)
        best_model, hyperparameters = clf.best_estimator_, clf.best_params_

        # Evaluate model
        predictions = best_model.predict(X_fold_test)
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
            f'{score_func.__name__.replace("_", " ")}: {round(performance_score, 3)} \u00B1 {round(st_dev * 100, 2)}%')
        print(f'Tuned parameters: {best_parameters}')

    return performance_score, st_dev, best_parameters


models = {
    "KNearestNeighbor": KNeighborsRegressor(),
    "SVM": SVR(random_state=42),
}


model_parameters = {
    "KNearestNeighbor": {
        'n_neighbors' : range(50, 100), 'weights' : ["distance", "uniform"]
    },
    "SVM": {
        'C' : [0.001, 0.01, 0.1, 1, 10], 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    },
}

