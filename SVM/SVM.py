import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


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

