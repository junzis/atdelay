import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import sys

sys.path.append(".")
from tools.tools_thing import double_cross_validation
from tools.tools_thing import parameter_search

models = {
    "KNearestNeighbor": KNeighborsRegressor(),
    "SVM": SVR(),
}


model_parameters = {
    "KNearestNeighbor": {
        'n_neighbors' : range(0, 100, 10), 'weights' : ["distance", "uniform"]
    },
    "SVM": {
        'C' : [0.001, 0.01, 0.1, 1, 10], 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    },
}


dform = "%Y-%m-%d %H:%M:%S"

X = pd.read_csv("./scaled_2016_encoded_data.csv", header= None).to_numpy()
Y = pd.read_csv("./y_2016_data.csv", header= None).to_numpy()
Y = Y.reshape((-1,))
# print("finding parameters for SVM")
# print(parameter_search(models["SVM"], model_parameters["SVM"], X, Y))
print("finding parameters for KNN")
print(parameter_search(models["KNearestNeighbor"], model_parameters["KNearestNeighbor"], X, Y))