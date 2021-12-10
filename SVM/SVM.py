print("importing...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import sys

print('importing from tool box...')
sys.path.append(".")
# from tools.tool_box import double_cross_validation
from tools.tool_box import parameter_search

print('creating models...')
models = {
    "KNearestNeighbor": KNeighborsRegressor(),
    "SVM": SVR(),
}


model_parameters = {
    "KNearestNeighbor": {
        'n_neighbors' : range(1, 100, 10), 'weights' : ["uniform"]
    },
    "SVM": {
        'C' : [0.1, 1, 10, 100, 1000], 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    },
}


dform = "%Y-%m-%d %H:%M:%S"

print("reading data...")
X = pd.read_csv("./tools/xdata.csv", header= None).to_numpy()
Y = pd.read_csv("./tools/ydata.csv", header= None).to_numpy()
Y = Y.reshape((-1,))
print('average y = ', np.average(Y))
# print("finding parameters for SVM")
# print(parameter_search(models["SVM"], model_parameters["SVM"], X, Y))
print("finding parameters for KNN...")
print(parameter_search(models["KNearestNeighbor"], model_parameters["KNearestNeighbor"], X, Y, 'neg_mean_absolute_error'))