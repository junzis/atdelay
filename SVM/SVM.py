print("importing...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import seaborn as sns
import sys

print('importing from tool box...')
sys.path.append(".")
# from tools.tool_box import double_cross_validation
from tools.tool_box import parameter_search
from tools.tool_box import filtering_data_onehot
from tools.tool_box import plot


print('creating models...')
models = {
    "KNearestNeighbor": KNeighborsRegressor(),
    "SVM": SVR(),
}


model_parameters = {
    "KNearestNeighbor": {
        'n_neighbors' : range(1, 70, 2), 'weights' : ["uniform"]
    },
    "SVM": {
        'C' : [0.1, 1, 10, 100, 1000], 'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    },
}


dform = "%Y-%m-%d %H:%M:%S"


# print("finding parameters for SVM")
# print(parameter_search(models["SVM"], model_parameters["SVM"], X, Y))
# print("finding parameters for KNN...")

predictions = {}
filtering_data_onehot('./LRData/LRDATA.csv', datetime(2019, 3, 1), datetime(2019, 3, 2), 'EGLL')
print("reading data...")
X = pd.read_csv("./tools/xdata.csv", header= None).to_numpy()
Y = pd.read_csv("./tools/ydata.csv", header= None).to_numpy()
Y = Y.reshape((-1,))
print('average y = ', np.average(Y))
best_parameters, prediction, y_test_day = parameter_search(models["KNearestNeighbor"], model_parameters["KNearestNeighbor"], X, Y, 'neg_mean_squared_error')
predictions['day real'] = y_test_day
predictions['day prdct'] = prediction
predictions['day error'] = prediction - y_test_day

predictions_month = {}
filtering_data_onehot('./LRData/LRDATA.csv', datetime(2019, 3, 1), datetime(2019, 3, 31), 'EGLL')
X = pd.read_csv("./tools/xdata.csv", header= None).to_numpy()
Y = pd.read_csv("./tools/ydata.csv", header= None).to_numpy()
Y = Y.reshape((-1,))
best_parameters, prediction, y_test_month = parameter_search(models["KNearestNeighbor"], model_parameters["KNearestNeighbor"], X, Y, 'neg_mean_squared_error')
predictions_month['month real'] = y_test_month
predictions_month['month prdct'] = prediction
predictions_month['month error'] = prediction - y_test_month

predictions_df = pd.DataFrame.from_dict(predictions)
predictions_month_df = pd.DataFrame.from_dict(predictions_month)

plot(predictions_month_df, 'month real', 'month prdct')