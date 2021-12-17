
# from extraction.extract import generateNNdata

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Input data
np.random.seed(42)
# dataframe = generateNNdata("EHAM", numRunways=6, numGates=223)
dataframe = pd.read_csv('testNN.csv')
# dataframe = (
#     dataframe.assign(timeslot=lambda x: pd.to_datetime(x.timeslot, format=dform))
# )


Y = dataframe.loc[:,["DepartureDelay"]]  # Y = dataframe.loc[:,["ArrivalDelay", "DepartureDelay"]]
X = dataframe.drop(["ArrivalDelay", "DepartureDelay"], axis=1) 

X = X.values

X, timeslots = X[:, 1:].astype('float32'), X[:, 0]



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

print(X[:10], timeslots[:10])

# Split data into train and test

train_ratio = 0.7

train_size = int(len(X) * train_ratio)
test_size = len(X) - train_size
x_train, x_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = Y[0:train_size,:], Y[train_size:len(X),:]


# ===============================================================
#            Doesn't work for us right now
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY), a



X = np.array([[1,2,3],
    [4,5,6],
    [7,8,9],
    [9,8,7],
	[6,5,4],
	[3,2,1]])

dataX, dataY, a = create_dataset(X)
print('X:')
print(X)
print('dataX:')
print(dataX)
print('dataY:')
print(dataY)
print("a: ")
print(a)
# =============================================================

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(x_train, look_back)
testX, testY = create_dataset(x_test, look_back)

