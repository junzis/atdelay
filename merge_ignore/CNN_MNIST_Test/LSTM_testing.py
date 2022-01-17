import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#-------------------------------------
#  Weather forecasting implementation
#-------------------------------------

# Data Preprocessing
    # Hyperparameters
split_fraction = 0.7
train_split = int(split_fraction * int(df.shape[0]))
step = 4
past =  # number of timesteps used for the training (train)
future =  # number of timesteps used for the predictions (test)
learning_rate = 0.001
batch_size = 200
epochs = 10

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

train_data = df.loc[0 : train_split - 1]
val_data = df.loc[train_split:]



# Training dataset

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(len(df.columns)-2)]].values
y_train = df.iloc[start:end][[1]]

sequence_length = int(past / step)


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


# Validation dataset

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(len(df.columns)-2)]].values   
y_val = df.iloc[label_start:][[1]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
