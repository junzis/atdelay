from keras.engine import input_layer
from keras.layers.core import Activation
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np


data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])
look_back = 3  # how many samples back in time you want to use for your prediction
data_gen = TimeseriesGenerator(data, targets,
                               length=look_back, sampling_rate=1, #sampling_rate: the step with which the inputs are sampled for predictions
                               batch_size=1, stride=1)  #stride: the step for the future samples we predict
                                                        #batch_size: how many batches we want to use for inputs, as well as the number of
                                                        #predicted future samples

# for i in range(len(data_gen)):
#     x, y = data_gen[i]
#     print('%s => %s' % (x, y))
#print((data_gen[0]))

batch_0 = data_gen[0]

x, y = batch_0

# x_input = np.array([1,2,3])
# # print(x_input.reshape((2,6)))
# print(x_input.shape)
# # n_input, n_features = 3, 1

# model = Sequential()
# # model.add(Embedding(input_dim=3, output_dim=64))
# # model.add(Input(shape=(n_input, n_features)))
# model.add(LSTM(100, activation='relu', input_shape=(4, 3)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# print(model.summary())

# #fit model
# model.fit(x, y, steps_per_epoch=1, epochs=100, verbose=0)
# yhat = model.predict(x_input, verbose=0)
# print(yhat)


x = np.arange(-2, 2, 0.01)
y = 3*x
# y = x**5 - x**4 + x**3 - x**2 - 2*x

train_size_ratio = 0.7

x_train = x[:int(len(x)*train_size_ratio)]
y_train = y[:int(len(y)*train_size_ratio)]
x_test = y[int(len(y)*train_size_ratio):]
y_test = x[int(len(x)*train_size_ratio):]


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=10,
    sampling_rate=1,
    batch_size=3,
)

# dataset_val = keras.preprocessing.timeseries_dataset_from_array(
#     x_test,
#     y_test,
#     sequence_length=10,
#     sampling_rate=1,
#     batch_size=3,
# )


# for batch in dataset_train.take(1):
#     print(batch)
#     inputs, targets = batch

# print('=====================================')
# print(inputs.shape)
# print('=====================================')


model = Sequential()
model.add(Dense(3, activation='relu', input_dim=1))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(1))


# inputs = keras.layers.Input(shape=(inputs.shape[0], inputs.shape[1]))
# lstm_out = keras.layers.LSTM(32)(inputs)
# outputs = keras.layers.Dense(1)(lstm_out)

# model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# path_checkpoint = "model_checkpoint.h5"
# es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

# modelckpt_callback = keras.callbacks.ModelCheckpoint(
#     monitor="val_loss",
#     filepath=path_checkpoint,
#     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
# )

model.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=100,
    validation_split=0.2
)

predictions = model.predict(x_test)

plt.plot(x_test, y_test, 'g')
plt.plot(x_test, predictions, 'r')
plt.show()

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(len(loss))
# plt.figure()
# plt.plot(epochs, loss, "b", label="Training loss")
# plt.plot(epochs, val_loss, "r", label="Validation loss")
# plt.show()