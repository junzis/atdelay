# from extraction.extract import generateNNdataMultiple
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
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np



#======================================================
#                  DATA PREPARATION         
#======================================================

dataframe = pd.read_csv('EHAM_15m.csv')



X, y = dataframe['planes'], dataframe['departuresDepartureDelay']
X = X.to_numpy()
y = y.to_numpy()
y = np.ravel(y)

X = X.reshape((-1, 1))

split_percent = 0.5
split = int(split_percent*len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]


look_back = 3  # how many samples back in time you want to use for your prediction
train_dataset = TimeseriesGenerator(X_train, X_train,
                               length=look_back, sampling_rate=1, #sampling_rate: the step with which the inputs are sampled for predictions
                               batch_size=10, stride=1)  #stride: the step for the future samples we predict
                                                        #batch_size: how many batches we want to use for inputs, as well as the number of
                                                        #predicted future samples

test_dataset = TimeseriesGenerator(X_test, X_test,
                               length=look_back, sampling_rate=1, #sampling_rate: the step with which the inputs are sampled for predictions
                               batch_size=10, stride=1)



#======================================================
#               DEFINE AND FIT MODEL          
#======================================================


# model = Sequential()

# model.add(LSTM(10, activation="relu", input_shape=(look_back, 1)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')


# model.fit(
#     train_dataset,
#     epochs=5,
#     verbose=1
#     )

# model.save('LSTM.h5')

model = keras.models.load_model('LSTM.h5')

#======================================================
#        EVALUATE MODEL AND VISUALIZE PREDICTIONS
#======================================================

predictions = model.predict(test_dataset)

print(predictions.shape)
print(test_dataset[0])

plt.plot(test_dataset[0], predictions, 'r')
plt.plot(X_test, y_test, 'g')
plt.show()

























# model.add(Dense(3, activation='relu', input_dim=1))

# x_train = x[:int(len(x)*train_size_ratio)]
# y_train = y[:int(len(y)*train_size_ratio)]
# x_test = y[int(len(y)*train_size_ratio):]
# y_test = x[int(len(x)*train_size_ratio):]


# dataset_train = keras.preprocessing.timeseries_dataset_from_array(
#     x_train,
#     y_train,
#     sequence_length=10,
#     sampling_rate=1,
#     batch_size=3,
# )

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




# inputs = keras.layers.Input(shape=(inputs.shape[0], inputs.shape[1]))
# lstm_out = keras.layers.LSTM(32)(inputs)
# outputs = keras.layers.Dense(1)(lstm_out)

# model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
# model.summary()

# path_checkpoint = "model_checkpoint.h5"
# es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

# modelckpt_callback = keras.callbacks.ModelCheckpoint(
#     monitor="val_loss",
#     filepath=path_checkpoint,
#     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
# )

# model.fit(
#     x_train,
#     y_train,
#     batch_size=10,
#     epochs=100,
#     validation_split=0.2
# )

# predictions = model.predict(x_test)

# plt.plot(x_test, y_test, 'g')
# plt.plot(x_test, predictions, 'r')
# plt.show()

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(len(loss))
# plt.figure()
# plt.plot(epochs, loss, "b", label="Training loss")
# plt.plot(epochs, val_loss, "r", label="Validation loss")
# plt.show()