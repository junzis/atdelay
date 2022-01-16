import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense, LSTM, Reshape,Concatenate 
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
# from spektral.layers import GCNConv, GlobalSumPool, DiffPool
import keras.backend as K
from spektral.data.loaders import SingleLoader, DisjointLoader, BatchLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess
from datasets import *
from extraction.extractionvalues import *

set_seed(0)

airports = ICAOTOP10
lookback = 8 # steps
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 2e-4  # Learning rate

graphData = FlightNetworkDataset(airports, timeslotLength=60, start=datetime(2019,3,1))

print("HELLO", len(graphData))

N = len(airports)  # Number of nodes in the graph
F = graphData.n_node_features  # Original size of node features
n_out = graphData.n_labels  # Number of classes

x_in = Input(shape=(N, F), name="Features")
a_in = Input((N, N), sparse=True, name="Adjacency")

gat = GATConv(
    channels=20,
    attn_heads=12,
    # concat_heads=False,
    # activation="relu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([x_in, a_in])
# reshaper = Concatenate()
# lstm1 = LSTM(F, return_sequences=True)(reshaper)
# lstm2 = LSTM(20)(lstm1)
dense1 = Dense(100)(gat)
dense2 = Dense(10)(dense1)
outtput = Dense(2)(dense2)

model = Model(inputs=[x_in, a_in], outputs=outtput)

tf.keras.utils.plot_model(model, show_shapes=True)

optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=MeanAbsoluteError(reduction="auto", name="mean_absolute_error"),
    weighted_metrics=["acc"],
)
model.summary()

# Train model
# 720 points total 600 
loader_tr = BatchLoader(graphData[0:504], shuffle=False)
loader_val = BatchLoader(graphData[504: (504+144)], shuffle=False)
loader_test = BatchLoader(graphData[(504+144)::], shuffle=False)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_tr.load(),
    validation_steps=loader_tr.steps_per_epoch,
    epochs=10,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1,
)

timeSlice = 2
ypred = model.predict(loader_test).load()
yactual = [g.y for g in graphData[(504+144)::]]

plt.plot(ypred)
plt.plot(yactual)

# graphData.visualiseGraph(timeSlice)