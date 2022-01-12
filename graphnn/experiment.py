"""
This example implements the experiments on citation networks from the paper:
Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
from spektral.layers import GCNConv, GlobalSumPool
import keras.backend as K
from spektral.data.loaders import SingleLoader, DisjointLoader, BatchLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess
from datasets import *
from extraction.extractionvalues import *

set_seed(0)

# graphData = Citation("cora", normalize_x=True, transforms=[LayerPreprocess(GATConv)])
airports = ICAOTOP10
# Load data
graphData = FlightNetworkDataset(
    airports, timeslotLength=60
)

# Parameters
channels = 8  # Number of channels in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 1   # Learning rate
epochs = 1000  # Number of training epochs
patience = 100  # Patience for early stopping

N = graphData.n_nodes  # Number of nodes in the graph
F = graphData.n_node_features  # Original size of node features
n_out = graphData.n_labels  # Number of classes


# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)

gc_1 = GATConv(
    channels,
    attn_heads=n_attn_heads,
    concat_heads=True,
    activation="relu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([x_in, a_in])

gc_2 = GATConv(
    n_out,
    attn_heads=1,
    concat_heads=False,
    activation="softmax",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([gc_1, a_in])
dense = Dense(2)(gc_2)

# Build model
model = Model(inputs=[x_in, a_in], outputs=dense)
optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=MeanAbsoluteError(reduction="auto", name="mean_absolute_error"),
    weighted_metrics=["acc"],
)
model.summary()

# Train model
loader_tr = BatchLoader(graphData, shuffle=False, batch_size=50)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_tr.load(),
    validation_steps=loader_tr.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],verbose = 0
)


timeSlice = 2
print(
    model.predict(BatchLoader(graphData[timeSlice:timeSlice+1], shuffle=False).load(), steps=1)[:, :]
)
graphData.visualiseGraph(timeSlice)
# print(model.evaluate(BatchLoader(graphData).load()))
