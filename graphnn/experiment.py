"""
This example implements the experiments on citation networks from the paper:
Graph Attention Networks (https://arxiv.org/abs/1710.10903)
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
from spektral.layers import GCNConv, GlobalSumPool
import keras.backend as K
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess

set_seed(0)

# class MyModel(Model):
#     def __init__(self):
#         super().__init__()
#         # Parameters
#         channels = 8  # Number of channels in each head of the first GAT layer
#         n_attn_heads = 8  # Number of attention heads in first GAT layer
#         dropout = 0.6  # Dropout rate for the features and adjacency matrix
#         l2_reg = 2.5e-4  # L2 regularization rate
#         learning_rate = 5e-3  # Learning rate
#         epochs = 200  # Number of training epochs
#         patience = 100  # Patience for early stopping

#         self.gc_1 = GATConv(
#         channels,
#         attn_heads=n_attn_heads,
#         concat_heads=True,
#         dropout_rate=dropout,
#         activation="elu",
#         kernel_regularizer=l2(l2_reg),
#         attn_kernel_regularizer=l2(l2_reg),
#         bias_regularizer=l2(l2_reg),
#         )
#         self.gc_2 = GATConv(
#         10,
#         attn_heads=1,
#         concat_heads=False,
#         dropout_rate=dropout,
#         activation="softmax",
#         kernel_regularizer=l2(l2_reg),
#         attn_kernel_regularizer=l2(l2_reg),
#         bias_regularizer=l2(l2_reg),
#         )
#         self.do_1 = Dropout(dropout)
#         self.do_2 = Dropout(dropout)

#     def call(self, dataset, training=False):
#         N = dataset.n_nodes  # Number of nodes in the graph
#         F = dataset.n_node_features  # Original size of node features
#         n_out = dataset.n_labels  # Number of classes

#         # Model definition
#         x_in = Input(shape=(F,))
#         a_in = Input((N,), sparse=True)

#         out = self.gc_1(inputs)
#         # if training:
#         #     out = self.do_1(out, training=training)
#         out = self.gc_2(out)
#         # if training:
#         #     out = self.do_2(out, training=training)
#         return out


# Load data
dataset = Citation("cora", normalize_x=True, transforms=[LayerPreprocess(GATConv)])
# dataset = TrafficDataset( ICAOTOP10, 60)
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

# Parameters
channels = 8  # Number of channels in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
dropout = 0.6  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 5e-3  # Learning rate
epochs = 200  # Number of training epochs
patience = 100  # Patience for early stopping


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes

# N = dataset.n_airports  # Number of nodes in the graph
# F = 7  # Original size of node features
# n_out = dataset.n_labels  # Number of classes


# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)

# do_1 = Dropout(dropout)(x_in)
gc_1 = GATConv(
    channels,
    attn_heads=n_attn_heads,
    concat_heads=True,
    dropout_rate=dropout,
    activation="elu",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([x_in, a_in])
# do_2 = Dropout(dropout)(gc_1)
gc_2 = GATConv(
    n_out,
    attn_heads=1,
    concat_heads=False,
    dropout_rate=dropout,
    activation="softmax",
    kernel_regularizer=l2(l2_reg),
    attn_kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg),
)([gc_1, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=gc_2)
optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer,
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)
model.summary()

# Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)
print(gc_2)

print(
    model.predict(SingleLoader(dataset, sample_weights=weights_tr).load(), steps=2)[
        0, :
    ]
)
