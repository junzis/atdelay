import pandas
import tensorflow as tf

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input, Dense
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
from spektral.layers import GCNConv, GlobalSumPool, DiffPool
from spektral.data.loaders import SingleLoader, DisjointLoader, BatchLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess
from datasets import *
from extraction.extractionvalues import *


class GATLSTM(Model):
    def __init__(self, channels=8, n_attn_heads=8, n_out=2, l2_reg=2.5e-4):
        self.layers = [
            GATConv(
                channels,
                attn_heads=n_attn_heads,
                concat_heads=True,
                activation="relu",
                kernel_regularizer=l2(l2_reg),
                attn_kernel_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg),
            ),
            GATConv(
                n_out,
                attn_heads=1,
                concat_heads=False,
                activation="softmax",
                kernel_regularizer=l2(l2_reg),
                attn_kernel_regularizer=l2(l2_reg),
                bias_regularizer=l2(l2_reg),
            ),
        ]

    def call(self, inputs):
        for layer in self.layers:
            out = layer(inputs)
        return out
