import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj

from . import generateNNdataMultiple


class TrafficDataset(Dataset):
    def __init__(self, airports, timeslotLength, days=10,**kwargs):
        self.timeslotLength = timeslotLength
        self.airports = airports
        self.n_airports = len(airports)
        self._maxIndex = int((60/timeslotLength) * 24 * days)

        super().__init__(**kwargs)

    def read(self):
        dataDict = generateNNdataMultiple(self.airports, self.timeslotLength, GNNFormat=True)
        # print(list(dataDict.values()))
        n_features = len(list(dataDict.values())[0]["X"].columns)
        n_labels = len(list(dataDict.values())[0]["Y"].columns) # 2

        def makeGraph(timeIndex):
            # print((self.n_airports, n_features))
            X = np.zeros((self.n_airports, n_features))
            Y = np.zeros((self.n_airports, n_labels))

            # fully connected graph for now
            A = np.ones((self.n_airports, self.n_airports)) - np.identity(self.n_airports)
            # print(A)

            for AirportIndex, airport in enumerate(self.airports):
                X[AirportIndex] = dataDict[airport]["X"].iloc[timeIndex,:].to_numpy()
                Y[AirportIndex] = dataDict[airport]["Y"].iloc[timeIndex,:].to_numpy()

            return Graph(x=X, a=A, y=Y)


        final = [] # list of graphs
        for timeIndex in range(self._maxIndex+1):
            final.append(makeGraph(timeIndex))

        return final


    # def read(self):

# def visualiseGraph(graph):
#     adj = graph.a
#     G = nx.convert_matrix.from_numpy_array(adj)
#     print(G.number_of_nodes())
#     subax1 = plt.subplot(121)
#     nx.draw(G, with_labels=True, font_weight='bold')
#     plt.show()  