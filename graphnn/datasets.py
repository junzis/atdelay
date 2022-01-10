import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensorflow as tf
import networkx as nx

from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj

from extraction.extractadjacency import distance_weight_adjacency

from . import generateNNdataMultiple
from . import getAdjacencyMatrix
from . import airport_dict


class TrafficDataset(Dataset):
    WEIGHT = 0.4
    THRESHOLD = 1000
    
    def __init__(self, airports, timeslotLength, days=10,**kwargs):
        self.timeslotLength = timeslotLength
        self.airports = sorted(airports)
        self.n_airports = len(airports)
        self._maxIndex = int((60/timeslotLength) * 24 * days)

        super().__init__(**kwargs)

    def read(self):
        dataDict = generateNNdataMultiple(self.airports, self.timeslotLength, GNNFormat=True)
        flight_adjacency = getAdjacencyMatrix(self.airports)
        distance_adjacency = distance_weight_adjacency(self.airports, threshold=self.THRESHOLD)
        adjacencies = self.WEIGHT * distance_adjacency + (1 - self.WEIGHT) * flight_adjacency
        # print(list(dataDict.values()))
        n_features = len(list(dataDict.values())[0]["X"].columns)
        n_labels = len(list(dataDict.values())[0]["Y"].columns) # 2

        def makeGraph(timeIndex):
            # print((self.n_airports, n_features))
            X = np.zeros((self.n_airports, n_features))
            Y = np.zeros((self.n_airports, n_labels))

            # fully connected graph for now
            # A = np.ones((self.n_airports, self.n_airports)) - np.identity(self.n_airports)
            A = adjacencies[timeIndex]
            print(A)
            print(A.shape)


            for AirportIndex, airport in enumerate(self.airports):
                X[AirportIndex] = dataDict[airport]["X"].iloc[timeIndex,:].to_numpy()
                Y[AirportIndex] = dataDict[airport]["Y"].iloc[timeIndex,:].to_numpy()

            return Graph(x=X, a=A, y=Y)


        final = [] # list of graphs
        for timeIndex in range(self._maxIndex+1):
            final.append(makeGraph(timeIndex))

        return final


    def visualiseGraph(self, nthGraph=0):
        graph = self[nthGraph]
        adj = graph.a
        G = nx.convert_matrix.from_numpy_array(adj)
        labels = {}
        pos = {}
        for idx, airport in enumerate(self.airports):
            G.nodes[idx]['name'] = airport , round(graph.y[idx][0],2) , round(graph.y[idx][1],2)
            pos[idx] = [airport_dict[airport]["latitude"],  airport_dict[airport]["longitude"]]
            labels[idx]= airport

        nx.draw(G, pos)
        node_labels = nx.get_node_attributes(G,'name')
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        plt.show()