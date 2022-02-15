import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp
import tensorflow as tf
from datetime import datetime

from spektral.data import Dataset, DisjointLoader, Graph

from tools.extractadjacency import distance_weight_adjacency

from . import generateNNdataMultiple
from . import getAdjacencyMatrix
from . import airport_dict


class FlightNetworkDataset(Dataset):
    WEIGHT = 0.4
    THRESHOLD = 1000

    def __init__(
        self,
        airports: list,
        timeslotLength: int,
        start: datetime = datetime(2019, 3, 1),
        end: datetime = datetime(2019, 3, 31),
        **kwargs
    ):
        """The essential dataset generator for aggregated network flight data in Spektral Graph format
        returns a ready to use spektral dataset.

        Args:
            airports (list): list of aiports to generate the dataset of
            timeslotLength (int): length of timeslot to aggregate by
            start (datetime, optional): start time to generate graphs for (Inclusive). Defaults to datetime(2019, 3, 1).
            end (datetime, optional): end time to generate graphs for (Inclusive). Defaults to datetime(2019, 3, 31).
        """
        self.timeslotLength = timeslotLength
        self.airports = airports
        self.n_airports = len(airports)
        self.start = start
        self.end = end

        timespan = end - start
        self._maxIndex = (
            int((timespan.days * 24 * 60 + timespan.seconds // 60) / timeslotLength) - 1
        )

        super().__init__(**kwargs)

    def read(self) -> list:
        """Reads network data using fuctions in extract module

        Returns:
            list: returns a list of graphs
        """
        dataDict = generateNNdataMultiple(
            self.airports,
            self.timeslotLength,
            GNNFormat=True,
            start=self.start,
            end=self.end,
        )
        flight_adjacency = getAdjacencyMatrix(
            self.airports, self.start, self.end, timeslotLength=self.timeslotLength
        )
        distance_adjacency = distance_weight_adjacency(
            self.airports, threshold=self.THRESHOLD
        )
        adjacencies = (
            self.WEIGHT * distance_adjacency + (1 - self.WEIGHT) * flight_adjacency
        )

        n_features = (list(dataDict.values())[0]["X"]).shape[1]
        n_labels = len(list(dataDict.values())[0]["Y"].columns)  # 2

        def makeGraph(timeIndex):
            X = np.zeros((self.n_airports, n_features))
            Y = np.zeros((self.n_airports, n_labels))
            A = adjacencies[timeIndex]

            for AirportIndex, airport in enumerate(self.airports):
                X[AirportIndex] = dataDict[airport]["X"].iloc[timeIndex, :].to_numpy()
                Y[AirportIndex] = dataDict[airport]["Y"].iloc[timeIndex, :].to_numpy()

            return Graph(x=X, a=A, y=Y)

        final = []  # list of graphs
        for timeIndex in range(self._maxIndex + 1):
            final.append(makeGraph(timeIndex))

        return final

    def visualiseGraph(self, nthGraph=0):
        """visualise a quick representation of a graph without a map

        Args:
            nthGraph (int, optional): the index of the graph to display. Defaults to 0.
        """
        graph = self[nthGraph]
        adj = graph.a
        G = nx.convert_matrix.from_numpy_array(adj)
        labels = {}
        pos = {}
        for idx, airport in enumerate(self.airports):
            G.nodes[idx]["name"] = (
                airport,
                round(graph.y[idx][0], 2),
                round(graph.y[idx][1], 2),
            )
            pos[idx] = [
                airport_dict[airport]["latitude"],
                airport_dict[airport]["longitude"],
            ]
            labels[idx] = airport

        nx.draw(G, pos)
        node_labels = nx.get_node_attributes(G, "name")
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        plt.show()
