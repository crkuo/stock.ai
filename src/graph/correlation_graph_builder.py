import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional


class MockGraph:
    def __init__(self):
        self._nodes = set()
        self._edges = {}

    def nodes(self):
        return list(self._nodes)

    def edges(self):
        return list(self._edges.keys())

    def has_edge(self, node1, node2):
        return (node1, node2) in self._edges or (node2, node1) in self._edges

    def add_node(self, node):
        self._nodes.add(node)

    def add_edge(self, node1, node2, weight=None):
        self._nodes.add(node1)
        self._nodes.add(node2)
        self._edges[(node1, node2)] = {'weight': weight}

    def __getitem__(self, nodes):
        if isinstance(nodes, list) and len(nodes) == 2:
            node1, node2 = nodes
            if (node1, node2) in self._edges:
                return self._edges[(node1, node2)]
            elif (node2, node1) in self._edges:
                return self._edges[(node2, node1)]
        return {}


class CorrelationGraphBuilder:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def calculate_correlation_matrix(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        # Pivot data to have symbols as columns
        pivoted = stock_data.pivot(index='timestamp', columns='symbol', values='close_price')

        # Calculate correlation matrix
        correlation_matrix = pivoted.corr()

        return correlation_matrix

    def build_graph_from_correlation(self, correlation_matrix: pd.DataFrame) -> MockGraph:
        graph = MockGraph()

        # Add all symbols as nodes
        for symbol in correlation_matrix.index:
            graph.add_node(symbol)

        # Add edges based on correlation threshold
        for i, symbol1 in enumerate(correlation_matrix.index):
            for j, symbol2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate edges
                    correlation = correlation_matrix.loc[symbol1, symbol2]
                    if abs(correlation) >= self.threshold:
                        graph.add_edge(symbol1, symbol2, weight=correlation)

        return graph

    def calculate_rolling_correlation(self, stock_data: pd.DataFrame, window: int) -> pd.Series:
        # Simple mock implementation
        pivoted = stock_data.pivot(index='timestamp', columns='symbol', values='close_price')

        if len(pivoted.columns) >= 2:
            col1, col2 = pivoted.columns[0], pivoted.columns[1]
            rolling_corr = pivoted[col1].rolling(window=window).corr(pivoted[col2])
            return rolling_corr.dropna()

        return pd.Series([])

    def build_graph(self, stock_data: pd.DataFrame) -> MockGraph:
        # Complete pipeline: calculate correlation matrix and build graph
        correlation_matrix = self.calculate_correlation_matrix(stock_data)
        graph = self.build_graph_from_correlation(correlation_matrix)
        return graph