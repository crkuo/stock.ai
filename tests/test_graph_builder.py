import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.graph.correlation_graph_builder import CorrelationGraphBuilder


class TestCorrelationGraphBuilder:
    def test_should_create_correlation_matrix_from_stock_data(self):
        builder = CorrelationGraphBuilder()

        # Create sample stock data with multiple symbols
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10 + ['MSFT'] * 10,
            'close_price': [100 + i for i in range(10)] +
                          [200 + i*2 for i in range(10)] +
                          [150 + i*1.5 for i in range(10)]
        })

        correlation_matrix = builder.calculate_correlation_matrix(stock_data)

        assert correlation_matrix is not None
        assert correlation_matrix.shape == (3, 3)  # 3 stocks
        assert 'AAPL' in correlation_matrix.index
        assert 'GOOGL' in correlation_matrix.index
        assert 'MSFT' in correlation_matrix.index

    def test_should_build_graph_from_correlation_matrix(self):
        builder = CorrelationGraphBuilder(threshold=0.5)

        # Create a simple correlation matrix
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.3],
            'GOOGL': [0.8, 1.0, 0.6],
            'MSFT': [0.3, 0.6, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        graph = builder.build_graph_from_correlation(correlation_matrix)

        assert graph is not None
        assert len(graph.nodes()) == 3
        assert len(graph.edges()) > 0
        # Should have edges for correlations above threshold (0.5)
        assert graph.has_edge('AAPL', 'GOOGL')  # correlation 0.8
        assert graph.has_edge('GOOGL', 'MSFT')  # correlation 0.6

    def test_should_filter_edges_by_threshold(self):
        builder = CorrelationGraphBuilder(threshold=0.7)

        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.3],
            'GOOGL': [0.8, 1.0, 0.6],
            'MSFT': [0.3, 0.6, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        graph = builder.build_graph_from_correlation(correlation_matrix)

        # With threshold 0.7, only AAPL-GOOGL edge should exist (0.8 > 0.7)
        assert graph.has_edge('AAPL', 'GOOGL')
        assert not graph.has_edge('GOOGL', 'MSFT')  # 0.6 < 0.7
        assert not graph.has_edge('AAPL', 'MSFT')   # 0.3 < 0.7

    def test_should_add_edge_weights_to_graph(self):
        builder = CorrelationGraphBuilder(threshold=0.5)

        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8],
            'GOOGL': [0.8, 1.0]
        }, index=['AAPL', 'GOOGL'])

        graph = builder.build_graph_from_correlation(correlation_matrix)

        # Check edge weight
        edge_weight = graph[['AAPL', 'GOOGL']]['weight']
        assert abs(edge_weight - 0.8) < 1e-10

    def test_should_calculate_rolling_correlation(self):
        builder = CorrelationGraphBuilder()

        # Create time series data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 20 + ['GOOGL'] * 20,
            'close_price': [100 + i + np.random.normal(0, 1) for i in range(20)] +
                          [200 + i*2 + np.random.normal(0, 2) for i in range(20)]
        })

        rolling_corr = builder.calculate_rolling_correlation(stock_data, window=10)

        assert rolling_corr is not None
        assert len(rolling_corr) > 0  # Should have some correlation values

    def test_should_build_complete_graph_pipeline(self):
        builder = CorrelationGraphBuilder(threshold=0.6)

        # Create sample stock data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(15)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 15 + ['GOOGL'] * 15 + ['MSFT'] * 15,
            'close_price': [100 + i*0.5 for i in range(15)] +
                          [200 + i*1.0 for i in range(15)] +
                          [150 + i*0.8 for i in range(15)]
        })

        graph = builder.build_graph(stock_data)

        assert graph is not None
        assert len(graph.nodes()) == 3
        assert all(node in ['AAPL', 'GOOGL', 'MSFT'] for node in graph.nodes())