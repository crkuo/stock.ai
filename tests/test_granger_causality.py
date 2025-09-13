import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.graph.granger_causality import GrangerCausalityAnalyzer


class TestGrangerCausalityAnalyzer:
    def test_should_calculate_granger_causality_between_two_stocks(self):
        analyzer = GrangerCausalityAnalyzer(max_lag=3)

        # Create sample time series data where X causes Y
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # X series: independent
        x_series = [100 + np.random.normal(0, 1) for _ in range(30)]

        # Y series: depends on lagged X values
        y_series = [200]  # Initial value
        for i in range(1, 30):
            lag_effect = x_series[i-1] * 0.5 if i > 0 else 0
            y_series.append(200 + lag_effect + np.random.normal(0, 1))

        stock_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 30 + ['GOOGL'] * 30,
            'close_price': x_series + y_series
        })

        result = analyzer.test_granger_causality(stock_data, 'AAPL', 'GOOGL')

        assert result is not None
        assert 'p_value' in result
        assert 'f_statistic' in result
        assert 'causality_detected' in result
        assert isinstance(result['p_value'], float)

    def test_should_calculate_granger_causality_matrix_for_multiple_stocks(self):
        analyzer = GrangerCausalityAnalyzer(max_lag=2)

        # Create sample data for 3 stocks
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(25)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 25 + ['GOOGL'] * 25 + ['MSFT'] * 25,
            'close_price': [100 + i + np.random.normal(0, 1) for i in range(25)] +
                          [200 + i*2 + np.random.normal(0, 1) for i in range(25)] +
                          [150 + i*1.5 + np.random.normal(0, 1) for i in range(25)]
        })

        causality_matrix = analyzer.calculate_causality_matrix(stock_data)

        assert causality_matrix is not None
        assert causality_matrix.shape == (3, 3)  # 3x3 matrix for 3 stocks
        assert 'AAPL' in causality_matrix.index
        assert 'GOOGL' in causality_matrix.columns
        assert 'MSFT' in causality_matrix.index

    def test_should_detect_significant_causality_relationships(self):
        analyzer = GrangerCausalityAnalyzer(max_lag=2, significance_level=0.05)

        # Create data with clear causal relationship
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]

        # Create strong causal relationship: X -> Y
        x_values = [100 + i + np.random.normal(0, 0.5) for i in range(20)]
        y_values = [200]
        for i in range(1, 20):
            y_values.append(200 + x_values[i-1] * 0.8 + np.random.normal(0, 0.1))

        stock_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['CAUSE'] * 20 + ['EFFECT'] * 20,
            'close_price': x_values + y_values
        })

        result = analyzer.test_granger_causality(stock_data, 'CAUSE', 'EFFECT')

        assert result['causality_detected'] is True
        assert result['p_value'] < 0.05

    def test_should_handle_insufficient_data_gracefully(self):
        analyzer = GrangerCausalityAnalyzer(max_lag=5)

        # Create minimal dataset (less than required for lag analysis)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 3 + ['GOOGL'] * 3,
            'close_price': [100, 101, 102, 200, 201, 202]
        })

        result = analyzer.test_granger_causality(stock_data, 'AAPL', 'GOOGL')

        assert result is not None
        assert 'error' in result or result['causality_detected'] is False

    def test_should_calculate_optimal_lag_length(self):
        analyzer = GrangerCausalityAnalyzer()

        # Create time series data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['X'] * 50 + ['Y'] * 50,
            'close_price': [100 + i + np.random.normal(0, 1) for i in range(50)] +
                          [200 + i*1.5 + np.random.normal(0, 1) for i in range(50)]
        })

        optimal_lag = analyzer.find_optimal_lag(stock_data, 'X', 'Y', max_lag=5)

        assert optimal_lag is not None
        assert isinstance(optimal_lag, int)
        assert 1 <= optimal_lag <= 5

    def test_should_build_causality_graph_from_matrix(self):
        analyzer = GrangerCausalityAnalyzer(significance_level=0.05)

        # Create mock causality matrix
        causality_matrix = pd.DataFrame({
            'AAPL': [0.0, 0.02, 0.10],      # AAPL causes GOOGL (p=0.02)
            'GOOGL': [0.08, 0.0, 0.03],     # GOOGL causes MSFT (p=0.03)
            'MSFT': [0.15, 0.12, 0.0]       # No significant causality from MSFT
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        causality_graph = analyzer.build_causality_graph(causality_matrix)

        assert causality_graph is not None
        assert len(causality_graph.nodes()) == 3
        # Should have edges for significant relationships (p < 0.05)
        assert causality_graph.has_edge('AAPL', 'GOOGL')  # p=0.02 < 0.05
        assert causality_graph.has_edge('GOOGL', 'MSFT')  # p=0.03 < 0.05