import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.graph.graph_pruner import GraphPruner


class TestGraphPruner:
    def test_should_prune_weak_edges_by_correlation_threshold(self):
        pruner = GraphPruner(correlation_threshold=0.6)

        # Create correlation matrix with weak and strong correlations
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.3, 0.7],
            'GOOGL': [0.8, 1.0, 0.2, 0.9],
            'MSFT': [0.3, 0.2, 1.0, 0.4],
            'TSLA': [0.7, 0.9, 0.4, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT', 'TSLA'])

        pruned_matrix = pruner.prune_by_correlation(correlation_matrix)

        # Should remove weak correlations (< 0.6)
        assert pruned_matrix.loc['AAPL', 'MSFT'] == 0.0  # Was 0.3, below threshold
        assert pruned_matrix.loc['GOOGL', 'MSFT'] == 0.0  # Was 0.2, below threshold
        assert pruned_matrix.loc['AAPL', 'GOOGL'] == 0.8  # Above threshold, kept
        assert pruned_matrix.loc['GOOGL', 'TSLA'] == 0.9  # Above threshold, kept

    def test_should_prune_edges_by_statistical_significance(self):
        pruner = GraphPruner(significance_level=0.05)

        # Create p-value matrix (lower values = more significant)
        p_value_matrix = pd.DataFrame({
            'AAPL': [0.0, 0.02, 0.15, 0.03],
            'GOOGL': [0.02, 0.0, 0.08, 0.01],
            'MSFT': [0.15, 0.08, 0.0, 0.12],
            'TSLA': [0.03, 0.01, 0.12, 0.0]
        }, index=['AAPL', 'GOOGL', 'MSFT', 'TSLA'])

        pruned_matrix = pruner.prune_by_significance(p_value_matrix)

        # Should keep only significant relationships (p < 0.05)
        assert pruned_matrix.loc['AAPL', 'GOOGL'] > 0  # p=0.02, significant
        assert pruned_matrix.loc['GOOGL', 'TSLA'] > 0  # p=0.01, significant
        assert pruned_matrix.loc['AAPL', 'MSFT'] == 0  # p=0.15, not significant
        assert pruned_matrix.loc['MSFT', 'TSLA'] == 0  # p=0.12, not significant

    def test_should_prune_low_degree_nodes(self):
        pruner = GraphPruner(min_degree=2)

        # Create sample stock data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 5,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10 + ['MSFT'] * 10 +
                      ['ISOLATED1'] * 10 + ['ISOLATED2'] * 10,
            'close_price': [100 + i for i in range(10)] +
                          [200 + i*2 for i in range(10)] +
                          [150 + i*1.5 for i in range(10)] +
                          [50 + np.random.normal(0, 10) for _ in range(10)] +
                          [300 + np.random.normal(0, 15) for _ in range(10)]
        })

        pruned_graph = pruner.prune_low_degree_nodes(stock_data)

        # Should remove isolated nodes with few connections
        remaining_nodes = pruned_graph.nodes()
        assert 'AAPL' in remaining_nodes
        assert 'GOOGL' in remaining_nodes
        assert 'MSFT' in remaining_nodes

    def test_should_keep_top_k_strongest_edges(self):
        pruner = GraphPruner()

        # Create correlation matrix
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.6, 0.4, 0.2],
            'GOOGL': [0.8, 1.0, 0.7, 0.3, 0.1],
            'MSFT': [0.6, 0.7, 1.0, 0.5, 0.3],
            'TSLA': [0.4, 0.3, 0.5, 1.0, 0.6],
            'NVDA': [0.2, 0.1, 0.3, 0.6, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'])

        # Keep only top 5 strongest edges
        pruned_matrix = pruner.keep_top_k_edges(correlation_matrix, k=5)

        # Count non-zero edges (excluding diagonal)
        non_diagonal_mask = ~np.eye(pruned_matrix.shape[0], dtype=bool)
        non_zero_edges = np.count_nonzero(pruned_matrix.values[non_diagonal_mask])

        # Should have exactly 10 non-zero entries (5 edges * 2 due to symmetry)
        assert non_zero_edges == 10

    def test_should_apply_percentile_based_pruning(self):
        pruner = GraphPruner()

        # Create correlation matrix with range of values
        np.random.seed(42)  # For reproducible results
        correlation_matrix = pd.DataFrame(
            np.random.uniform(0, 1, (6, 6)),
            index=['A', 'B', 'C', 'D', 'E', 'F'],
            columns=['A', 'B', 'C', 'D', 'E', 'F']
        )
        # Make it symmetric and set diagonal to 1
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix.values, 1.0)

        # Keep only top 20% of edges
        pruned_matrix = pruner.prune_by_percentile(correlation_matrix, percentile=80)

        # Count remaining edges
        non_diagonal_mask = ~np.eye(pruned_matrix.shape[0], dtype=bool)
        total_possible_edges = np.count_nonzero(non_diagonal_mask)
        remaining_edges = np.count_nonzero(pruned_matrix.values[non_diagonal_mask])

        # Should keep approximately 20% of edges
        edge_ratio = remaining_edges / total_possible_edges
        assert edge_ratio <= 0.25  # Allow some tolerance

    def test_should_preserve_minimum_spanning_tree(self):
        pruner = GraphPruner(preserve_mst=True)

        # Create correlation matrix
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.9, 0.2],
            'GOOGL': [0.9, 1.0, 0.1],
            'MSFT': [0.2, 0.1, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        # Prune with low threshold but preserve MST
        pruned_matrix = pruner.prune_with_mst_preservation(
            correlation_matrix,
            threshold=0.8
        )

        # Should preserve connectivity even with high threshold
        # At least the strongest edge (AAPL-GOOGL) should remain
        assert pruned_matrix.loc['AAPL', 'GOOGL'] > 0

        # Graph should remain connected
        assert pruner.is_connected(pruned_matrix)

    def test_should_apply_adaptive_thresholding(self):
        pruner = GraphPruner()

        # Create time series data with varying market conditions
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]

        # High volatility period
        high_vol_data = pd.DataFrame({
            'timestamp': dates[:15] * 3,
            'symbol': ['AAPL'] * 15 + ['GOOGL'] * 15 + ['MSFT'] * 15,
            'close_price': [100 + np.random.normal(0, 5) for _ in range(15)] +
                          [200 + np.random.normal(0, 10) for _ in range(15)] +
                          [150 + np.random.normal(0, 8) for _ in range(15)]
        })

        # Low volatility period
        low_vol_data = pd.DataFrame({
            'timestamp': dates[15:] * 3,
            'symbol': ['AAPL'] * 15 + ['GOOGL'] * 15 + ['MSFT'] * 15,
            'close_price': [100 + i*0.1 for i in range(15)] +
                          [200 + i*0.2 for i in range(15)] +
                          [150 + i*0.15 for i in range(15)]
        })

        high_vol_threshold = pruner.calculate_adaptive_threshold(high_vol_data)
        low_vol_threshold = pruner.calculate_adaptive_threshold(low_vol_data)

        # High volatility should have lower threshold (keep more edges)
        assert high_vol_threshold < low_vol_threshold

    def test_should_handle_empty_graph_gracefully(self):
        pruner = GraphPruner(correlation_threshold=0.9)

        # Create correlation matrix with all weak correlations
        weak_correlation_matrix = pd.DataFrame({
            'A': [1.0, 0.1, 0.2],
            'B': [0.1, 1.0, 0.3],
            'C': [0.2, 0.3, 1.0]
        }, index=['A', 'B', 'C'])

        pruned_matrix = pruner.prune_by_correlation(weak_correlation_matrix)

        # Should handle gracefully and return valid matrix
        assert pruned_matrix is not None
        assert pruned_matrix.shape == weak_correlation_matrix.shape
        # All off-diagonal elements should be 0
        assert (pruned_matrix.values[~np.eye(3, dtype=bool)] == 0).all()

    def test_should_combine_multiple_pruning_criteria(self):
        pruner = GraphPruner(
            correlation_threshold=0.4,
            significance_level=0.05,
            min_degree=1
        )

        correlation_matrix = pd.DataFrame({
            'A': [1.0, 0.7, 0.3, 0.8],
            'B': [0.7, 1.0, 0.2, 0.6],
            'C': [0.3, 0.2, 1.0, 0.1],
            'D': [0.8, 0.6, 0.1, 1.0]
        }, index=['A', 'B', 'C', 'D'])

        p_value_matrix = pd.DataFrame({
            'A': [0.0, 0.01, 0.15, 0.02],
            'B': [0.01, 0.0, 0.20, 0.03],
            'C': [0.15, 0.20, 0.0, 0.25],
            'D': [0.02, 0.03, 0.25, 0.0]
        }, index=['A', 'B', 'C', 'D'])

        pruned_matrix = pruner.prune_combined_criteria(
            correlation_matrix,
            p_value_matrix
        )

        # Should apply all criteria: correlation > 0.4 AND p-value < 0.05
        assert pruned_matrix.loc['A', 'B'] > 0  # 0.7 correlation, 0.01 p-value
        assert pruned_matrix.loc['A', 'D'] > 0  # 0.8 correlation, 0.02 p-value
        assert pruned_matrix.loc['A', 'C'] == 0  # 0.3 correlation < 0.4
        assert pruned_matrix.loc['B', 'C'] == 0  # 0.2 correlation < 0.4