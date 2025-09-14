import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .correlation_graph_builder import CorrelationGraphBuilder, MockGraph


class GraphPruner:
    def __init__(self, correlation_threshold: float = 0.5, significance_level: float = 0.05,
                 min_degree: int = 1, preserve_mst: bool = False):
        self.correlation_threshold = correlation_threshold
        self.significance_level = significance_level
        self.min_degree = min_degree
        self.preserve_mst = preserve_mst
        self.graph_builder = CorrelationGraphBuilder()

    def prune_by_correlation(self, correlation_matrix: pd.DataFrame) -> pd.DataFrame:
        """Prune edges below correlation threshold"""
        pruned_matrix = correlation_matrix.copy()

        # Set weak correlations to 0
        mask = np.abs(pruned_matrix) < self.correlation_threshold
        # Keep diagonal elements (self-correlation = 1)
        np.fill_diagonal(mask.values, False)
        pruned_matrix[mask] = 0.0

        return pruned_matrix

    def prune_by_significance(self, p_value_matrix: pd.DataFrame) -> pd.DataFrame:
        """Prune edges that are not statistically significant"""
        pruned_matrix = p_value_matrix.copy()

        # Convert p-values to binary significance (1 if significant, 0 if not)
        significance_mask = p_value_matrix < self.significance_level
        # Keep diagonal (p-value for self = 0)
        np.fill_diagonal(significance_mask.values, True)

        # Set non-significant relationships to 0
        pruned_matrix[~significance_mask] = 0.0
        # Convert significant p-values to edge weights (1 - p_value)
        pruned_matrix[significance_mask] = 1.0 - pruned_matrix[significance_mask]

        return pruned_matrix

    def prune_low_degree_nodes(self, stock_data: pd.DataFrame) -> MockGraph:
        """Remove nodes with degree below minimum threshold"""
        # Build initial graph
        correlation_matrix = self.graph_builder.calculate_correlation_matrix(stock_data)

        # Prune by correlation first
        pruned_matrix = self.prune_by_correlation(correlation_matrix)

        # Calculate degrees (number of non-zero connections)
        degrees = {}
        for symbol in pruned_matrix.index:
            # Count non-zero, non-diagonal elements
            connections = (pruned_matrix.loc[symbol] != 0).sum() - 1  # Subtract diagonal
            degrees[symbol] = connections

        # Keep only nodes with sufficient degree
        high_degree_nodes = [node for node, degree in degrees.items()
                           if degree >= self.min_degree]

        # Filter matrix to keep only high-degree nodes
        if high_degree_nodes:
            final_matrix = pruned_matrix.loc[high_degree_nodes, high_degree_nodes]
            return self.graph_builder.build_graph_from_correlation(final_matrix)
        else:
            # Return empty graph if no nodes meet criteria
            empty_graph = MockGraph()
            return empty_graph

    def keep_top_k_edges(self, correlation_matrix: pd.DataFrame, k: int) -> pd.DataFrame:
        """Keep only the k strongest correlations"""
        pruned_matrix = pd.DataFrame(
            np.zeros_like(correlation_matrix),
            index=correlation_matrix.index,
            columns=correlation_matrix.columns
        )

        # Keep diagonal (self-correlations)
        np.fill_diagonal(pruned_matrix.values, 1.0)

        # Get upper triangle (avoid double counting symmetric edges)
        upper_triangle = np.triu_indices_from(correlation_matrix, k=1)
        edge_values = correlation_matrix.values[upper_triangle]
        edge_positions = list(zip(upper_triangle[0], upper_triangle[1]))

        # Sort by absolute correlation strength
        sorted_edges = sorted(zip(edge_values, edge_positions),
                            key=lambda x: abs(x[0]), reverse=True)

        # Keep only top k edges
        for i, (value, (row, col)) in enumerate(sorted_edges):
            if i < k:
                pruned_matrix.iloc[row, col] = value
                pruned_matrix.iloc[col, row] = value  # Symmetric
            else:
                break

        return pruned_matrix

    def prune_by_percentile(self, correlation_matrix: pd.DataFrame, percentile: float) -> pd.DataFrame:
        """Keep only edges above specified percentile"""
        pruned_matrix = correlation_matrix.copy()

        # Get upper triangle values (excluding diagonal)
        upper_triangle = np.triu_indices_from(correlation_matrix, k=1)
        edge_values = np.abs(correlation_matrix.values[upper_triangle])

        # Calculate threshold
        threshold = np.percentile(edge_values, percentile)

        # Apply threshold
        mask = np.abs(pruned_matrix) < threshold
        np.fill_diagonal(mask.values, False)  # Keep diagonal
        pruned_matrix[mask] = 0.0

        return pruned_matrix

    def prune_with_mst_preservation(self, correlation_matrix: pd.DataFrame,
                                   threshold: float) -> pd.DataFrame:
        """Prune while preserving minimum spanning tree for connectivity"""
        # Simple MST preservation: keep strongest edge per node
        pruned_matrix = correlation_matrix.copy()

        # First apply threshold
        mask = np.abs(pruned_matrix) < threshold
        np.fill_diagonal(mask.values, False)

        # For each node, preserve its strongest connection
        for i, node in enumerate(correlation_matrix.index):
            row = np.abs(correlation_matrix.iloc[i])
            # Find strongest connection (excluding self)
            strongest_idx = row.iloc[1:].idxmax() if i == 0 else row.drop(node).idxmax()
            strongest_col = correlation_matrix.columns.get_loc(strongest_idx)

            # Preserve this edge even if below threshold
            mask.iloc[i, strongest_col] = False
            mask.iloc[strongest_col, i] = False

        pruned_matrix[mask] = 0.0
        return pruned_matrix

    def calculate_adaptive_threshold(self, stock_data: pd.DataFrame) -> float:
        """Calculate adaptive threshold based on market conditions"""
        # Calculate volatilities
        symbols = stock_data['symbol'].unique()
        volatilities = []

        for symbol in symbols:
            symbol_data = stock_data[stock_data['symbol'] == symbol]['close_price']
            if len(symbol_data) > 1:
                returns = symbol_data.pct_change().dropna()
                vol = returns.std()
                volatilities.append(vol)

        if volatilities:
            avg_volatility = np.mean(volatilities)
            # High volatility -> lower threshold (keep more edges)
            # Low volatility -> higher threshold (prune more edges)
            adaptive_threshold = max(0.1, min(0.9, 0.5 - avg_volatility))
            return adaptive_threshold

        return self.correlation_threshold

    def is_connected(self, correlation_matrix: pd.DataFrame) -> bool:
        """Check if pruned graph remains connected"""
        # Simple connectivity check: ensure no isolated nodes
        non_diagonal_sums = correlation_matrix.sum(axis=1) - 1  # Subtract diagonal
        return all(s > 0 for s in non_diagonal_sums)

    def prune_combined_criteria(self, correlation_matrix: pd.DataFrame,
                               p_value_matrix: pd.DataFrame) -> pd.DataFrame:
        """Apply multiple pruning criteria simultaneously"""
        pruned_matrix = correlation_matrix.copy()

        # Combined criteria: correlation above threshold AND significant
        correlation_mask = np.abs(correlation_matrix) >= self.correlation_threshold
        significance_mask = p_value_matrix < self.significance_level

        # Keep diagonal
        np.fill_diagonal(correlation_mask.values, True)
        np.fill_diagonal(significance_mask.values, True)

        # Apply combined criteria
        combined_mask = correlation_mask & significance_mask
        pruned_matrix[~combined_mask] = 0.0

        return pruned_matrix