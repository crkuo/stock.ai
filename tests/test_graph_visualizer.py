import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from src.graph.graph_visualizer import GraphVisualizer


class TestGraphVisualizer:
    def test_should_create_static_graph_visualization(self):
        visualizer = GraphVisualizer()

        # Create sample stock data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10 + ['MSFT'] * 10,
            'close_price': [100 + i for i in range(10)] +
                          [200 + i*2 for i in range(10)] +
                          [150 + i*1.5 for i in range(10)]
        })

        plot_result = visualizer.create_static_plot(stock_data, title="Stock Correlation Graph")

        assert plot_result is not None
        assert 'figure' in plot_result
        assert 'layout' in plot_result

    def test_should_create_interactive_graph_visualization(self):
        visualizer = GraphVisualizer()

        # Create sample stock data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(15)]
        stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 15 + ['GOOGL'] * 15 + ['MSFT'] * 15,
            'close_price': [100 + i*0.5 for i in range(15)] +
                          [200 + i*1.0 for i in range(15)] +
                          [150 + i*0.8 for i in range(15)]
        })

        interactive_result = visualizer.create_interactive_plot(
            stock_data,
            title="Interactive Stock Network",
            node_size_metric="centrality"
        )

        assert interactive_result is not None
        assert 'html_content' in interactive_result or 'plotly_figure' in interactive_result

    def test_should_visualize_correlation_matrix_heatmap(self):
        visualizer = GraphVisualizer()

        # Create correlation matrix
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.3],
            'GOOGL': [0.8, 1.0, 0.6],
            'MSFT': [0.3, 0.6, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        heatmap_result = visualizer.create_correlation_heatmap(
            correlation_matrix,
            title="Stock Correlation Matrix"
        )

        assert heatmap_result is not None
        assert 'figure' in heatmap_result

    def test_should_visualize_temporal_graph_changes(self):
        visualizer = GraphVisualizer()

        # Create time series data with changing correlations
        periods = []
        for period in range(3):
            dates = [datetime(2024, 1, 1) + timedelta(days=period*10 + i) for i in range(10)]
            period_data = pd.DataFrame({
                'timestamp': dates * 2,
                'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
                'close_price': [100 + i + period*5 for i in range(10)] +
                              [200 + i*2 + period*10 for i in range(10)]
            })
            periods.append(period_data)

        temporal_result = visualizer.create_temporal_visualization(
            periods,
            title="Graph Evolution Over Time"
        )

        assert temporal_result is not None
        assert 'animation_frames' in temporal_result or 'time_series_plots' in temporal_result

    def test_should_highlight_nodes_by_metrics(self):
        visualizer = GraphVisualizer()

        # Create sample data with different node importance
        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)] * 4,
            'symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
            'close_price': [100, 200, 150, 300]
        })

        node_metrics = {
            'AAPL': {'centrality': 0.8, 'volume': 1000000},
            'GOOGL': {'centrality': 0.6, 'volume': 800000},
            'MSFT': {'centrality': 0.9, 'volume': 1200000},
            'TSLA': {'centrality': 0.4, 'volume': 600000}
        }

        highlighted_result = visualizer.create_highlighted_plot(
            stock_data,
            node_metrics,
            highlight_metric="centrality",
            title="Nodes by Centrality"
        )

        assert highlighted_result is not None
        assert 'figure' in highlighted_result

    def test_should_save_visualization_to_file(self):
        visualizer = GraphVisualizer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1)] * 2,
            'symbol': ['AAPL', 'GOOGL'],
            'close_price': [100, 200]
        })

        # Test saving to different formats
        png_result = visualizer.save_plot(
            stock_data,
            output_path="/tmp/test_graph.png",
            format="png"
        )

        html_result = visualizer.save_plot(
            stock_data,
            output_path="/tmp/test_graph.html",
            format="html"
        )

        assert png_result is True
        assert html_result is True

    def test_should_create_causality_flow_diagram(self):
        visualizer = GraphVisualizer()

        # Create causality matrix (p-values)
        causality_matrix = pd.DataFrame({
            'AAPL': [0.0, 0.02, 0.15],      # AAPL causes GOOGL
            'GOOGL': [0.08, 0.0, 0.03],     # GOOGL causes MSFT
            'MSFT': [0.12, 0.10, 0.0]       # No significant causality from MSFT
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        causality_result = visualizer.create_causality_diagram(
            causality_matrix,
            significance_threshold=0.05,
            title="Stock Causality Network"
        )

        assert causality_result is not None
        assert 'figure' in causality_result
        assert 'directed_edges' in causality_result

    def test_should_create_network_statistics_dashboard(self):
        visualizer = GraphVisualizer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)] * 3,
            'symbol': ['AAPL'] * 20 + ['GOOGL'] * 20 + ['MSFT'] * 20,
            'close_price': [100 + i + np.random.normal(0, 1) for i in range(20)] +
                          [200 + i*2 + np.random.normal(0, 2) for i in range(20)] +
                          [150 + i*1.5 + np.random.normal(0, 1.5) for i in range(20)]
        })

        dashboard_result = visualizer.create_network_dashboard(
            stock_data,
            title="Stock Network Analytics Dashboard"
        )

        assert dashboard_result is not None
        assert 'dashboard_components' in dashboard_result
        assert 'summary_statistics' in dashboard_result