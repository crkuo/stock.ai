import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.graph.dynamic_graph_updater import DynamicGraphUpdater


class TestDynamicGraphUpdater:
    def test_should_initialize_with_base_graph(self):
        updater = DynamicGraphUpdater(update_frequency='daily', window_size=30)

        # Create initial stock data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        initial_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 20 + ['GOOGL'] * 20,
            'close_price': [100 + i for i in range(20)] + [200 + i*2 for i in range(20)]
        })

        result = updater.initialize_graph(initial_data)

        assert result is True
        assert updater.current_graph is not None
        assert len(updater.current_graph.nodes()) >= 2

    def test_should_update_graph_with_new_data(self):
        updater = DynamicGraphUpdater(update_frequency='daily', window_size=30)

        # Initialize with base data
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(15)]
        initial_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 15 + ['GOOGL'] * 15,
            'close_price': [100 + i for i in range(15)] + [200 + i*2 for i in range(15)]
        })
        updater.initialize_graph(initial_data)

        # Add new data points
        new_dates = [datetime(2024, 1, 16) + timedelta(days=i) for i in range(5)]
        new_data = pd.DataFrame({
            'timestamp': new_dates * 2,
            'symbol': ['AAPL'] * 5 + ['GOOGL'] * 5,
            'close_price': [115 + i for i in range(5)] + [230 + i*2 for i in range(5)]
        })

        updated = updater.update_graph(new_data)

        assert updated is True
        assert updater.current_graph is not None

    def test_should_detect_structural_changes_in_graph(self):
        updater = DynamicGraphUpdater(update_frequency='daily', change_threshold=0.3)

        # Create data with changing correlation structure
        dates1 = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        dates2 = [datetime(2024, 1, 11) + timedelta(days=i) for i in range(10)]

        # Period 1: High correlation
        data1 = pd.DataFrame({
            'timestamp': dates1 * 2,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'close_price': [100 + i for i in range(10)] + [200 + i*2 for i in range(10)]  # Correlated
        })

        # Period 2: Low correlation
        data2 = pd.DataFrame({
            'timestamp': dates2 * 2,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'close_price': [110 + np.random.normal(0, 5) for _ in range(10)] +
                          [220 + np.random.normal(0, 10) for _ in range(10)]  # Less correlated
        })

        updater.initialize_graph(data1)
        changes = updater.detect_structural_changes(data2)

        assert changes is not None
        assert 'changes_detected' in changes
        assert 'changed_edges' in changes

    def test_should_maintain_rolling_window_of_data(self):
        updater = DynamicGraphUpdater(window_size=10)

        # Add data points sequentially
        for day in range(15):
            single_day_data = pd.DataFrame({
                'timestamp': [datetime(2024, 1, 1) + timedelta(days=day)] * 2,
                'symbol': ['AAPL', 'GOOGL'],
                'close_price': [100 + day, 200 + day*2]
            })

            if day == 0:
                updater.initialize_graph(single_day_data)
            else:
                updater.update_graph(single_day_data)

        # Check that only last 10 days are kept
        stored_data = updater.get_current_data_window()
        assert len(stored_data) <= 10 * 2  # 10 days * 2 symbols

    def test_should_trigger_updates_based_on_frequency(self):
        updater = DynamicGraphUpdater(update_frequency='weekly')

        # Initialize
        initial_date = datetime(2024, 1, 1)  # Monday
        initial_data = pd.DataFrame({
            'timestamp': [initial_date] * 2,
            'symbol': ['AAPL', 'GOOGL'],
            'close_price': [100, 200]
        })
        updater.initialize_graph(initial_data)

        # Add data for same week (should not trigger update)
        same_week_data = pd.DataFrame({
            'timestamp': [initial_date + timedelta(days=2)] * 2,  # Wednesday
            'symbol': ['AAPL', 'GOOGL'],
            'close_price': [102, 204]
        })

        should_update_same_week = updater.should_update(same_week_data)
        assert should_update_same_week is False

        # Add data for next week (should trigger update)
        next_week_data = pd.DataFrame({
            'timestamp': [initial_date + timedelta(days=7)] * 2,  # Next Monday
            'symbol': ['AAPL', 'GOOGL'],
            'close_price': [110, 220]
        })

        should_update_next_week = updater.should_update(next_week_data)
        assert should_update_next_week is True

    def test_should_handle_new_stocks_being_added(self):
        updater = DynamicGraphUpdater()

        # Initialize with 2 stocks
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        initial_data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'close_price': [100 + i for i in range(10)] + [200 + i*2 for i in range(10)]
        })
        updater.initialize_graph(initial_data)

        initial_nodes = len(updater.current_graph.nodes())

        # Add new stock
        new_stock_data = pd.DataFrame({
            'timestamp': dates * 3,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10 + ['MSFT'] * 10,
            'close_price': [100 + i for i in range(10)] +
                          [200 + i*2 for i in range(10)] +
                          [150 + i*1.5 for i in range(10)]
        })

        updater.update_graph(new_stock_data)

        # Should have one more node
        assert len(updater.current_graph.nodes()) > initial_nodes

    def test_should_calculate_graph_stability_metrics(self):
        updater = DynamicGraphUpdater()

        # Create two similar graphs
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        data = pd.DataFrame({
            'timestamp': dates * 2,
            'symbol': ['AAPL'] * 10 + ['GOOGL'] * 10,
            'close_price': [100 + i for i in range(10)] + [200 + i*2 for i in range(10)]
        })

        updater.initialize_graph(data)
        stability = updater.calculate_stability_metrics()

        assert stability is not None
        assert 'edge_stability' in stability
        assert 'node_stability' in stability
        assert isinstance(stability['edge_stability'], float)
        assert isinstance(stability['node_stability'], float)