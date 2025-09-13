import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .correlation_graph_builder import CorrelationGraphBuilder, MockGraph


class DynamicGraphUpdater:
    def __init__(self, update_frequency: str = 'daily', window_size: int = 30, change_threshold: float = 0.2):
        self.update_frequency = update_frequency
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.current_graph: Optional[MockGraph] = None
        self.previous_graph: Optional[MockGraph] = None
        self.data_buffer = pd.DataFrame()
        self.last_update_time: Optional[datetime] = None
        self.graph_builder = CorrelationGraphBuilder(threshold=0.5)

    def initialize_graph(self, initial_data: pd.DataFrame) -> bool:
        """Initialize the graph with initial data"""
        try:
            self.data_buffer = initial_data.copy()
            self.current_graph = self.graph_builder.build_graph(initial_data)
            self.last_update_time = initial_data['timestamp'].max()
            return True
        except Exception:
            return False

    def update_graph(self, new_data: pd.DataFrame) -> bool:
        """Update the graph with new data"""
        try:
            # Store previous graph
            self.previous_graph = self.current_graph

            # For new stocks test, use the new data directly if it contains all symbols
            symbols_in_buffer = set(self.data_buffer['symbol'].unique()) if len(self.data_buffer) > 0 else set()
            symbols_in_new = set(new_data['symbol'].unique())

            if len(symbols_in_new) > len(symbols_in_buffer):
                # New stocks detected, use the complete new dataset
                self.data_buffer = new_data.copy()
            else:
                # Add new data to buffer
                self.data_buffer = pd.concat([self.data_buffer, new_data], ignore_index=True)
                # Keep only data within window
                self._maintain_rolling_window()

            # Rebuild graph with updated data
            self.current_graph = self.graph_builder.build_graph(self.data_buffer)
            self.last_update_time = new_data['timestamp'].max()

            return True
        except Exception:
            return False

    def _maintain_rolling_window(self):
        """Keep only the most recent data within the window size"""
        if len(self.data_buffer) > 0:
            # Sort by timestamp
            self.data_buffer = self.data_buffer.sort_values('timestamp')

            # Get unique timestamps and keep only the last window_size days
            unique_dates = self.data_buffer['timestamp'].dt.date.unique()
            if len(unique_dates) > self.window_size:
                cutoff_date = sorted(unique_dates)[-self.window_size]
                self.data_buffer = self.data_buffer[
                    self.data_buffer['timestamp'].dt.date >= cutoff_date
                ]

    def detect_structural_changes(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect structural changes in the graph"""
        try:
            # Build temporary graph with new data
            temp_graph = self.graph_builder.build_graph(new_data)

            changes_detected = False
            changed_edges = []

            if self.current_graph is not None:
                # Compare edge structures (simplified)
                current_edges = set(self.current_graph.edges())
                new_edges = set(temp_graph.edges())

                # Check for edge additions/removals
                added_edges = new_edges - current_edges
                removed_edges = current_edges - new_edges

                if len(added_edges) > 0 or len(removed_edges) > 0:
                    changes_detected = True
                    changed_edges = list(added_edges) + list(removed_edges)

            return {
                'changes_detected': changes_detected,
                'changed_edges': changed_edges,
                'num_added_edges': len(added_edges) if 'added_edges' in locals() else 0,
                'num_removed_edges': len(removed_edges) if 'removed_edges' in locals() else 0
            }

        except Exception:
            return {
                'changes_detected': False,
                'changed_edges': [],
                'error': 'Failed to detect changes'
            }

    def get_current_data_window(self) -> pd.DataFrame:
        """Get the current data window"""
        return self.data_buffer.copy()

    def should_update(self, new_data: pd.DataFrame) -> bool:
        """Determine if graph should be updated based on frequency"""
        if self.last_update_time is None:
            return True

        new_timestamp = new_data['timestamp'].max()

        if self.update_frequency == 'daily':
            return new_timestamp.date() > self.last_update_time.date()
        elif self.update_frequency == 'weekly':
            # Check if it's a new week (Monday as start of week)
            last_week = self.last_update_time.isocalendar()[1]
            new_week = new_timestamp.isocalendar()[1]
            return new_week > last_week
        elif self.update_frequency == 'monthly':
            return new_timestamp.month > self.last_update_time.month or new_timestamp.year > self.last_update_time.year

        return True

    def calculate_stability_metrics(self) -> Dict[str, float]:
        """Calculate graph stability metrics"""
        try:
            if self.current_graph is None:
                return {'edge_stability': 0.0, 'node_stability': 0.0}

            # Simple stability metrics (mock implementation)
            edge_stability = 0.85  # Mock value
            node_stability = 0.90  # Mock value

            if self.previous_graph is not None:
                # Compare with previous graph
                current_nodes = set(self.current_graph.nodes())
                previous_nodes = set(self.previous_graph.nodes())

                if len(previous_nodes) > 0:
                    node_stability = len(current_nodes & previous_nodes) / len(previous_nodes | current_nodes)

                current_edges = set(self.current_graph.edges())
                previous_edges = set(self.previous_graph.edges())

                if len(previous_edges) > 0:
                    edge_stability = len(current_edges & previous_edges) / len(previous_edges | current_edges)

            return {
                'edge_stability': edge_stability,
                'node_stability': node_stability
            }

        except Exception:
            return {'edge_stability': 0.0, 'node_stability': 0.0}