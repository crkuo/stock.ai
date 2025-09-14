import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
from src.graph.graph_persistence import GraphPersistenceManager


class TestGraphPersistenceManager:
    def test_should_save_and_load_graph_snapshot(self):
        persistence_manager = GraphPersistenceManager()

        # Create sample graph data
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.6],
            'GOOGL': [0.8, 1.0, 0.7],
            'MSFT': [0.6, 0.7, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        metadata = {
            'timestamp': datetime(2024, 1, 1),
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'correlation_threshold': 0.5,
            'data_window_days': 30
        }

        # Save graph snapshot
        version_id = persistence_manager.save_graph_snapshot(
            correlation_matrix=correlation_matrix,
            metadata=metadata,
            version_name="test_snapshot_v1"
        )

        assert version_id is not None

        # Load graph snapshot
        loaded_data = persistence_manager.load_graph_snapshot(version_id)

        assert loaded_data is not None
        assert 'correlation_matrix' in loaded_data
        assert 'metadata' in loaded_data
        pd.testing.assert_frame_equal(loaded_data['correlation_matrix'], correlation_matrix)
        assert loaded_data['metadata']['symbols'] == ['AAPL', 'GOOGL', 'MSFT']

    def test_should_maintain_graph_version_history(self):
        persistence_manager = GraphPersistenceManager()

        # Create multiple versions of graph snapshots
        versions = []
        for i in range(3):
            correlation_matrix = pd.DataFrame({
                'AAPL': [1.0, 0.8 + i*0.05, 0.6 + i*0.1],
                'GOOGL': [0.8 + i*0.05, 1.0, 0.7 + i*0.05],
                'MSFT': [0.6 + i*0.1, 0.7 + i*0.05, 1.0]
            }, index=['AAPL', 'GOOGL', 'MSFT'])

            metadata = {
                'timestamp': datetime(2024, 1, 1) + timedelta(days=i),
                'version': i + 1,
                'data_window_days': 30
            }

            version_id = persistence_manager.save_graph_snapshot(
                correlation_matrix=correlation_matrix,
                metadata=metadata,
                version_name=f"test_v{i+1}"
            )
            versions.append(version_id)

        # Get version history
        history = persistence_manager.get_version_history()

        assert len(history) >= 3
        assert all(v in [h['version_id'] for h in history] for v in versions)

    def test_should_compare_graph_versions(self):
        persistence_manager = GraphPersistenceManager()

        # Create two different graph versions
        matrix_v1 = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.6],
            'GOOGL': [0.8, 1.0, 0.7],
            'MSFT': [0.6, 0.7, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        matrix_v2 = pd.DataFrame({
            'AAPL': [1.0, 0.9, 0.5],  # Changed correlations
            'GOOGL': [0.9, 1.0, 0.8],
            'MSFT': [0.5, 0.8, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        version_1 = persistence_manager.save_graph_snapshot(
            correlation_matrix=matrix_v1,
            metadata={'name': 'version_1'},
            version_name="v1"
        )

        version_2 = persistence_manager.save_graph_snapshot(
            correlation_matrix=matrix_v2,
            metadata={'name': 'version_2'},
            version_name="v2"
        )

        # Compare versions
        comparison = persistence_manager.compare_versions(version_1, version_2)

        assert comparison is not None
        assert 'changes' in comparison
        assert 'edge_differences' in comparison
        assert 'statistics' in comparison

    def test_should_export_and_import_graph_data(self):
        persistence_manager = GraphPersistenceManager()

        # Create graph data
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.6],
            'GOOGL': [0.8, 1.0, 0.7],
            'MSFT': [0.6, 0.7, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        metadata = {
            'created_at': datetime(2024, 1, 1),
            'format_version': '1.0'
        }

        version_id = persistence_manager.save_graph_snapshot(
            correlation_matrix=correlation_matrix,
            metadata=metadata,
            version_name="export_test"
        )

        # Export to file
        export_path = "/tmp/test_graph_export.json"
        success = persistence_manager.export_graph(version_id, export_path)
        assert success is True
        assert os.path.exists(export_path)

        # Import from file
        imported_version_id = persistence_manager.import_graph(
            import_path=export_path,
            version_name="imported_test"
        )

        assert imported_version_id is not None

        # Verify imported data matches original
        original_data = persistence_manager.load_graph_snapshot(version_id)
        imported_data = persistence_manager.load_graph_snapshot(imported_version_id)

        pd.testing.assert_frame_equal(
            original_data['correlation_matrix'],
            imported_data['correlation_matrix']
        )

    def test_should_clean_up_old_versions(self):
        persistence_manager = GraphPersistenceManager()

        # Create multiple old versions
        old_versions = []
        for i in range(5):
            correlation_matrix = pd.DataFrame({
                'A': [1.0, 0.5],
                'B': [0.5, 1.0]
            }, index=['A', 'B'])

            metadata = {
                'timestamp': datetime(2024, 1, 1) - timedelta(days=30 + i),
                'temporary': True
            }

            version_id = persistence_manager.save_graph_snapshot(
                correlation_matrix=correlation_matrix,
                metadata=metadata,
                version_name=f"old_v{i}"
            )
            old_versions.append(version_id)

        # Clean up versions older than 25 days
        cleanup_result = persistence_manager.cleanup_old_versions(
            days_to_keep=25,
            keep_tagged_versions=False
        )

        assert cleanup_result is not None
        assert 'deleted_count' in cleanup_result
        assert cleanup_result['deleted_count'] > 0

    def test_should_handle_graph_backup_and_restore(self):
        persistence_manager = GraphPersistenceManager()

        # Create graph data
        correlation_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8],
            'GOOGL': [0.8, 1.0]
        }, index=['AAPL', 'GOOGL'])

        version_id = persistence_manager.save_graph_snapshot(
            correlation_matrix=correlation_matrix,
            metadata={'backup_test': True},
            version_name="backup_test"
        )

        # Create backup
        backup_path = "/tmp/graph_backup.zip"
        backup_result = persistence_manager.create_backup(
            backup_path=backup_path,
            include_versions=[version_id]
        )

        assert backup_result is True
        assert os.path.exists(backup_path)

        # Simulate data loss by deleting version
        persistence_manager.delete_version(version_id)

        # Restore from backup
        restore_result = persistence_manager.restore_from_backup(backup_path)

        assert restore_result is not None
        assert 'restored_versions' in restore_result
        assert len(restore_result['restored_versions']) > 0

    def test_should_validate_graph_data_integrity(self):
        persistence_manager = GraphPersistenceManager()

        # Create graph with valid data
        valid_matrix = pd.DataFrame({
            'AAPL': [1.0, 0.8, 0.6],
            'GOOGL': [0.8, 1.0, 0.7],
            'MSFT': [0.6, 0.7, 1.0]
        }, index=['AAPL', 'GOOGL', 'MSFT'])

        version_id = persistence_manager.save_graph_snapshot(
            correlation_matrix=valid_matrix,
            metadata={'validation_test': True},
            version_name="validation_test"
        )

        # Validate data integrity
        validation_result = persistence_manager.validate_graph_integrity(version_id)

        assert validation_result is not None
        assert 'is_valid' in validation_result
        assert validation_result['is_valid'] is True
        assert 'checks_passed' in validation_result