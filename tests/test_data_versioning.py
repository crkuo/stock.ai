import pytest
import os
import tempfile
from pathlib import Path
from src.data.data_versioning import DataVersionManager


class TestDataVersionManager:
    def test_should_initialize_dvc_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)

            result = manager.init_dvc()

            assert result is True
            assert (Path(temp_dir) / ".dvc").exists()

    def test_should_add_data_file_to_dvc_tracking(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            # Create a test data file
            data_file = Path(temp_dir) / "test_data.csv"
            data_file.write_text("timestamp,symbol,price\n2024-01-01,AAPL,100.0")

            result = manager.add_data_file(str(data_file))

            assert result is True
            assert (Path(temp_dir) / "test_data.csv.dvc").exists()

    def test_should_commit_data_version_with_message(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            # Create and add a test data file
            data_file = Path(temp_dir) / "test_data.csv"
            data_file.write_text("timestamp,symbol,price\n2024-01-01,AAPL,100.0")
            manager.add_data_file(str(data_file))

            result = manager.commit_version("Initial data version")

            assert result is True

    def test_should_create_data_version_tag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            result = manager.create_tag("v1.0.0", "First stable data version")

            assert result is True

    def test_should_list_data_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            versions = manager.list_versions()

            assert isinstance(versions, list)
            assert len(versions) >= 0

    def test_should_checkout_specific_data_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            result = manager.checkout_version("v1.0.0")

            assert result is True

    def test_should_get_data_file_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(repo_path=temp_dir)
            manager.init_dvc()

            # Create a test data file
            data_file = Path(temp_dir) / "test_data.csv"
            data_file.write_text("timestamp,symbol,price\n2024-01-01,AAPL,100.0")

            status = manager.get_status(str(data_file))

            assert status is not None
            assert "tracked" in status or "untracked" in status