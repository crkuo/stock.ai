import pytest
from src.data.database import DatabaseManager


def test_database_manager_can_be_instantiated():
    """Test that DatabaseManager can be created"""
    db_manager = DatabaseManager()
    assert db_manager is not None


def test_database_manager_has_connect_method():
    """Test that DatabaseManager has a connect method"""
    db_manager = DatabaseManager()
    assert hasattr(db_manager, 'connect')