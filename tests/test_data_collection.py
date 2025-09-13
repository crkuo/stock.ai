import pytest
from src.data.collector import StockDataCollector


def test_stock_data_collector_can_be_instantiated():
    """Test that StockDataCollector can be created"""
    collector = StockDataCollector()
    assert collector is not None


def test_stock_data_collector_has_fetch_method():
    """Test that StockDataCollector has a fetch method"""
    collector = StockDataCollector()
    assert hasattr(collector, 'fetch_stock_data')