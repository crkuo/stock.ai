import pytest
from unittest.mock import Mock, patch
from src.data.api_client import DataCollectionAPI


class TestDataCollectionAPI:
    def test_should_fetch_stock_data_from_alpha_vantage(self):
        api_client = DataCollectionAPI(provider="alpha_vantage", api_key="test_key")

        stock_data = api_client.fetch_stock_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")

        assert stock_data is not None
        assert len(stock_data) > 0
        assert "timestamp" in stock_data[0]
        assert "symbol" in stock_data[0]
        assert "open_price" in stock_data[0]
        assert "close_price" in stock_data[0]
        assert "high_price" in stock_data[0]
        assert "low_price" in stock_data[0]
        assert "volume" in stock_data[0]
        assert stock_data[0]["symbol"] == "AAPL"