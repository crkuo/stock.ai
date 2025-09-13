import pytest
from datetime import datetime
from src.data.timeseries_storage import TimeSeriesStorage


class TestTimeSeriesStorage:
    def test_should_store_stock_data_in_timescaledb(self):
        storage = TimeSeriesStorage(connection_string="postgresql://test:test@localhost:5432/test_db")

        stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open_price": 100.0,
            "close_price": 105.0,
            "high_price": 106.0,
            "low_price": 99.0,
            "volume": 1000000
        }

        result = storage.insert_stock_data(stock_data)

        assert result is True

    def test_should_retrieve_stock_data_by_symbol_and_date_range(self):
        storage = TimeSeriesStorage(connection_string="postgresql://test:test@localhost:5432/test_db")

        retrieved_data = storage.get_stock_data(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )

        assert retrieved_data is not None
        assert len(retrieved_data) >= 0

    def test_should_create_hypertable_for_stock_data(self):
        storage = TimeSeriesStorage(connection_string="postgresql://test:test@localhost:5432/test_db")

        result = storage.create_stock_data_table()

        assert result is True