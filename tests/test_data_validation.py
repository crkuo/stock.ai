import pytest
from datetime import datetime
from src.data.data_validator import DataValidator, ValidationResult


class TestDataValidator:
    def test_should_validate_stock_data_schema(self):
        validator = DataValidator()

        valid_stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open_price": 100.0,
            "close_price": 105.0,
            "high_price": 106.0,
            "low_price": 99.0,
            "volume": 1000000
        }

        result = validator.validate_stock_data(valid_stock_data)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_should_detect_missing_required_fields(self):
        validator = DataValidator()

        invalid_stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            # Missing required price fields
            "volume": 1000000
        }

        result = validator.validate_stock_data(invalid_stock_data)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "open_price" in str(result.errors)

    def test_should_detect_invalid_price_relationships(self):
        validator = DataValidator()

        invalid_stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open_price": 100.0,
            "close_price": 105.0,
            "high_price": 98.0,  # High price lower than open/close
            "low_price": 107.0,  # Low price higher than high
            "volume": 1000000
        }

        result = validator.validate_stock_data(invalid_stock_data)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_should_detect_negative_volume(self):
        validator = DataValidator()

        invalid_stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open_price": 100.0,
            "close_price": 105.0,
            "high_price": 106.0,
            "low_price": 99.0,
            "volume": -1000  # Negative volume
        }

        result = validator.validate_stock_data(invalid_stock_data)

        assert result.is_valid is False
        assert "volume" in str(result.errors)

    def test_should_detect_outliers_in_stock_data(self):
        validator = DataValidator()

        outlier_stock_data = {
            "timestamp": datetime(2024, 1, 1, 9, 30),
            "symbol": "AAPL",
            "open_price": 100.0,
            "close_price": 1000.0,  # 10x price jump (outlier)
            "high_price": 1001.0,
            "low_price": 99.0,
            "volume": 1000000
        }

        result = validator.validate_for_outliers(outlier_stock_data)

        assert result.has_outliers is True
        assert len(result.outlier_fields) > 0