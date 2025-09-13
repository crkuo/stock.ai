from typing import Dict, Any, List
from datetime import datetime


class ValidationResult:
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


class OutlierResult:
    def __init__(self, has_outliers: bool, outlier_fields: List[str] = None):
        self.has_outliers = has_outliers
        self.outlier_fields = outlier_fields or []


class DataValidator:
    def __init__(self):
        self.required_fields = ["timestamp", "symbol", "open_price", "close_price", "high_price", "low_price", "volume"]

    def validate_stock_data(self, stock_data: Dict[str, Any]) -> ValidationResult:
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in stock_data:
                errors.append(f"Missing required field: {field}")

        # Check price relationships if all price fields exist
        if all(field in stock_data for field in ["open_price", "close_price", "high_price", "low_price"]):
            high = stock_data["high_price"]
            low = stock_data["low_price"]
            open_price = stock_data["open_price"]
            close_price = stock_data["close_price"]

            if high < max(open_price, close_price) or low > min(open_price, close_price):
                errors.append("Invalid price relationships: high/low prices inconsistent with open/close")

        # Check negative volume
        if "volume" in stock_data and stock_data["volume"] < 0:
            errors.append("volume cannot be negative")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_for_outliers(self, stock_data: Dict[str, Any]) -> OutlierResult:
        outlier_fields = []

        # Simple outlier detection: 10x price change
        if "open_price" in stock_data and "close_price" in stock_data:
            open_price = stock_data["open_price"]
            close_price = stock_data["close_price"]
            price_change_ratio = abs(close_price - open_price) / open_price

            if price_change_ratio >= 9.0:  # 900% change or more
                outlier_fields.append("close_price")

        return OutlierResult(has_outliers=len(outlier_fields) > 0, outlier_fields=outlier_fields)