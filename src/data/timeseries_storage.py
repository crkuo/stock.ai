from datetime import datetime
from typing import List, Dict, Any


class TimeSeriesStorage:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def insert_stock_data(self, stock_data: Dict[str, Any]) -> bool:
        return True

    def get_stock_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return []

    def create_stock_data_table(self) -> bool:
        return True