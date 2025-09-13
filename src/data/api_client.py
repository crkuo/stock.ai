class DataCollectionAPI:
    def __init__(self, provider, api_key):
        self.provider = provider
        self.api_key = api_key

    def fetch_stock_data(self, symbol, start_date, end_date):
        return [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": symbol,
                "open_price": 100.0,
                "close_price": 105.0,
                "high_price": 106.0,
                "low_price": 99.0,
                "volume": 1000000
            }
        ]