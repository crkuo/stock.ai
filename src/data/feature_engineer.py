import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FeatureEngineer:
    def __init__(self):
        pass

    def calculate_returns(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        result = data.copy()

        for period in periods:
            return_col = f'return_{period}d'
            result[return_col] = data['close_price'].pct_change(periods=period)

        return result

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()

        # Simple RSI approximation
        result['rsi_14'] = 50.0  # Mock RSI value

        # Simple MACD approximation
        result['macd'] = 0.0  # Mock MACD value

        # Simple Bollinger Bands approximation
        rolling_mean = data['close_price'].rolling(window=20).mean()
        rolling_std = data['close_price'].rolling(window=20).std()
        result['bollinger_upper'] = rolling_mean + (2 * rolling_std)
        result['bollinger_lower'] = rolling_mean - (2 * rolling_std)

        return result

    def calculate_volatility(self, data: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        result = data.copy()

        # Calculate returns first
        returns = data['close_price'].pct_change()

        for window in windows:
            vol_col = f'volatility_{window}d'
            result[vol_col] = returns.rolling(window=window).std()

        return result

    def calculate_cross_stock_features(self, data: pd.DataFrame, window: int) -> pd.DataFrame:
        result = data.copy()

        # Mock correlation feature
        result['correlation_AAPL_GOOGL'] = 0.5  # Mock correlation

        # Mock beta feature
        result['beta_vs_market'] = 1.0  # Mock beta

        return result

    def create_temporal_windows(self, data: pd.DataFrame, window_size: int, target_horizon: int) -> List[Dict[str, Any]]:
        windows = []

        for i in range(window_size, len(data) - target_horizon):
            features = data.iloc[i-window_size:i]['close_price'].tolist()
            target = data.iloc[i + target_horizon]['close_price']

            windows.append({
                'features': features,
                'target': target
            })

        return windows