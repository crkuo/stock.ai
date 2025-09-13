import pytest
import pandas as pd
from datetime import datetime
from src.data.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    def test_should_calculate_price_returns(self):
        engineer = FeatureEngineer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'close_price': [100.0, 105.0, 102.0]
        })

        result = engineer.calculate_returns(stock_data, periods=[1, 2])

        assert 'return_1d' in result.columns
        assert 'return_2d' in result.columns
        assert abs(result['return_1d'].iloc[1] - 0.05) < 1e-10  # 5% return

    def test_should_calculate_technical_indicators(self):
        engineer = FeatureEngineer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, i) for i in range(1, 21)],
            'symbol': ['AAPL'] * 20,
            'close_price': [100 + i for i in range(20)],
            'high_price': [102 + i for i in range(20)],
            'low_price': [98 + i for i in range(20)],
            'volume': [1000000] * 20
        })

        result = engineer.calculate_technical_indicators(stock_data)

        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'bollinger_upper' in result.columns
        assert 'bollinger_lower' in result.columns

    def test_should_calculate_volatility_features(self):
        engineer = FeatureEngineer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, i) for i in range(1, 11)],
            'symbol': ['AAPL'] * 10,
            'close_price': [100, 105, 98, 103, 99, 107, 95, 102, 104, 101]
        })

        result = engineer.calculate_volatility(stock_data, windows=[5, 10])

        assert 'volatility_5d' in result.columns
        assert 'volatility_10d' in result.columns
        assert result['volatility_5d'].notna().any()

    def test_should_calculate_cross_stock_features(self):
        engineer = FeatureEngineer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 1),
                         datetime(2024, 1, 2), datetime(2024, 1, 2)],
            'symbol': ['AAPL', 'GOOGL', 'AAPL', 'GOOGL'],
            'close_price': [100.0, 200.0, 105.0, 210.0]
        })

        result = engineer.calculate_cross_stock_features(stock_data, window=2)

        assert 'correlation_AAPL_GOOGL' in result.columns
        assert 'beta_vs_market' in result.columns

    def test_should_create_temporal_windows(self):
        engineer = FeatureEngineer()

        stock_data = pd.DataFrame({
            'timestamp': [datetime(2024, 1, i) for i in range(1, 11)],
            'symbol': ['AAPL'] * 10,
            'close_price': [100 + i for i in range(10)]
        })

        result = engineer.create_temporal_windows(stock_data, window_size=3, target_horizon=1)

        assert len(result) > 0
        assert 'features' in result[0]
        assert 'target' in result[0]
        assert len(result[0]['features']) == 3  # window_size