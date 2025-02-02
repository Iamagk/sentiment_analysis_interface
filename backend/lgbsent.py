import yfinance as yf
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import ta

class LightGBMModel:
    def __init__(self):
        self.model = None
        self.features = [
            "Close_lag1", "High_lag1", "Low_lag1", "Volume_lag1",
            "RSI", "MACD", "SMA20", "EMA20", "ATR", "Sentiment",
        ]

    def ensure_1d(self, series):
        if isinstance(series, pd.DataFrame):
            return series.squeeze()
        return series

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure all columns are 1D
        for col in df.columns:
            df[col] = self.ensure_1d(df[col])

        # Add lag features
        df['Close_lag1'] = df['Close'].shift(1)
        df['High_lag1'] = df['High'].shift(1)
        df['Low_lag1'] = df['Low'].shift(1)
        df['Volume_lag1'] = df['Volume'].shift(1)

        # Add technical indicators
        df['RSI'] = ta.momentum.rsi(self.ensure_1d(df['Close']), window=14)
        df['SMA20'] = ta.trend.sma_indicator(self.ensure_1d(df['Close']), window=20)
        df['EMA20'] = ta.trend.ema_indicator(self.ensure_1d(df['Close']), window=20)
        df['MACD'] = ta.trend.macd_diff(self.ensure_1d(df['Close']))
        df['ATR'] = ta.volatility.average_true_range(self.ensure_1d(df['High']), self.ensure_1d(df['Low']), self.ensure_1d(df['Close']))

        return df

    def prepare_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(stock_data)
        df['Sentiment'] = sentiment_data['avg_sentiment']
        df['Target'] = df['Close'].shift(-1)
        df = df.dropna()
        return df

    def train(self, X: pd.DataFrame, y: pd.Series):
        # Ensure feature names are valid for LightGBM
        X.columns = [col[0] if isinstance(col, tuple) else col for col in X.columns]
        X.columns = [str(col).replace('_', '') for col in X.columns]
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = lgb.train(params, train_data, num_boost_round=1000)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Ensure feature names are valid for LightGBM
        X.columns = [col[0] if isinstance(col, tuple) else col for col in X.columns]
        X.columns = [str(col).replace('_', '') for col in X.columns]
        return self.model.predict(X)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance = self.model.feature_importance()
        feature_names = [col[0] if isinstance(col, tuple) else col for col in self.features]
        feature_names = [str(col).replace('_', '') for col in feature_names]
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, ticker: str, start_date: str, end_date: str) -> str:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual Next Day Close', alpha=0.8)
        plt.plot(y_true.index, y_pred, label='Predicted Next Day Close', alpha=0.8)
        plt.title(f'{ticker} Next Day Stock Price Prediction (LightGBM)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        image_filename = f'next_day_stock_prediction_plot_lgb_{ticker.lower()}_{start_date.replace("-", "")}_{end_date.replace("-", "")}.png'
        image_path = os.path.join('outputs', image_filename)
        plt.savefig(image_path)
        frontend_image_path = os.path.join('../frontend/outputs', image_filename)
        os.makedirs(os.path.dirname(frontend_image_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(frontend_image_path)
        plt.close()
        return image_path

def load_and_prepare_data(ticker: str, start_date: str, end_date: str, sentiment_files: List[str]) -> Dict[str, Any]:
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    sentiment_data = pd.concat([
        pd.read_csv(file, parse_dates=['date'], index_col='date')
        for file in sentiment_files
    ])
    sentiment_data = sentiment_data.sort_index()

    return {"stock_data": stock_data, "sentiment_data": sentiment_data, "ticker": ticker, "start_date": start_date, "end_date": end_date}

def run_lightgbm_model(data: Dict[str, Any]) -> Dict[str, Any]:
    model = LightGBMModel()
    df = model.prepare_data(data['stock_data'], data['sentiment_data'])

    # Split the data
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]

    X_train = train_data[model.features]
    y_train = train_data['Target']
    X_test = test_data[model.features]
    y_test = test_data['Target']

    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = model.evaluate(y_test, y_pred)
    feature_importance = model.get_feature_importance()

    plot_path = model.plot_predictions(y_test, y_pred, data['ticker'], data['start_date'], data['end_date'])

    return {
        "metrics": metrics,
        "feature_importance": feature_importance.to_dict(orient='records'),
        "plot_path": plot_path,
        "predictions": y_pred.tolist(),
        "actual_values": y_test.tolist(),
        "dates": [date.strftime('%Y-%m-%d') for date in y_test.index]
    }
