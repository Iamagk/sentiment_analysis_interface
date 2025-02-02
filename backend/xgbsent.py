import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import matplotlib.pyplot as plt
import yfinance as yf
import xgboost as xgb
import os
from typing import List, Dict, Any

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.features = [
            "Close_lag1", "High_lag1", "Low_lag1", "Volume_lag1",
            "RSI", "SMA_20", "EMA_20", "Sentiment",
            "Close_rolling_7", "Close_std_7", "Volume_rolling_7",
            "Close_rolling_14", "Close_std_14",
            "MACD"
        ]

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add lag features
        df['Close_lag1'] = df['Close'].shift(1)
        df['High_lag1'] = df['High'].shift(1)
        df['Low_lag1'] = df['Low'].shift(1)
        df['Volume_lag1'] = df['Volume'].shift(1)

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()

        # Rolling features
        df['Close_rolling_7'] = df['Close'].rolling(window=7).mean()
        df['Close_std_7'] = df['Close'].rolling(window=7).std()
        df['Volume_rolling_7'] = df['Volume'].rolling(window=7).mean()
        df['Close_rolling_14'] = df['Close'].rolling(window=14).mean()
        df['Close_std_14'] = df['Close'].rolling(window=14).std()

        return df

    def enhance_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sentiment_ma3'] = df['sentiment_polarity'].rolling(window=3).mean()
        df['sentiment_ma7'] = df['sentiment_polarity'].rolling(window=7).mean()
        df['sentiment_momentum'] = df['sentiment_polarity'] - df['sentiment_polarity'].shift(1)
        df['sentiment_volatility'] = df['sentiment_polarity'].rolling(window=7).std()
        df['sentiment_lag1'] = df['sentiment_polarity'].shift(1)
        df['sentiment_lag3'] = df['sentiment_polarity'].shift(3)
        df['high_sentiment'] = (df['sentiment_polarity'] > 0.8).astype(int)
        df['low_sentiment'] = (df['sentiment_polarity'] < -0.8).astype(int)
        df['sentiment_volume_interaction'] = df['sentiment_polarity'] * df['Volume']
        df['sentiment_divergence'] = df['sentiment_polarity'] - df['sentiment_ma7']
        return df

    def prepare_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(stock_data)
        df.index = df.index.date
        sentiment_data.index = sentiment_data.index.date
        df['Sentiment'] = sentiment_data['avg_sentiment']
        df['Next_Close'] = df['Close'].shift(-1)
        return df.dropna()

    def train(self, X: pd.DataFrame, y: pd.Series):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = xgb.train(params, dtrain, num_boost_round=1000)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "mse": round(mean_squared_error(y_true, y_pred), 4),
            "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
            "r2": round(r2_score(y_true, y_pred), 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),  # Convert to percentage
            "explained_variance": round(explained_variance_score(y_true, y_pred), 4)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        importance = self.model.get_score(importance_type='weight')
        total_importance = sum(importance.values())
        df = pd.DataFrame({
            'feature': importance.keys(),
            'importance': [(v / total_importance) * 100 for v in importance.values()]
        })
        df['importance'] = df['importance'].round(2)
        return df.sort_values('importance', ascending=False)

    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, ticker: str, start_date: str, end_date: str) -> str:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual Next Day Close', alpha=0.8)
        plt.plot(y_true.index, y_pred, label='Predicted Next Day Close', alpha=0.8)
        plt.title(f'{ticker} Next Day Stock Price Prediction (XGBoost)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        image_filename = f'next_day_stock_prediction_plot_xgb_{ticker.lower()}_{start_date.replace("-", "")}_{end_date.replace("-", "")}.png'
        image_path = os.path.join('outputs', image_filename)
        plt.savefig(image_path)
        # Additional save location
        frontend_image_path = os.path.join('../frontend/outputs', image_filename)
        os.makedirs(os.path.dirname(frontend_image_path), exist_ok=True)
        plt.savefig(frontend_image_path)
        plt.close()
        return image_path

def load_and_prepare_data(ticker: str, start_date: str, end_date: str, sentiment_files: List[str]) -> Dict[str, Any]:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.index = stock_data.index.date

    sentiment_data = pd.concat([
        pd.read_csv(file, parse_dates=['date'], index_col='date')
        for file in sentiment_files
    ])
    sentiment_data.index = sentiment_data.index.date
    sentiment_data = sentiment_data.sort_index()

    return {"stock_data": stock_data, "sentiment_data": sentiment_data, "ticker": ticker, "start_date": start_date, "end_date": end_date}

def run_xgboost_model(data: Dict[str, Any]) -> Dict[str, Any]:
    model = XGBoostModel()
    df = model.prepare_data(data['stock_data'], data['sentiment_data'])

    split_date = df.index[-int(len(df) * 0.2)]
    train_data = df[df.index < split_date]
    test_data = df[df.index >= split_date]

    X_train = train_data[model.features]
    y_train = train_data['Next_Close']
    X_test = test_data[model.features]
    y_test = test_data['Next_Close']

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
