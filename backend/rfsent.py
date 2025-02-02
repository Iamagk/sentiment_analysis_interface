import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import matplotlib.pyplot as plt
import yfinance as yf
import os
from typing import List, Dict, Any

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD',
                         'Momentum', 'Volatility', 'EMA12', 'EMA26', 'ATR', 'Sentiment']

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(4)

        # Volatility
        df['Volatility'] = df['Close'].rolling(window=5).std()

        # EMAs
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        return df

    def prepare_data(self, stock_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        df = self.add_technical_indicators(stock_data)
        df['Sentiment'] = df.index.map(lambda x: sentiment_data.loc[x, 'avg_sentiment']
                                       if x in sentiment_data.index else 0)
        df['Next_Close'] = df['Close'].shift(-1)
        return df.dropna()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

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
        return pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


    def plot_predictions(self, y_true: pd.Series, y_pred: np.ndarray, ticker: str, start_date: str, end_date: str) -> str:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true.index, y_true.values, label='Actual Next Day Close', alpha=0.8)
        plt.plot(y_true.index, y_pred, label='Predicted Next Day Close', alpha=0.8)
        plt.title(f'{ticker} Next Day Stock Price Prediction (Random Forest)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        image_filename = f'next_day_stock_prediction_plot_rf_{ticker.lower()}_{start_date.replace("-", "")}_{end_date.replace("-", "")}.png'
        image_path = os.path.join('outputs', image_filename)
        plt.savefig(image_path)
        frontend_image_path = os.path.join('../frontend/outputs', image_filename)
        os.makedirs(os.path.dirname(frontend_image_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(frontend_image_path)
        plt.close()
        return image_path

def load_and_prepare_data(ticker: str, start_date: str, end_date: str, sentiment_files: List[str]) -> Dict[str, Any]:
    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Load sentiment data
    sentiment_data = pd.concat([pd.read_csv(file, parse_dates=['date'], index_col='date')
                                for file in sentiment_files])
    sentiment_data = sentiment_data.sort_index()

    return {"stock_data": stock_data, "sentiment_data": sentiment_data}

def run_random_forest_model(data: Dict[str, Any]) -> Dict[str, Any]:
    model = RandomForestModel()
    df = model.prepare_data(data['stock_data'], data['sentiment_data'])

    # Split data
    split_date = df.index[-int(len(df) * 0.2)]
    train_data = df[df.index < split_date]
    test_data = df[df.index >= split_date]

    X_train = train_data[model.features]
    y_train = train_data['Next_Close']
    X_test = test_data[model.features]
    y_test = test_data['Next_Close']

    # Train and predict
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = model.evaluate(y_test, y_pred)
    # Convert feature importance to list of dictionaries
    feature_importance_df = model.get_feature_importance()
    feature_importance = [
        {"feature": row["feature"], "importance": float(row["importance"])}
        for _, row in feature_importance_df.iterrows()
    ]

    # Plot
    plot_path = model.plot_predictions(y_test, y_pred, data['ticker'], data['start_date'], data['end_date'])

    return {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "plot_path": plot_path,
        "predictions": y_pred.tolist(),
        "actual_values": y_test.tolist(),
        "dates": y_test.index.strftime('%Y-%m-%d').tolist()
    }