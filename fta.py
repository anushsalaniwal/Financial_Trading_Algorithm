import yfinance as yf
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_indicators(df):
    df['RSI'] = calculate_rsi(df)
    df = calculate_stoch_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_macd(df)
    df = calculate_atr(df)
    df = generate_momentum_signals(df)
    return df

def calculate_rsi(df, period=14):
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, period=14, smoothK=3, smoothD=3):
    df['RSI'] = calculate_rsi(df, period)
    df['StochRSI'] = (df['RSI'] - df['RSI'].rolling(window=period).min()) / (df['RSI'].rolling(window=period).max() - df['RSI'].rolling(window=period).min())
    df['%K'] = df['StochRSI'].rolling(window=smoothK).mean()
    df['%D'] = df['%K'].rolling(window=smoothD).mean()
    return df

def calculate_bollinger_bands(df, period=20, num_std=2):
    df['SMA'] = df['Adj Close'].rolling(window=period).mean()
    df['Upper Band'] = df['SMA'] + (df['Adj Close'].rolling(window=period).std() * num_std)
    df['Lower Band'] = df['SMA'] - (df['Adj Close'].rolling(window=period).std() * num_std)
    return df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    df['MACD'] = df['Adj Close'].ewm(span=fast_period, adjust=False).mean() - df['Adj Close'].ewm(span=slow_period, adjust=False).mean()
    df['MACD Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD Histogram'] = df['MACD'] - df['MACD Signal']
    return df

def calculate_atr(df, period=14):
    df['HL'] = df['High'] - df['Low']
    df['HC'] = abs(df['High'] - df['Adj Close'].shift())
    df['LC'] = abs(df['Low'] - df['Adj Close'].shift())
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period, min_periods=1).mean()
    return df

def generate_momentum_signals(df, period=20):
    df['Momentum'] = df['Adj Close'].diff(period)
    df['Buy Signal'] = (df['Momentum'] > 0).astype(int)
    df['Sell Signal'] = (df['Momentum'] < 0).astype(int)
    return df

class Portfolio:
    def __init__(self, initial_cash, transaction_cost, stop_loss_pct, take_profit_pct, risk_per_trade):
        self.cash = initial_cash
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.risk_per_trade = risk_per_trade
        self.positions = {}
        self.portfolio_value = []
        self.trade_log = []

    def _position_size(self, asset, entry_price, atr):
        risk_amount = self.cash * self.risk_per_trade
        position_size = risk_amount / atr
        return position_size

    def buy(self, asset, price, atr):
        if asset not in self.positions:
            position_size = self._position_size(asset, price, atr)
            if self.cash >= position_size * price * (1 + self.transaction_cost):
                self.positions[asset] = {'size': position_size, 'entry_price': price, 'value': position_size * price}
                self.cash -= position_size * price * (1 + self.transaction_cost)
                self.trade_log.append({'asset': asset, 'price': price, 'action': 'buy', 'size': position_size, 'cash': self.cash})
                logging.info(f'Bought {position_size} shares of {asset} at {price}')
    
    def sell(self, asset, price):
        if asset in self.positions:
            position = self.positions.pop(asset)
            sale_value = position['size'] * price
            self.cash += sale_value * (1 - self.transaction_cost)
            self.trade_log.append({'asset': asset, 'price': price, 'action': 'sell', 'size': position['size'], 'cash': self.cash})
            logging.info(f'Sold {position["size"]} shares of {asset} at {price}')

    def update_positions(self, prices):
        assets_to_remove = []
        for asset in list(self.positions.keys()):
            position = self.positions[asset]
            position['value'] = position['size'] * prices[asset]
            if position['value'] <= position['entry_price'] * (1 - self.stop_loss_pct) or position['value'] >= position['entry_price'] * (1 + self.take_profit_pct):
                self.sell(asset, prices[asset])
                if asset in self.positions:
                    assets_to_remove.append(asset)
        for asset in assets_to_remove:
            del self.positions[asset]
        total_value = self.cash + sum(position['value'] for position in self.positions.values())
        self.portfolio_value.append(total_value)


def generate_signals(data):
    for asset, df in data.items():
        df = calculate_indicators(df)
        df['Signal'] = 0
        df.loc[df['Buy Signal'] == 1, 'Signal'] = 1
        df.loc[df['Sell Signal'] == 1, 'Signal'] = -1
        data[asset] = df
    return data

def create_features(data):
    features = []
    labels = []
    for asset, df in data.items():
        df = df.dropna()
        for i in range(20, len(df)):
            features.append(df.iloc[i-20:i][['RSI', 'StochRSI', 'MACD Histogram', 'ATR', 'Momentum']].values.flatten())
            labels.append(df.iloc[i]['Signal'])
    return np.array(features), np.array(labels)

def train_ml_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Model Accuracy: {accuracy}')
    return model

def main():
    assets = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    initial_cash = 100000
    transaction_cost = 0.001
    stop_loss_pct = 0.05
    take_profit_pct = 0.1
    risk_per_trade = 0.01

    data = download_data(assets, start_date, end_date)
    data = generate_signals(data)
    
    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    for asset, df in data.items():
        data[asset][['RSI', 'StochRSI', 'MACD Histogram', 'ATR', 'Momentum']] = imputer.fit_transform(df[['RSI', 'StochRSI', 'MACD Histogram', 'ATR', 'Momentum']])
    
    X, y = create_features(data)
    model = train_ml_model(X, y)
    
    portfolio = Portfolio(initial_cash, transaction_cost, stop_loss_pct, take_profit_pct, risk_per_trade)
    
    for i in range(20, len(data[assets[0]])):
        prices = {asset: data[asset]['Adj Close'].iloc[i] for asset in assets}
        signals = {asset: model.predict([data[asset].iloc[i-20:i][['RSI', 'StochRSI', 'MACD Histogram', 'ATR', 'Momentum']].values.flatten()])[0] for asset in assets}
        for asset, signal in signals.items():
            if signal == 1:
                portfolio.buy(asset, prices[asset], data[asset]['ATR'].iloc[i])
            elif signal == -1:
                portfolio.sell(asset, prices[asset])
        portfolio.update_positions(prices)
        logging.info(f'Date: {data[assets[0]].index[i]}, Portfolio Value: {portfolio.portfolio_value[-1]}')

if __name__ == '__main__':
    main()
