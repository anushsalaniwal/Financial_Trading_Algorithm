import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
initial_cash = 100000
transaction_cost = 0.001  # 0.1% transaction cost
stop_loss_pct = 0.95  # 5% stop loss
take_profit_pct = 1.10  # 10% take profit
risk_per_trade = 0.01  # 1% of portfolio
lookback_period = 252  # 1 year for annualized calculations

# Download historical price data for multiple assets
assets = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(assets, start='2020-01-01', end='2023-01-01')['Adj Close']

# Ensure 'Adj Close' column is present for each asset
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(1)  # Flatten multi-level columns to single level
    data.columns.name = None  # Remove the name of the column index

# Calculate technical indicators
def calculate_indicators(df):
    if 'Adj Close' not in df.columns:
        raise KeyError("'Adj Close' column is missing in the DataFrame")
    
    # RSI calculation
    def calculate_rsi(series, window):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['Adj Close'], 14)
    
    # StochRSI calculation
    def calculate_stochrsi(series, window):
        rsi = calculate_rsi(series, window)
        stochrsi = ((rsi - rsi.rolling(window=window, min_periods=1).min()) /
                    (rsi.rolling(window=window, min_periods=1).max() - rsi.rolling(window=window, min_periods=1).min())).fillna(0)
        return stochrsi
    
    df['StochRSI'] = calculate_stochrsi(df['Adj Close'], 14)
    
    # Bollinger Bands calculation
    def calculate_bollinger_bands(series, window, num_std_dev):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * num_std_dev)
        lower_band = sma - (std * num_std_dev)
        return sma, upper_band, lower_band
    
    sma, upper_band, lower_band = calculate_bollinger_bands(df['Adj Close'], 20, 2)
    df['SMA'] = sma
    df['Upper Band'] = upper_band
    df['Lower Band'] = lower_band
    
    return df

# Generate trading signals
def generate_signals(df):
    df['Buy Signal'] = ((df['RSI'] < 30) | (df['StochRSI'] < 0.2) | (df['Adj Close'] < df['Lower Band'])).astype(int)
    df['Sell Signal'] = ((df['RSI'] > 70) | (df['StochRSI'] > 0.8) | (df['Adj Close'] > df['Upper Band'])).astype(int)
    return df

# Initialize a dictionary to store processed data
processed_data = {}

# Calculate indicators for all assets
for asset in assets:
    print(f"Calculating indicators for asset: {asset}")
    asset_data = data[[asset]].copy()
    asset_data.columns = ['Adj Close']  # Rename column for compatibility with functions
    calculated_data = calculate_indicators(asset_data)
    processed_data[asset] = calculated_data  # Store the processed data separately

# Combine processed data into a single DataFrame
combined_data = pd.concat(processed_data.values(), axis=1, keys=processed_data.keys())

# Generate signals for all assets
signals = {asset: generate_signals(processed_data[asset]) for asset in assets}

# Backtesting with dynamic position sizing and diversification
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

    def _position_size(self, asset, entry_price):
        risk_amount = self.cash * self.risk_per_trade
        stop_loss_amount = entry_price * (1 - self.stop_loss_pct)
        position_size = risk_amount / (entry_price - stop_loss_amount)
        return position_size

    def buy(self, asset, price):
        if asset not in self.positions:
            position_size = self._position_size(asset, price)
            if self.cash >= position_size * price * (1 + self.transaction_cost):
                self.positions[asset] = {'size': position_size, 'entry_price': price, 'value': position_size * price}
                self.cash -= position_size * price * (1 + self.transaction_cost)
                self.trade_log.append({'asset': asset, 'price': price, 'action': 'buy', 'size': position_size, 'cash': self.cash})

    def sell(self, asset, price):
        if asset in self.positions:
            position = self.positions.pop(asset)
            sale_value = position['size'] * price * (1 - self.transaction_cost)
            self.cash += sale_value
            self.trade_log.append({'asset': asset, 'price': price, 'action': 'sell', 'size': position['size'], 'cash': self.cash})

    def update_positions(self, prices):
        for asset, position in self.positions.items():
            current_price = prices[asset]
            if current_price <= position['entry_price'] * self.stop_loss_pct or current_price >= position['entry_price'] * self.take_profit_pct:
                self.sell(asset, current_price)

    def update_portfolio_value(self, prices):
        value = self.cash
        for asset, position in self.positions.items():
            value += position['size'] * prices[asset]
        self.portfolio_value.append(value)
        return value

# Initialize portfolio
portfolio = Portfolio(initial_cash, transaction_cost, stop_loss_pct, take_profit_pct, risk_per_trade)

# Backtesting loop
for i in range(1, len(data)):
    prices = {asset: data[asset]['Adj Close'].iloc[i] for asset in assets}
    signals_today = {asset: signals[asset].iloc[i] for asset in assets}
    
    # Update positions with current prices
    portfolio.update_positions(prices)
    
    # Buy signals
    for asset in assets:
        if signals_today[asset]['Buy Signal'] == 1:
            portfolio.buy(asset, prices[asset])
    
    # Sell signals
    for asset in assets:
        if signals_today[asset]['Sell Signal'] == 1:
            portfolio.sell(asset, prices[asset])
    
    # Update portfolio value
    portfolio_value = portfolio.update_portfolio_value(prices)
    logging.info(f'Date: {data.index[i]}, Portfolio Value: {portfolio_value}')

# Performance metrics
portfolio_value_series = pd.Series(portfolio.portfolio_value)
returns = portfolio_value_series.pct_change().dropna()
total_return = portfolio_value_series.iloc[-1] / initial_cash - 1
annualized_return = (1 + total_return) ** (252 / len(data)) - 1
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
max_drawdown = (portfolio_value_series / portfolio_value_series.cummax() - 1).min()

performance_metrics = {
    "Initial Cash": initial_cash,
    "Final Portfolio Value": portfolio_value_series.iloc[-1],
    "Total Return": total_return,
    "Annualized Return": annualized_return,
    "Sharpe Ratio": sharpe_ratio,
    "Max Drawdown": max_drawdown
}

print(performance_metrics)
