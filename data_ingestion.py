"""
Data Ingestion Module
Handles fetching stock data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Class to handle data ingestion from Yahoo Finance API
    """
    
    def __init__(self):
        self.stocks_data = {}
        
    def fetch_stock_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period for data
            interval (str): Data interval
            
        Returns:
            pd.DataFrame: Stock data with OHLCV and technical indicators
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return None
                
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add date column
            data['Date'] = data.index
            data = data.reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        try:
            # RSI
            data['rsi'] = self._calculate_rsi(data['Close'], config.RSI_PERIOD)
            
            # Moving Averages
            data['ma_short'] = data['Close'].rolling(window=config.SHORT_MA_PERIOD).mean()
            data['ma_long'] = data['Close'].rolling(window=config.LONG_MA_PERIOD).mean()
            
            # MACD
            macd_data = self._calculate_macd(data['Close'])
            data['macd'] = macd_data['macd']
            data['macd_signal'] = macd_data['macd_signal']
            data['macd_histogram'] = macd_data['macd_histogram']
            
            # Volume indicators
            data['volume_ma'] = data['Volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_ma']
            
            # Price change
            data['price_change'] = data['Close'].pct_change()
            
            # MA Crossover signal
            data['ma_crossover'] = np.where(data['ma_short'] > data['ma_long'], 1, 0)
            
            # Buy/Sell signals based on RSI
            data['rsi_buy_signal'] = np.where(data['rsi'] < config.RSI_OVERSOLD, 1, 0)
            data['rsi_sell_signal'] = np.where(data['rsi'] > config.RSI_OVERBOUGHT, 1, 0)
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices (pd.Series): Price series
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            Dict: MACD, signal, and histogram
        """
        ema_fast = prices.ewm(span=config.MACD_FAST).mean()
        ema_slow = prices.ewm(span=config.MACD_SLOW).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=config.MACD_SIGNAL).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols (List[str]): List of stock symbols
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with stock data
        """
        stocks_data = {}
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, config.DATA_PERIOD, config.DATA_INTERVAL)
            if data is not None:
                stocks_data[symbol] = data
                
        self.stocks_data = stocks_data
        return stocks_data
    
    def get_latest_data(self, symbol: str) -> Optional[pd.Series]:
        """
        Get the latest data point for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.Series: Latest data point
        """
        if symbol in self.stocks_data:
            return self.stocks_data[symbol].iloc[-1]
        return None
    
    def get_data_for_backtest(self, symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
        """
        Get data for backtesting
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days for backtest
            
        Returns:
            pd.DataFrame: Data for backtesting
        """
        if symbol in self.stocks_data:
            data = self.stocks_data[symbol].copy()
            # Get last 'days' records
            return data.tail(days)
        return None

if __name__ == "__main__":
    # Test data ingestion
    data_ingestion = DataIngestion()
    stocks_data = data_ingestion.fetch_multiple_stocks(config.NIFTY_50_STOCKS[:3])
    
    for symbol, data in stocks_data.items():
        print(f"\n{symbol}: {len(data)} records")
        print(f"Latest Close: {data['Close'].iloc[-1]:.2f}")
        print(f"Latest RSI: {data['rsi'].iloc[-1]:.2f}") 