"""
Configuration file for Algo-Trading System
Contains all settings, constants, and API configurations
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Stock Configuration
NIFTY_50_STOCKS = [
    'RELIANCE.NS',  # Reliance Industries
    'TCS.NS',       # Tata Consultancy Services
    'HDFCBANK.NS',  # HDFC Bank
    'INFY.NS',      # Infosys
    'ICICIBANK.NS'  # ICICI Bank
]

# Trading Strategy Parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SHORT_MA_PERIOD = 20
LONG_MA_PERIOD = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Backtesting Parameters
BACKTEST_DAYS = 180  # 6 months
INITIAL_CAPITAL = 100000  # 1 Lakh INR
POSITION_SIZE = 0.2  # 20% of capital per trade

# Google Sheets Configuration
GOOGLE_SHEETS_CREDENTIALS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Algo Trading System'
TRADE_LOG_SHEET = 'Trade Log'
SUMMARY_SHEET = 'Summary'
ANALYTICS_SHEET = 'Analytics'

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ML Model Configuration
ML_FEATURES = ['rsi', 'macd', 'macd_signal', 'volume_ratio', 'price_change', 'ma_crossover']
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'algo_trading.log'

# Data Configuration
DATA_INTERVAL = '1d'  # Daily data
DATA_PERIOD = '1y'   # 1 year of historical data 