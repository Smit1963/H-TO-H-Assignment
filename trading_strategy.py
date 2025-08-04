"""
Trading Strategy Module
Implements RSI + Moving Average crossover strategy with backtesting
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Trading strategy implementation with RSI + MA crossover
    """
    
    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on RSI + MA crossover strategy
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        try:
            signals = data.copy()
            
            # Initialize signal columns
            signals.loc[:, 'signal'] = 0  # 0: hold, 1: buy, -1: sell
            signals.loc[:, 'position'] = 0  # Current position
            
            # RSI + MA Crossover Strategy
            for i in range(1, len(signals)):
                current_rsi = signals.loc[signals.index[i], 'rsi']
                current_ma_short = signals.loc[signals.index[i], 'ma_short']
                current_ma_long = signals.loc[signals.index[i], 'ma_long']
                prev_ma_short = signals.loc[signals.index[i-1], 'ma_short']
                prev_ma_long = signals.loc[signals.index[i-1], 'ma_long']
                
                # Buy Signal: RSI < 40 (very relaxed) AND 20-DMA > 50-DMA
                if (current_rsi < 40 and 
                    current_ma_short > current_ma_long):
                    signals.loc[signals.index[i], 'signal'] = 1
                
                # Additional Buy Signal: Strong RSI oversold
                elif current_rsi < 30:
                    signals.loc[signals.index[i], 'signal'] = 1
                
                # Sell Signal: RSI > 60 (very relaxed) OR 20-DMA < 50-DMA
                elif (current_rsi > 60 or
                      current_ma_short < current_ma_long):
                    signals.loc[signals.index[i], 'signal'] = -1
                
                # Additional Sell Signal: Strong RSI overbought
                elif current_rsi > 75:
                    signals.loc[signals.index[i], 'signal'] = -1
                
                # Update position
                if i > 0:
                    signals.loc[signals.index[i], 'position'] = signals.loc[signals.index[i-1], 'position'] + signals.loc[signals.index[i], 'signal']
            
            logger.info(f"Generated signals for {len(signals)} data points")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return data
    
    def backtest_strategy(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Backtest the trading strategy
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            symbol (str): Stock symbol
            
        Returns:
            Dict: Backtest results
        """
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Generate signals
            signals = self.generate_signals(data)
            
            # Initialize backtest variables
            self.current_capital = self.initial_capital
            self.positions = {}
            self.trades = []
            self.portfolio_value = []
            
            position_size = self.initial_capital * config.POSITION_SIZE
            
            for i in range(len(signals)):
                current_price = signals.loc[signals.index[i], 'Close']
                current_date = signals.loc[signals.index[i], 'Date']
                
                # Execute trades based on signals
                if signals.loc[signals.index[i], 'signal'] == 1:  # Buy signal
                    if symbol not in self.positions or self.positions[symbol] == 0:
                        shares = int(position_size / current_price)
                        if shares > 0:
                            self.positions[symbol] = shares
                            trade = {
                                'date': current_date,
                                'symbol': symbol,
                                'action': 'BUY',
                                'price': current_price,
                                'shares': shares,
                                'value': shares * current_price,
                                'capital': self.current_capital
                            }
                            self.trades.append(trade)
                            logger.info(f"BUY: {shares} shares of {symbol} at {current_price:.2f}")
                
                elif signals.loc[signals.index[i], 'signal'] == -1:  # Sell signal
                    if symbol in self.positions and self.positions[symbol] > 0:
                        shares = self.positions[symbol]
                        trade_value = shares * current_price
                        self.current_capital += trade_value
                        
                        trade = {
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': current_price,
                            'shares': shares,
                            'value': trade_value,
                            'capital': self.current_capital
                        }
                        self.trades.append(trade)
                        self.positions[symbol] = 0
                        logger.info(f"SELL: {shares} shares of {symbol} at {current_price:.2f}")
                
                # Calculate portfolio value
                portfolio_value = self.current_capital
                for pos_symbol, pos_shares in self.positions.items():
                    if pos_shares > 0:
                        portfolio_value += pos_shares * current_price
                
                self.portfolio_value.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': self.current_capital,
                    'positions': self.positions.copy()
                })
            
            # Calculate final results
            final_value = self.portfolio_value[-1]['portfolio_value']
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate additional metrics
            metrics = self._calculate_performance_metrics()
            
            results = {
                'symbol': symbol,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'trades': self.trades,
                'portfolio_value': self.portfolio_value,
                'metrics': metrics
            }
            
            logger.info(f"Backtest completed for {symbol}. Total return: {total_return:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Returns:
            Dict: Performance metrics
        """
        try:
            if not self.trades:
                return {}
            
            # Calculate win/loss ratio
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            profitable_trades = 0
            total_trades = 0
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                
                if sell_price > buy_price:
                    profitable_trades += 1
                total_trades += 1
            
            win_ratio = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate average trade return
            trade_returns = []
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                trade_return = ((sell_price - buy_price) / buy_price) * 100
                trade_returns.append(trade_return)
            
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            
            # Calculate max drawdown
            portfolio_values = [pv['portfolio_value'] for pv in self.portfolio_value]
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            metrics = {
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_ratio': win_ratio,
                'avg_trade_return': avg_trade_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_values)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            portfolio_values (List[float]): List of portfolio values
            
        Returns:
            float: Maximum drawdown percentage
        """
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, portfolio_values: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            portfolio_values (List[float]): List of portfolio values
            risk_free_rate (float): Risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        if len(portfolio_values) < 2:
            return 0
        
        returns = []
        for i in range(1, len(portfolio_values)):
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns) * 252  # Annualized
        std_return = np.std(returns) * np.sqrt(252)  # Annualized
        
        if std_return == 0:
            return 0
        
        sharpe_ratio = (avg_return - risk_free_rate) / std_return
        return sharpe_ratio
    
    def get_current_signals(self, data: pd.DataFrame) -> Dict:
        """
        Get current trading signals for live trading
        
        Args:
            data (pd.DataFrame): Latest stock data
            
        Returns:
            Dict: Current trading signals
        """
        try:
            if len(data) < 2:
                return {}
            
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            signals = {
                'rsi': latest['rsi'],
                'ma_short': latest['ma_short'],
                'ma_long': latest['ma_long'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'signal': 0,
                'strength': 0
            }
            
            # Buy signal conditions
            if (latest['rsi'] < config.RSI_OVERSOLD and 
                latest['ma_short'] > latest['ma_long'] and
                previous['ma_short'] <= previous['ma_long']):
                signals['signal'] = 1
                signals['strength'] = config.RSI_OVERSOLD - latest['rsi']
            
            # Sell signal conditions
            elif (latest['rsi'] > config.RSI_OVERBOUGHT or
                  (latest['ma_short'] < latest['ma_long'] and
                   previous['ma_short'] >= previous['ma_long'])):
                signals['signal'] = -1
                signals['strength'] = latest['rsi'] - config.RSI_OVERBOUGHT
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting current signals: {str(e)}")
            return {}

if __name__ == "__main__":
    # Test trading strategy
    from data_ingestion import DataIngestion
    
    # Fetch data
    data_ingestion = DataIngestion()
    stocks_data = data_ingestion.fetch_multiple_stocks(config.NIFTY_50_STOCKS[:2])
    
    # Test strategy
    strategy = TradingStrategy()
    
    for symbol, data in stocks_data.items():
        print(f"\n=== Backtesting {symbol} ===")
        results = strategy.backtest_strategy(data, symbol)
        
        if results:
            print(f"Total Return: {results['total_return']:.2f}%")
            print(f"Total Trades: {results['metrics']['total_trades']}")
            print(f"Win Ratio: {results['metrics']['win_ratio']:.2f}%")
            print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
            print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}") 