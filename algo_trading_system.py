"""
Main Algo Trading System
Orchestrates all components for automated trading
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

from data_ingestion import DataIngestion
from trading_strategy import TradingStrategy
from ml_model import MLModel
from google_sheets import GoogleSheetsManager
from telegram_bot import TelegramBot
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlgoTradingSystem:
    """
    Main algo trading system that orchestrates all components
    """
    
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.trading_strategy = TradingStrategy()
        self.ml_model = MLModel()
        self.sheets_manager = GoogleSheetsManager()
        self.telegram_bot = TelegramBot()
        
        self.stocks_data = {}
        self.trained_models = {}
        self.is_running = False
        self.start_time = None
        
    def initialize_system(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Initializing Algo Trading System...")
            
            # Initialize Telegram bot (optional)
            telegram_initialized = False
            if self.telegram_bot.initialize():
                telegram_initialized = True
                logger.info("Telegram bot initialized successfully")
            else:
                logger.info("Telegram bot not configured - continuing without notifications")
            
            # Initialize Google Sheets (optional)
            sheets_initialized = False
            if self.sheets_manager.authenticate():
                if self.sheets_manager.create_or_open_spreadsheet():
                    sheets_initialized = True
                    logger.info("Google Sheets initialized successfully")
                else:
                    logger.warning("Failed to create/open Google Sheets - continuing without logging")
            else:
                logger.info("Google Sheets not configured - continuing without logging")
            
            # Fetch initial data
            logger.info("Fetching initial stock data...")
            self.stocks_data = self.data_ingestion.fetch_multiple_stocks(config.NIFTY_50_STOCKS)
            
            if not self.stocks_data:
                logger.error("Failed to fetch stock data")
                return False
            
            # Train ML models for each stock
            logger.info("Training ML models...")
            for symbol, data in self.stocks_data.items():
                try:
                    ml_model = MLModel()
                    training_results = ml_model.train_model(data)
                    if training_results:
                        self.trained_models[symbol] = ml_model
                        logger.info(f"Trained ML model for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to train ML model for {symbol}: {str(e)}")
            
            # System initialization complete
            if telegram_initialized:
                logger.info("System initialization complete with Telegram notifications")
            
            logger.info("Algo Trading System initialized successfully")
            logger.info(f"Features enabled: Telegram={telegram_initialized}, Google Sheets={sheets_initialized}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "System Initialization")
            return False
    
    def run_backtest(self) -> Dict:
        """
        Run backtest for all stocks
        
        Returns:
            Dict: Backtest results for all stocks
        """
        try:
            logger.info("Running backtest for all stocks...")
            backtest_results = {}
            
            for symbol, data in self.stocks_data.items():
                logger.info(f"Running backtest for {symbol}")
                
                # Get data for backtest
                backtest_data = self.data_ingestion.get_data_for_backtest(symbol, config.BACKTEST_DAYS)
                if backtest_data is None:
                    continue
                
                # Run backtest
                results = self.trading_strategy.backtest_strategy(backtest_data, symbol)
                if results:
                    backtest_results[symbol] = results
                    
                    # Log to Google Sheets
                    self.sheets_manager.log_summary({
                        'symbol': symbol,
                        'total_return': results['total_return'],
                        'total_trades': results['metrics']['total_trades'],
                        'win_ratio': results['metrics']['win_ratio'],
                        'max_drawdown': results['metrics']['max_drawdown'],
                        'sharpe_ratio': results['metrics']['sharpe_ratio']
                    })
                    
                    # Log trades
                    for trade in results['trades']:
                        self.sheets_manager.log_trade(trade)
            
            logger.info(f"Backtest completed for {len(backtest_results)} stocks")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "Backtest")
            return {}
    
    def scan_for_signals(self) -> List[Dict]:
        """
        Scan all stocks for trading signals
        
        Returns:
            List[Dict]: List of signals found
        """
        try:
            logger.info("Scanning for trading signals...")
            signals = []
            
            for symbol, data in self.stocks_data.items():
                try:
                    # Get current signals from strategy
                    strategy_signals = self.trading_strategy.get_current_signals(data)
                    
                    # Get ML prediction
                    ml_prediction = {}
                    if symbol in self.trained_models:
                        ml_prediction = self.trained_models[symbol].predict(data)
                    
                    # Combine signals
                    combined_signal = {
                        'symbol': symbol,
                        'timestamp': datetime.now(),
                        'strategy_signal': strategy_signals.get('signal', 0),
                        'rsi': strategy_signals.get('rsi', 0),
                        'macd': strategy_signals.get('macd', 0),
                        'ma_crossover': strategy_signals.get('ma_crossover', 0),
                        'ml_prediction': ml_prediction.get('prediction', 'N/A'),
                        'ml_confidence': ml_prediction.get('confidence', 0),
                        'signal': 'HOLD'
                    }
                    
                    # Determine final signal
                    if (strategy_signals.get('signal', 0) == 1 and 
                        ml_prediction.get('prediction', 0) == 1):
                        combined_signal['signal'] = 'BUY'
                    elif (strategy_signals.get('signal', 0) == -1 or 
                          ml_prediction.get('prediction', 0) == 0):
                        combined_signal['signal'] = 'SELL'
                    
                    if combined_signal['signal'] != 'HOLD':
                        signals.append(combined_signal)
                        
                        # Log analytics
                        self.sheets_manager.log_analytics(combined_signal)
                        
                        # Send Telegram alert
                        if self.telegram_bot.is_initialized:
                            self.telegram_bot.send_signal_alert_sync(combined_signal)
                            
                            # Send ML prediction alert if available
                            if ml_prediction:
                                self.telegram_bot.send_ml_prediction_alert_sync({
                                    'symbol': symbol,
                                    'prediction': ml_prediction.get('prediction', 0),
                                    'confidence': ml_prediction.get('confidence', 0),
                                    'probability_up': ml_prediction.get('probability_up', 0),
                                    'probability_down': ml_prediction.get('probability_down', 0)
                                })
                
                except Exception as e:
                    logger.error(f"Error scanning signals for {symbol}: {str(e)}")
            
            logger.info(f"Found {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error scanning for signals: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "Signal Scanning")
            return []
    
    def update_data(self) -> bool:
        """
        Update stock data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Updating stock data...")
            
            # Fetch fresh data
            new_data = self.data_ingestion.fetch_multiple_stocks(config.NIFTY_50_STOCKS)
            
            if new_data:
                self.stocks_data = new_data
                logger.info("Stock data updated successfully")
                return True
            else:
                logger.error("Failed to update stock data")
                return False
                
        except Exception as e:
            logger.error(f"Error updating data: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "Data Update")
            return False
    
    def generate_daily_summary(self) -> Dict:
        """
        Generate daily summary report
        
        Returns:
            Dict: Daily summary
        """
        try:
            logger.info("Generating daily summary...")
            
            # Get trade history
            all_trades = self.sheets_manager.get_trade_history()
            
            # Calculate summary metrics
            total_trades = len(all_trades)
            buy_trades = [t for t in all_trades if t['Action'] == 'BUY']
            sell_trades = [t for t in all_trades if t['Action'] == 'SELL']
            
            profitable_trades = 0
            total_pnl = 0
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = float(buy_trades[i]['Price'])
                sell_price = float(sell_trades[i]['Price'])
                shares = int(buy_trades[i]['Shares'])
                
                pnl = (sell_price - buy_price) * shares
                total_pnl += pnl
                
                if pnl > 0:
                    profitable_trades += 1
            
            win_ratio = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0
            
            # Calculate portfolio value
            portfolio_value = config.INITIAL_CAPITAL + total_pnl
            total_return = (total_pnl / config.INITIAL_CAPITAL) * 100
            
            summary = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_ratio': win_ratio,
                'total_pnl': total_pnl,
                'portfolio_value': portfolio_value,
                'total_return': total_return
            }
            
            # Update portfolio summary in Google Sheets
            self.sheets_manager.update_portfolio_summary({
                'total_value': portfolio_value,
                'cash': config.INITIAL_CAPITAL,
                'invested_amount': portfolio_value - config.INITIAL_CAPITAL,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'num_positions': len(set(t['Symbol'] for t in all_trades))
            })
            
            # Send daily summary via Telegram
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_daily_summary_sync(summary)
            
            logger.info("Daily summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "Daily Summary")
            return {}
    
    def start_automated_trading(self):
        """
        Start the automated trading system
        """
        try:
            logger.info("Starting automated trading system...")
            
            if not self.initialize_system():
                logger.error("Failed to initialize system")
                return
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Schedule tasks
            schedule.every(5).minutes.do(self.scan_for_signals)
            schedule.every(1).hour.do(self.update_data)
            schedule.every().day.at("18:00").do(self.generate_daily_summary)
            
            # Send startup message
            if self.telegram_bot.is_initialized:
                logger.info("Automated trading system started with Telegram notifications")
            
            logger.info("Automated trading system started successfully")
            
            # Main loop
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                    
                    # Update uptime (no system status notifications)
                    if self.start_time:
                        uptime = datetime.now() - self.start_time
                        if uptime.seconds % 3600 == 0:  # Every hour
                            logger.info(f"System running normally - Uptime: {str(uptime).split('.')[0]}")
                
                except KeyboardInterrupt:
                    logger.info("Received interrupt signal, shutting down...")
                    self.stop_automated_trading()
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    if self.telegram_bot.is_initialized:
                        self.telegram_bot.send_error_alert_sync(str(e), "Main Loop")
                    time.sleep(300)  # Wait 5 minutes before retrying
            
        except Exception as e:
            logger.error(f"Error starting automated trading: {str(e)}")
            if self.telegram_bot.is_initialized:
                self.telegram_bot.send_error_alert_sync(str(e), "System Startup")
    
    def stop_automated_trading(self):
        """
        Stop the automated trading system
        """
        try:
            logger.info("Stopping automated trading system...")
            
            self.is_running = False
            
            # Log shutdown
            if self.telegram_bot.is_initialized:
                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                logger.info(f"System shutdown requested - Uptime: {str(uptime).split('.')[0]}")
            
            logger.info("Automated trading system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping automated trading: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """
        Get current system status
        
        Returns:
            Dict: System status information
        """
        try:
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            
            status = {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': str(uptime).split('.')[0],
                'stocks_monitored': len(self.stocks_data),
                'ml_models_trained': len(self.trained_models),
                'telegram_connected': self.telegram_bot.is_initialized,
                'sheets_connected': self.sheets_manager.client is not None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {}

def main():
    """
    Main function to run the algo trading system
    """
    print("🤖 Algo Trading System")
    print("=" * 50)
    
    # Create system instance
    algo_system = AlgoTradingSystem()
    
    # Run backtest first
    print("\n📊 Running Backtest...")
    backtest_results = algo_system.run_backtest()
    
    if backtest_results:
        print(f"\n✅ Backtest completed for {len(backtest_results)} stocks")
        for symbol, results in backtest_results.items():
            print(f"\n{symbol}:")
            print(f"  Total Return: {results['total_return']:.2f}%")
            print(f"  Total Trades: {results['metrics'].get('total_trades', 0)}")
            print(f"  Win Ratio: {results['metrics'].get('win_ratio', 0):.2f}%")
            print(f"  Max Drawdown: {results['metrics'].get('max_drawdown', 0):.2f}%")
            print(f"  Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
    else:
        print("\n⚠️ No backtest results - this is normal for first run")
    
    # Ask user if they want to start automated trading
    print("\n" + "=" * 50)
    response = input("Do you want to start automated trading? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🚀 Starting Automated Trading System...")
        print("Press Ctrl+C to stop the system")
        algo_system.start_automated_trading()
    else:
        print("\n✅ System ready. You can run automated trading later.")
        print("Use: python algo_trading_system.py")

if __name__ == "__main__":
    main() 