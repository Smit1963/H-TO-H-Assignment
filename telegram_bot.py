"""
Telegram Bot Module
Handles sending trading alerts and notifications via Telegram
"""

import asyncio
from telegram import Bot
from telegram.error import TelegramError
import logging
from typing import Dict, List, Optional
from datetime import datetime
import config
import requests

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for sending trading alerts and notifications
    """
    
    def __init__(self, token: str = config.TELEGRAM_BOT_TOKEN, 
                 chat_id: str = config.TELEGRAM_CHAT_ID):
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self.is_initialized = False
        self._session = None  # Add session management
        
    def initialize(self) -> bool:
        """
        Initialize the Telegram bot
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.token or not self.chat_id:
                logger.warning("Telegram token or chat ID not configured")
                return False
            
            # Create bot with proper session management
            import httpx
            self._session = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=5)
            )
            self.bot = Bot(token=self.token, request=self._session)
            self.is_initialized = True
            logger.info("Telegram bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {str(e)}")
            return False
    
    async def send_message(self, message: str) -> bool:
        """
        Send a message via Telegram
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized or not self.bot:
                logger.error("Telegram bot not initialized")
                return False
            
            # Send message
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logger.info("Message sent via Telegram")
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    async def send_trade_alert(self, trade_data: Dict) -> bool:
        """
        Send trade alert
        
        Args:
            trade_data (Dict): Trade information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            # Format trade alert message
            action = trade_data['action']
            symbol = trade_data['symbol']
            price = trade_data['price']
            shares = trade_data['shares']
            value = trade_data['value']
            
            message = f"""
🚨 <b>TRADE ALERT</b> 🚨

<b>Action:</b> {action}
<b>Symbol:</b> {symbol}
<b>Price:</b> ₹{price:.2f}
<b>Shares:</b> {shares}
<b>Value:</b> ₹{value:.2f}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 <b>Algo Trading System</b>
            """
            
            return await self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending trade alert: {str(e)}")
            return False
    
    async def send_signal_alert(self, signal_data: Dict) -> bool:
        """
        Send trading signal alert
        
        Args:
            signal_data (Dict): Signal information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            rsi = signal_data.get('rsi', 0)
            confidence = signal_data.get('confidence', 0)
            
            # Choose emoji based on signal
            if signal == 'BUY':
                emoji = "🟢"
                action = "BUY SIGNAL"
            elif signal == 'SELL':
                emoji = "🔴"
                action = "SELL SIGNAL"
            else:
                emoji = "🟡"
                action = "HOLD"
            
            message = f"""
{emoji} <b>{action}</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal}
<b>RSI:</b> {rsi:.2f}
<b>Confidence:</b> {confidence:.2f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>📊 Algo Trading System</b>
            """
            
            return await self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending signal alert: {str(e)}")
            return False
    
    async def send_error_alert(self, error_message: str, context: str = "") -> bool:
        """
        Send error alert
        
        Args:
            error_message (str): Error message
            context (str): Additional context
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            message = f"""
⚠️ <b>ERROR ALERT</b> ⚠️

<b>Error:</b> {error_message}
<b>Context:</b> {context}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔧 <b>Algo Trading System</b>
            """
            
            return await self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending error alert: {str(e)}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Send daily summary
        
        Args:
            summary_data (Dict): Summary information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            total_return = summary_data.get('total_return', 0)
            total_trades = summary_data.get('total_trades', 0)
            win_ratio = summary_data.get('win_ratio', 0)
            portfolio_value = summary_data.get('portfolio_value', 0)
            
            # Choose emoji based on performance
            if total_return > 0:
                emoji = "📈"
                performance = "POSITIVE"
            else:
                emoji = "📉"
                performance = "NEGATIVE"
            
            message = f"""
📊 <b>DAILY SUMMARY</b> 📊

<b>Performance:</b> {emoji} {performance}
<b>Total Return:</b> {total_return:.2f}%
<b>Total Trades:</b> {total_trades}
<b>Win Ratio:</b> {win_ratio:.2f}%
<b>Portfolio Value:</b> ₹{portfolio_value:.2f}
<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}

💰 <b>Algo Trading System</b>
            """
            
            return await self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
            return False
    
    async def send_ml_prediction_alert(self, prediction_data: Dict) -> bool:
        """
        Send ML prediction alert
        
        Args:
            prediction_data (Dict): Prediction information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            symbol = prediction_data['symbol']
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']
            probability_up = prediction_data.get('probability_up', 0)
            probability_down = prediction_data.get('probability_down', 0)
            
            # Choose emoji based on prediction
            if prediction == 1:
                emoji = "🤖🟢"
                direction = "UP"
            else:
                emoji = "🤖🔴"
                direction = "DOWN"
            
            message = f"""
{emoji} <b>ML PREDICTION</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Prediction:</b> {direction}
<b>Confidence:</b> {confidence:.2f}%
<b>Probability Up:</b> {probability_up:.2f}%
<b>Probability Down:</b> {probability_down:.2f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🧠 <b>Machine Learning Model</b>
            """
            
            return await self.send_message(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending ML prediction alert: {str(e)}")
            return False
    
    async def send_system_status(self, status_data: Dict) -> bool:
        """
        Send system status update
        
        Args:
            status_data (Dict): Status information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                return False
            
            status = status_data.get('status', 'UNKNOWN')
            message = status_data.get('message', '')
            uptime = status_data.get('uptime', '')
            
            # Choose emoji based on status
            if status == 'RUNNING':
                emoji = "🟢"
            elif status == 'ERROR':
                emoji = "🔴"
            else:
                emoji = "🟡"
            
            message_text = f"""
{emoji} <b>SYSTEM STATUS</b> {emoji}

<b>Status:</b> {status}
<b>Message:</b> {message}
<b>Uptime:</b> {uptime}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚙️ <b>Algo Trading System</b>
            """
            
            return await self.send_message(message_text.strip())
            
        except Exception as e:
            logger.error(f"Error sending system status: {str(e)}")
            return False

    def send_message_sync(self, message: str) -> bool:
        """
        Synchronous wrapper for sending messages via Telegram
        
        Args:
            message (str): Message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized or not self.token:
                logger.error("Telegram bot not initialized")
                return False
            
            # Use simple requests approach
            import requests
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Message sent via Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False
    
    def send_trade_alert_sync(self, trade_data: Dict) -> bool:
        """
        Synchronous wrapper for sending trade alerts
        
        Args:
            trade_data (Dict): Trade information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return asyncio.run(self.send_trade_alert(trade_data))
        except Exception as e:
            logger.error(f"Error in sync trade alert: {str(e)}")
            return False
    
    def send_signal_alert_sync(self, signal_data: Dict) -> bool:
        """
        Synchronous wrapper for sending signal alerts
        
        Args:
            signal_data (Dict): Signal information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized or not self.token:
                logger.error("Telegram bot not initialized")
                return False
            
            symbol = signal_data['symbol']
            signal = signal_data['signal']
            rsi = signal_data.get('rsi', 0)
            confidence = signal_data.get('confidence', 0)
            
            # Choose emoji based on signal
            if signal == 'BUY':
                emoji = "🟢"
                action = "BUY SIGNAL"
            elif signal == 'SELL':
                emoji = "🔴"
                action = "SELL SIGNAL"
            else:
                emoji = "🟡"
                action = "HOLD"
            
            message = f"""
{emoji} <b>{action}</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal}
<b>RSI:</b> {rsi:.2f}
<b>Confidence:</b> {confidence:.2f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>📊 Algo Trading System</b>
            """
            
            # Use simple requests approach
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message.strip(),
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Signal alert sent via Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error in sync signal alert: {str(e)}")
            return False
    
    def send_error_alert_sync(self, error_message: str, context: str = "") -> bool:
        """
        Synchronous wrapper for sending error alerts
        
        Args:
            error_message (str): Error message
            context (str): Additional context
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            return asyncio.run(self.send_error_alert(error_message, context))
        except Exception as e:
            logger.error(f"Error in sync error alert: {str(e)}")
            return False
    
    def send_daily_summary_sync(self, summary_data: Dict) -> bool:
        """
        Synchronous wrapper for sending daily summaries
        
        Args:
            summary_data (Dict): Summary information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized or not self.token:
                logger.error("Telegram bot not initialized")
                return False
            
            date = summary_data.get('date', '')
            total_signals = summary_data.get('total_signals', 0)
            buy_signals = summary_data.get('buy_signals', 0)
            sell_signals = summary_data.get('sell_signals', 0)
            hold_signals = summary_data.get('hold_signals', 0)
            total_return = summary_data.get('total_return', 0)
            win_ratio = summary_data.get('win_ratio', 0)
            
            message = f"""
📊 <b>DAILY SUMMARY</b> 📊

<b>Date:</b> {date}
<b>Total Signals:</b> {total_signals}
<b>Buy Signals:</b> 🟢 {buy_signals}
<b>Sell Signals:</b> 🔴 {sell_signals}
<b>Hold Signals:</b> 🟡 {hold_signals}
<b>Total Return:</b> {total_return:.2f}%
<b>Win Ratio:</b> {win_ratio:.1f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>💰 Algo Trading System</b>
            """
            
            # Use simple requests approach
            import requests
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message.strip(),
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("Daily summary sent via Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error in sync daily summary: {str(e)}")
            return False
    
    def send_ml_prediction_alert_sync(self, prediction_data: Dict) -> bool:
        """
        Synchronous wrapper for sending ML prediction alerts
        
        Args:
            prediction_data (Dict): Prediction information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized or not self.token:
                logger.error("Telegram bot not initialized")
                return False
            
            symbol = prediction_data['symbol']
            prediction = prediction_data['prediction']
            confidence = prediction_data.get('confidence', 0)
            probability_up = prediction_data.get('probability_up', 0)
            probability_down = prediction_data.get('probability_down', 0)
            
            # Choose emoji and direction based on prediction
            if prediction == 1:
                emoji = "📈"
                direction = "UP"
            elif prediction == 0:
                emoji = "📉"
                direction = "DOWN"
            else:
                emoji = "❓"
                direction = "UNKNOWN"
            
            message = f"""
🧠 <b>ML PREDICTION</b> 🧠

<b>Symbol:</b> {symbol}
<b>Prediction:</b> {emoji} {direction}
<b>Confidence:</b> {confidence:.1f}%
<b>Probability UP:</b> {probability_up:.1f}%
<b>Probability DOWN:</b> {probability_down:.1f}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>🤖 Algo Trading System</b>
            """
            
            # Use simple requests approach
            import requests
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message.strip(),
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("ML prediction alert sent via Telegram")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error in sync ML prediction alert: {str(e)}")
            return False
    


if __name__ == "__main__":
    # Test Telegram bot
    async def test_telegram_bot():
        telegram_bot = TelegramBot()
        
        if telegram_bot.initialize():
            print("Telegram bot initialized successfully")
            
            # Test sending a message
            test_message = "🤖 Algo Trading System is now online!"
            if await telegram_bot.send_message(test_message):
                print("Test message sent successfully")
            else:
                print("Failed to send test message")
        else:
            print("Failed to initialize Telegram bot")
    
    # Run the async test
    asyncio.run(test_telegram_bot()) 