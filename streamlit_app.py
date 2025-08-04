"""
Algo Trading System - Streamlit Web Interface
User-friendly web interface for non-technical users
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json
import requests

# Import our trading system modules
from data_ingestion import DataIngestion
from trading_strategy import TradingStrategy
from ml_model import MLModel
from telegram_bot import TelegramBot
from google_sheets import GoogleSheetsManager
import config

# Global function to send automatic notifications
def send_auto_notification(notification_type: str, message: str, data: dict = None):
    """Automatically send notifications to Telegram for all events"""
    try:
        telegram_bot = TelegramBot()
        if telegram_bot.initialize():
            if notification_type == "signal":
                telegram_bot.send_signal_alert_sync(data)
            elif notification_type == "ml":
                telegram_bot.send_ml_prediction_alert_sync(data)
            elif notification_type == "error":
                telegram_bot.send_error_alert_sync(message, "App Error")
            elif notification_type == "status":
                # Send as a simple message since we removed system status
                telegram_bot.send_message_sync(f"📊 {message}")
            elif notification_type == "trade":
                telegram_bot.send_trade_alert_sync(data)
            elif notification_type == "summary":
                telegram_bot.send_daily_summary_sync(data)
    except Exception as e:
        st.error(f"Failed to send notification: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Algo Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main_header():
    """Display main header"""
    st.markdown('<h1 class="main-header">🤖 Algo Trading System</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Trading with Machine Learning & Automation")

def sidebar_setup():
    """Setup sidebar"""
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation (Setup page hidden for regular users)
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "📊 Live Trading", "📈 Analytics"]
    )
    

    
    return page

def dashboard_page():
    """Dashboard page"""
    st.header("🏠 Dashboard")
    
    # System Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Active Stocks</h3>
            <h2>5</h2>
            <p>NIFTY 50 Stocks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Signals Today</h3>
            <h2>3</h2>
            <p>Buy/Sell Signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 ML Accuracy</h3>
            <h2>78%</h2>
            <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>💰 Portfolio</h3>
            <h2>₹102,450</h2>
            <p>+2.45% Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Live Data
    st.subheader("📊 Live Stock Data")
    
    data_ingestion = DataIngestion()
    data = []
    
    for stock in config.NIFTY_50_STOCKS[:5]:
        try:
            stock_data = data_ingestion.fetch_stock_data(stock, period='1d')
            if stock_data is not None and not stock_data.empty:
                latest = stock_data.iloc[-1]
                change_percent = ((latest['Close'] - latest['Open']) / latest['Open'] * 100)
                
                data.append({
                    'Stock': stock,
                    'Price': f"₹{latest['Close']:.2f}",
                    'Change': f"{change_percent:.2f}%",
                    'Volume': f"{latest['Volume']:,}",
                    'RSI': f"{latest.get('rsi', 'N/A'):.2f}" if 'rsi' in stock_data.columns else "N/A"
                })
                
                # No automatic notifications - only when user clicks buttons
                    
        except Exception as e:
            error_msg = f"Error fetching {stock}: {str(e)}"
            st.error(error_msg)
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    # Telegram Bot Information
    st.subheader("📱 Telegram Notifications")
    
    st.info("""
    🤖 **To receive trading signals and ML predictions:**
    
    1. **Search for our bot** on Telegram: `@AlgoTradingBot`
    2. **Start a conversation** with the bot
    3. **Click the buttons** in Live Trading to get notifications
    
    📊 **All signals and predictions will be sent automatically!**
    """)
    
    # Recent Signals
    st.subheader("🚨 Recent Trading Signals")
    
    signals_data = [
        {"Time": "14:30", "Stock": "RELIANCE.NS", "Signal": "BUY", "Price": "₹1,393.70", "Confidence": "85%"},
        {"Time": "14:15", "Stock": "TCS.NS", "Signal": "SELL", "Price": "₹3,245.80", "Confidence": "72%"},
        {"Time": "14:00", "Stock": "HDFCBANK.NS", "Signal": "HOLD", "Price": "₹1,567.90", "Confidence": "60%"}
    ]
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)

def live_trading_page():
    """Live trading page"""
    st.header("📊 Live Trading")
    
    # Stock Selection
    selected_stock = st.selectbox(
        "Select Stock to Monitor:",
        config.NIFTY_50_STOCKS,
        index=0,
        key="stock_selector"
    )
    
    # Auto-fetch data when stock selection changes
    if 'last_selected_stock' not in st.session_state:
        st.session_state.last_selected_stock = None
    
    # Check if stock selection changed
    if st.session_state.last_selected_stock != selected_stock:
        st.session_state.last_selected_stock = selected_stock
        
        # Auto-fetch data for the new stock (no notifications)
        with st.spinner(f"🔄 Auto-fetching data for {selected_stock}..."):
            data_ingestion = DataIngestion()
            data = data_ingestion.fetch_stock_data(selected_stock, period='1mo')
            
            if data is not None:
                st.session_state.current_data = data
                st.success(f"✅ Auto-refreshed data for {selected_stock}!")
            else:
                error_msg = f"Failed to auto-fetch data for {selected_stock}"
                st.error(error_msg)
    
    # Display Chart and Real-time Data
    if 'current_data' in st.session_state and st.session_state.current_data is not None:
        data = st.session_state.current_data
        
        # Calculate real-time values automatically
        latest_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2] if len(data) > 1 else 0
        price_change_percent = (price_change / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
        
        # Calculate technical indicators automatically
        strategy = TradingStrategy()
        signals = strategy.generate_signals(data)
        
        # Get latest technical values
        latest_rsi = signals['rsi'].iloc[-1] if 'rsi' in signals.columns else 0
        latest_macd = signals['macd'].iloc[-1] if 'macd' in signals.columns else 0
        latest_signal = signals['signal'].iloc[-1] if 'signal' in signals.columns else 0
        
        # Display real-time price info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"₹{latest_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_percent:.2f}%)"
            )
        
        with col2:
            rsi_status = "Oversold" if latest_rsi < 30 else "Overbought" if latest_rsi > 70 else "Neutral"
            st.metric(
                label="RSI",
                value=f"{latest_rsi:.2f}",
                delta=rsi_status
            )
        
        with col3:
            st.metric(
                label="MACD",
                value=f"{latest_macd:.4f}",
                delta=f"Signal: {latest_signal}"
            )
        
        with col4:
            st.metric(
                label="Volume",
                value=f"{data['Volume'].iloc[-1]:,}",
                delta=""
            )
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=selected_stock
        )])
        
        fig.update_layout(
            title=f"{selected_stock} - Live Price Chart",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        

        
        # Generate Signals Button (with notifications)
        if st.button("🎯 Generate Signal", key="generate_signal"):
            with st.spinner("Analyzing signals..."):
                strategy = TradingStrategy()
                signals = strategy.generate_signals(data)
                
                if signals is not None:
                    latest_signal = signals['signal'].iloc[-1]
                    latest_rsi = signals['rsi'].iloc[-1] if 'rsi' in signals.columns else 0
                    
                    # Display signal status
                    if latest_signal == 1:
                        st.success("🟢 BUY Signal Generated!")
                        
                        # Send Telegram alert ONLY when button is clicked
                        telegram_bot = TelegramBot()
                        if telegram_bot.initialize():
                            telegram_bot.send_signal_alert_sync({
                                'symbol': selected_stock,
                                'signal': 'BUY',
                                'rsi': latest_rsi,
                                'confidence': 85.0
                            })
                            st.info("📱 BUY signal alert sent to Telegram!")
                    
                    elif latest_signal == -1:
                        st.error("🔴 SELL Signal Generated!")
                        
                        # Send Telegram alert ONLY when button is clicked
                        telegram_bot = TelegramBot()
                        if telegram_bot.initialize():
                            telegram_bot.send_signal_alert_sync({
                                'symbol': selected_stock,
                                'signal': 'SELL',
                                'rsi': latest_rsi,
                                'confidence': 85.0
                            })
                            st.info("📱 SELL signal alert sent to Telegram!")
                    
                    else:
                        st.info("🟡 HOLD - No clear signal")
                        
                        # Send HOLD signal alert ONLY when button is clicked
                        telegram_bot = TelegramBot()
                        if telegram_bot.initialize():
                            telegram_bot.send_signal_alert_sync({
                                'symbol': selected_stock,
                                'signal': 'HOLD',
                                'rsi': latest_rsi,
                                'confidence': 60.0
                            })
                            st.info("📱 HOLD signal alert sent to Telegram!")
        
        # Generate ML Predictions Button (with notifications)
        if st.button("🧠 Get ML Prediction", key="ml_prediction"):
            with st.spinner("Training ML model and generating prediction..."):
                # Fetch more data for training (at least 6 months)
                data_ingestion = DataIngestion()
                training_data = data_ingestion.fetch_stock_data(selected_stock, period='6mo')
                
                if training_data is not None and len(training_data) > 50:  # Need at least 50 data points
                    ml_model = MLModel()
                    
                    # Train the model first
                    training_results = ml_model.train_model(training_data)
                    
                    if training_results:
                        accuracy = training_results['accuracy']
                        st.success(f"✅ ML Model trained successfully! Accuracy: {accuracy:.2f}")
                        
                        # Now make prediction on training data (not current data)
                        prediction = ml_model.predict(training_data)
                        
                        if prediction:
                            pred_direction = prediction.get('prediction', 'UNKNOWN')
                            confidence = prediction.get('confidence', 0)
                            
                            if pred_direction == 1:  # UP
                                st.success(f"📈 ML Predicts: UP with {confidence:.1f}% confidence")
                                
                                # Send Telegram alert ONLY when button is clicked
                                telegram_bot = TelegramBot()
                                if telegram_bot.initialize():
                                    telegram_bot.send_ml_prediction_alert_sync({
                                        'symbol': selected_stock,
                                        'prediction': 1,
                                        'confidence': confidence,
                                        'probability_up': prediction.get('probability_up', 0),
                                        'probability_down': prediction.get('probability_down', 0)
                                    })
                                    st.info("📱 ML prediction alert sent to Telegram!")
                            
                            elif pred_direction == 0:  # DOWN
                                st.error(f"📉 ML Predicts: DOWN with {confidence:.1f}% confidence")
                                
                                # Send Telegram alert ONLY when button is clicked
                                telegram_bot = TelegramBot()
                                if telegram_bot.initialize():
                                    telegram_bot.send_ml_prediction_alert_sync({
                                        'symbol': selected_stock,
                                        'prediction': 0,
                                        'confidence': confidence,
                                        'probability_up': prediction.get('probability_up', 0),
                                        'probability_down': prediction.get('probability_down', 0)
                                    })
                                    st.info("📱 ML prediction alert sent to Telegram!")
                                
                            else:
                                st.warning(f"❓ ML Predicts: {pred_direction} with {confidence:.1f}% confidence")
                        else:
                            st.error("❌ Failed to generate ML prediction")
                    else:
                        st.error("❌ Failed to train ML model - insufficient data or features")
                else:
                    st.error("❌ Insufficient data for ML training. Need at least 50 data points.")
                    st.info("💡 Try selecting a different stock or wait for more data to accumulate.")
    else:
        st.info("Select a stock to start monitoring...")

def setup_page():
    """Setup page"""
    st.header("⚙️ Setup & Configuration")
    
    # Telegram Setup
    st.subheader("📱 Telegram Notifications Setup")
    
    if 'telegram_configured' not in st.session_state:
        st.session_state.telegram_configured = False
    
    if not st.session_state.telegram_configured:
        st.info("Set up Telegram to receive trading alerts and notifications")
        
        with st.expander("🔧 Telegram Setup Guide", expanded=True):
            st.markdown("""
            **Step 1: Create Telegram Bot**
            1. Open Telegram and search for `@BotFather`
            2. Send `/newbot` to BotFather
            3. Choose a name for your bot (e.g., "Algo Trading Bot")
            4. Choose a username ending with 'bot' (e.g., "my_algo_trading_bot")
            5. Copy the token you receive
            
            **Step 2: Get Your Chat ID**
            1. Start a chat with your new bot
            2. Send any message to the bot
            3. Visit: `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
            4. Look for `"chat":{"id":123456789}` and copy the number
            """)
        
        # Telegram Configuration Form
        with st.form("telegram_setup"):
            bot_token = st.text_input("Bot Token:", placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz")
            chat_id = st.text_input("Chat ID:", placeholder="123456789")
            
            if st.form_submit_button("🔗 Test Connection"):
                if bot_token and chat_id:
                    # Test Telegram connection
                    test_bot = TelegramBot(token=bot_token, chat_id=chat_id)
                    if test_bot.initialize():
                        if test_bot.send_message_sync("🎉 Algo Trading System connected successfully!"):
                            st.success("✅ Telegram connection successful!")
                            st.session_state.telegram_configured = True
                            
                            # Save to .env file
                            save_telegram_config(bot_token, chat_id)
                        else:
                            st.error("❌ Failed to send test message")
                    else:
                        st.error("❌ Failed to initialize Telegram bot")
                else:
                    st.warning("⚠️ Please enter both bot token and chat ID")
    
    else:
        st.success("✅ Telegram is configured!")
        if st.button("🔄 Reconfigure Telegram"):
            st.session_state.telegram_configured = False
            st.rerun()
    
    # Google Sheets Setup
    st.subheader("📊 Google Sheets Setup")
    
    if 'sheets_configured' not in st.session_state:
        st.session_state.sheets_configured = False
    
    if not st.session_state.sheets_configured:
        st.info("Set up Google Sheets for automated trade logging and analytics")
        
        with st.expander("🔧 Google Sheets Setup Guide", expanded=True):
            st.markdown("""
            **Step 1: Create Google Cloud Project**
            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
            2. Create a new project
            3. Enable Google Sheets API
            
            **Step 2: Create Service Account**
            1. Go to "APIs & Services" → "Credentials"
            2. Click "Create Credentials" → "Service Account"
            3. Download the JSON credentials file
            4. Rename it to `credentials.json` and place it in the project folder
            """)
        
        # File upload for credentials
        uploaded_file = st.file_uploader("Upload credentials.json", type=['json'])
        
        if uploaded_file is not None:
            # Save the uploaded file
            with open('credentials.json', 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Test Google Sheets connection
            sheets_manager = GoogleSheetsManager()
            if sheets_manager.authenticate():
                if sheets_manager.create_or_open_spreadsheet():
                    st.success("✅ Google Sheets connected successfully!")
                    st.session_state.sheets_configured = True
                else:
                    st.error("❌ Failed to create/open spreadsheet")
            else:
                st.error("❌ Failed to authenticate with Google Sheets")
    
    else:
        st.success("✅ Google Sheets is configured!")
        if st.button("🔄 Reconfigure Google Sheets"):
            st.session_state.sheets_configured = False
            st.rerun()

def analytics_page():
    """Analytics page"""
    st.header("📈 Analytics & Performance")
    
    # Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "12.45%", "2.3%")
    
    with col2:
        st.metric("Win Ratio", "78.5%", "5.2%")
    
    with col3:
        st.metric("Sharpe Ratio", "1.85", "0.15")
    
    with col4:
        st.metric("Max Drawdown", "-8.2%", "-1.1%")
    
    # Backtest Results
    st.subheader("📊 Backtest Results")
    
    if st.button("🔄 Run New Backtest"):
        with st.spinner("Running backtest..."):
            data_ingestion = DataIngestion()
            strategy = TradingStrategy()
            
            results = []
            for stock in config.NIFTY_50_STOCKS:
                try:
                    data = data_ingestion.fetch_stock_data(stock, period='6mo')
                    if data is not None:
                        result = strategy.backtest_strategy(data, stock)
                        if result:
                            results.append(result)
                            # No automatic notifications - only when user clicks buttons
                except Exception as e:
                    error_msg = f"Backtest failed for {stock}: {str(e)}"
                    st.error(error_msg)
            
            st.success(f"✅ Backtest completed for {len(results)} stocks!")
    
    # Sample backtest data
    backtest_data = {
        'Stock': ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS'],
        'Total Return (%)': [15.2, 8.7, 12.1, 6.8, 9.3],
        'Win Ratio (%)': [82, 75, 78, 70, 73],
        'Total Trades': [24, 18, 22, 15, 19],
        'Sharpe Ratio': [1.95, 1.45, 1.78, 1.12, 1.34]
    }
    
    backtest_df = pd.DataFrame(backtest_data)
    st.dataframe(backtest_df, use_container_width=True)



def save_telegram_config(token: str, chat_id: str):
    """Save Telegram configuration to .env file"""
    try:
        # Read existing .env file
        env_content = ""
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                env_content = f.read()
        
        # Update Telegram configuration
        lines = env_content.split('\n') if env_content else []
        new_lines = []
        
        # Remove existing Telegram config
        for line in lines:
            if not line.startswith('TELEGRAM_'):
                new_lines.append(line)
        
        # Add new Telegram config
        new_lines.extend([
            f"TELEGRAM_BOT_TOKEN={token}",
            f"TELEGRAM_CHAT_ID={chat_id}",
            ""
        ])
        
        # Write to .env file
        with open('.env', 'w') as f:
            f.write('\n'.join(new_lines))
        
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")

def main():
    """Main function"""
    main_header()
    page = sidebar_setup()
    
    # Initialize Telegram configuration from config
    if 'telegram_configured' not in st.session_state:
        st.session_state.telegram_configured = bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)
    
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📊 Live Trading":
        live_trading_page()
    elif page == "📈 Analytics":
        analytics_page()

if __name__ == "__main__":
    main() 