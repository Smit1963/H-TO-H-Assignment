"""
Google Sheets Automation Module
Handles logging trades, P&L, and analytics to Google Sheets
"""

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Optional
import config
import os

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class GoogleSheetsManager:
    """
    Manages Google Sheets operations for algo trading system
    """
    
    def __init__(self, credentials_file: str = config.GOOGLE_SHEETS_CREDENTIALS_FILE):
        self.credentials_file = credentials_file
        self.client = None
        self.spreadsheet = None
        self.sheets = {}
        
    def authenticate(self) -> bool:
        """
        Authenticate with Google Sheets API
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Define the scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Check if credentials file exists
            if not os.path.exists(self.credentials_file):
                logger.error(f"Credentials file not found: {self.credentials_file}")
                logger.error("Please download credentials.json from Google Cloud Console")
                return False
            
            # Load credentials
            credentials = Credentials.from_service_account_file(
                self.credentials_file, 
                scopes=scope
            )
            
            # Create client
            self.client = gspread.authorize(credentials)
            
            # Create service for advanced operations
            from googleapiclient.discovery import build
            self.service = build('sheets', 'v4', credentials=credentials)
            
            logger.info("Successfully authenticated with Google Sheets")
            return True
            
        except Exception as e:
            logger.error(f"Error authenticating with Google Sheets: {str(e)}")
            if "API has not been used" in str(e) or "disabled" in str(e):
                logger.error("Google Drive API not enabled. Please enable it at:")
                logger.error("https://console.developers.google.com/apis/api/drive.googleapis.com/overview")
            return False
    
    def create_or_open_spreadsheet(self) -> bool:
        """
        Create or open a Google Sheets spreadsheet
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to create a new spreadsheet
            spreadsheet = {
                'properties': {
                    'title': f'Algo Trading System - {datetime.now().strftime("%Y-%m-%d")}'
                },
                'sheets': [
                    {
                        'properties': {
                            'title': config.TRADE_LOG_SHEET,
                            'gridProperties': {
                                'rowCount': 1000,
                                'columnCount': 10
                            }
                        }
                    },
                    {
                        'properties': {
                            'title': config.SUMMARY_SHEET,
                            'gridProperties': {
                                'rowCount': 100,
                                'columnCount': 10
                            }
                        }
                    }
                ]
            }
            
            try:
                # Try to create spreadsheet using gspread (simpler approach)
                spreadsheet_title = f'Algo Trading System - {datetime.now().strftime("%Y-%m-%d")}'
                self.spreadsheet = self.client.create(spreadsheet_title)
                self.spreadsheet_id = self.spreadsheet.id
                logger.info(f"Created new spreadsheet: {self.spreadsheet_id}")
                
                # Initialize sheets
                self._initialize_sheets()
                return True
                
            except Exception as e:
                if "storage quota" in str(e).lower():
                    logger.error("Google Drive storage quota exceeded. Please free up space or upgrade storage.")
                    logger.info("Alternative: Use existing spreadsheet or local CSV logging")
                    return self._use_existing_spreadsheet()
                elif "permission" in str(e).lower() or "403" in str(e):
                    logger.warning("Permission denied for Google Sheets. Using local CSV logging instead.")
                    logger.info("This is normal if service account doesn't have spreadsheet creation permissions.")
                    return self._use_existing_spreadsheet()
                else:
                    logger.error(f"Error creating spreadsheet: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating/opening spreadsheet: {str(e)}")
            return False
    
    def _use_existing_spreadsheet(self) -> bool:
        """
        Use existing spreadsheet or create local CSV logging as fallback
        """
        try:
            # Try to use existing spreadsheet if available
            if hasattr(self, 'spreadsheet_id') and self.spreadsheet_id:
                logger.info("Using existing spreadsheet")
                return True
            else:
                # Create local CSV logging as fallback
                logger.info("Creating local CSV logging as fallback")
                self._create_local_csv_files()
                return True
        except Exception as e:
            logger.error(f"Error setting up fallback logging: {str(e)}")
            return False
    
    def _create_local_csv_files(self):
        """
        Create local CSV files for logging when Google Sheets is not available
        """
        try:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # Create CSV files
            trade_log_file = 'logs/trade_log.csv'
            summary_file = 'logs/summary.csv'
            
            # Create trade log CSV
            if not os.path.exists(trade_log_file):
                trade_headers = ['Date', 'Symbol', 'Action', 'Price', 'Shares', 'Value', 'Capital', 'P&L', 'Cumulative P&L']
                pd.DataFrame(columns=trade_headers).to_csv(trade_log_file, index=False)
            
            # Create summary CSV
            if not os.path.exists(summary_file):
                summary_headers = ['Date', 'Symbol', 'Total Return (%)', 'Total Trades', 'Win Ratio (%)', 'Max Drawdown (%)', 'Sharpe Ratio']
                pd.DataFrame(columns=summary_headers).to_csv(summary_file, index=False)
            
            logger.info("Local CSV logging files created successfully")
            
        except Exception as e:
            logger.error(f"Error creating local CSV files: {str(e)}")
    
    def _initialize_sheets(self):
        """
        Initialize required sheets in the spreadsheet
        """
        try:
            # Create or get required sheets
            sheet_names = [config.TRADE_LOG_SHEET, config.SUMMARY_SHEET, config.ANALYTICS_SHEET]
            
            for sheet_name in sheet_names:
                try:
                    sheet = self.spreadsheet.worksheet(sheet_name)
                    self.sheets[sheet_name] = sheet
                except gspread.WorksheetNotFound:
                    sheet = self.spreadsheet.add_worksheet(
                        title=sheet_name, 
                        rows=1000, 
                        cols=20
                    )
                    self.sheets[sheet_name] = sheet
                    self._setup_sheet_headers(sheet_name)
            
            logger.info("Initialized all required sheets")
            
        except Exception as e:
            logger.error(f"Error initializing sheets: {str(e)}")
    
    def _setup_sheet_headers(self, sheet_name: str):
        """
        Setup headers for different sheets
        
        Args:
            sheet_name (str): Name of the sheet
        """
        try:
            sheet = self.sheets[sheet_name]
            
            if sheet_name == config.TRADE_LOG_SHEET:
                headers = [
                    'Date', 'Symbol', 'Action', 'Price', 'Shares', 
                    'Value', 'Capital', 'P&L', 'Cumulative P&L'
                ]
            elif sheet_name == config.SUMMARY_SHEET:
                headers = [
                    'Date', 'Symbol', 'Total Return (%)', 'Total Trades', 
                    'Win Ratio (%)', 'Max Drawdown (%)', 'Sharpe Ratio'
                ]
            elif sheet_name == config.ANALYTICS_SHEET:
                headers = [
                    'Date', 'Symbol', 'RSI', 'MACD', 'MA Crossover', 
                    'ML Prediction', 'ML Confidence', 'Signal'
                ]
            
            # Clear existing data and add headers
            sheet.clear()
            sheet.append_row(headers)
            logger.info(f"Setup headers for {sheet_name}")
            
        except Exception as e:
            logger.error(f"Error setting up headers for {sheet_name}: {str(e)}")
    
    def log_trade(self, trade_data: Dict) -> bool:
        """
        Log a trade to the trade log sheet
        
        Args:
            trade_data (Dict): Trade information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if config.TRADE_LOG_SHEET not in self.sheets:
                logger.error("Trade log sheet not found")
                return False
            
            sheet = self.sheets[config.TRADE_LOG_SHEET]
            
            # Calculate P&L if this is a sell trade
            pnl = 0
            cumulative_pnl = 0
            
            if trade_data['action'] == 'SELL':
                # Find corresponding buy trade
                buy_trades = sheet.get_all_records()
                for buy_trade in buy_trades:
                    if (buy_trade['Symbol'] == trade_data['symbol'] and 
                        buy_trade['Action'] == 'BUY' and 
                        buy_trade['Shares'] == trade_data['shares']):
                        buy_price = float(buy_trade['Price'])
                        sell_price = trade_data['price']
                        pnl = (sell_price - buy_price) * trade_data['shares']
                        break
            
            # Calculate cumulative P&L
            all_trades = sheet.get_all_records()
            cumulative_pnl = sum(float(trade.get('P&L', 0)) for trade in all_trades) + pnl
            
            # Prepare row data
            row_data = [
                trade_data['date'].strftime('%Y-%m-%d %H:%M:%S'),
                trade_data['symbol'],
                trade_data['action'],
                trade_data['price'],
                trade_data['shares'],
                trade_data['value'],
                trade_data['capital'],
                pnl,
                cumulative_pnl
            ]
            
            # Append to sheet
            sheet.append_row(row_data)
            logger.info(f"Logged {trade_data['action']} trade for {trade_data['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            return False
    
    def log_summary(self, summary_data: Dict) -> bool:
        """
        Log summary data to the summary sheet
        
        Args:
            summary_data (Dict): Summary information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if config.SUMMARY_SHEET not in self.sheets:
                logger.error("Summary sheet not found")
                return False
            
            sheet = self.sheets[config.SUMMARY_SHEET]
            
            # Prepare row data
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                summary_data['symbol'],
                summary_data['total_return'],
                summary_data['total_trades'],
                summary_data['win_ratio'],
                summary_data['max_drawdown'],
                summary_data['sharpe_ratio']
            ]
            
            # Append to sheet
            sheet.append_row(row_data)
            logger.info(f"Logged summary for {summary_data['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging summary: {str(e)}")
            return False
    
    def log_analytics(self, analytics_data: Dict) -> bool:
        """
        Log analytics data to the analytics sheet
        
        Args:
            analytics_data (Dict): Analytics information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if config.ANALYTICS_SHEET not in self.sheets:
                logger.error("Analytics sheet not found")
                return False
            
            sheet = self.sheets[config.ANALYTICS_SHEET]
            
            # Prepare row data
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                analytics_data['symbol'],
                analytics_data.get('rsi', 0),
                analytics_data.get('macd', 0),
                analytics_data.get('ma_crossover', 0),
                analytics_data.get('ml_prediction', 'N/A'),
                analytics_data.get('ml_confidence', 0),
                analytics_data.get('signal', 'HOLD')
            ]
            
            # Append to sheet
            sheet.append_row(row_data)
            logger.info(f"Logged analytics for {analytics_data['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging analytics: {str(e)}")
            return False
    
    def update_portfolio_summary(self, portfolio_data: Dict) -> bool:
        """
        Update portfolio summary in a dedicated sheet
        
        Args:
            portfolio_data (Dict): Portfolio information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create portfolio summary sheet if it doesn't exist
            portfolio_sheet_name = 'Portfolio Summary'
            
            try:
                portfolio_sheet = self.spreadsheet.worksheet(portfolio_sheet_name)
            except gspread.WorksheetNotFound:
                portfolio_sheet = self.spreadsheet.add_worksheet(
                    title=portfolio_sheet_name, 
                    rows=100, 
                    cols=10
                )
                # Setup headers
                headers = [
                    'Date', 'Total Portfolio Value', 'Cash', 'Invested Amount',
                    'Total P&L', 'Total Return (%)', 'Number of Positions'
                ]
                portfolio_sheet.clear()
                portfolio_sheet.append_row(headers)
            
            # Prepare row data
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                portfolio_data['total_value'],
                portfolio_data['cash'],
                portfolio_data['invested_amount'],
                portfolio_data['total_pnl'],
                portfolio_data['total_return'],
                portfolio_data['num_positions']
            ]
            
            # Append to sheet
            portfolio_sheet.append_row(row_data)
            logger.info("Updated portfolio summary")
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio summary: {str(e)}")
            return False
    
    def get_trade_history(self, symbol: str = None) -> List[Dict]:
        """
        Get trade history from the trade log sheet
        
        Args:
            symbol (str): Filter by symbol (optional)
            
        Returns:
            List[Dict]: List of trades
        """
        try:
            if config.TRADE_LOG_SHEET not in self.sheets:
                logger.error("Trade log sheet not found")
                return []
            
            sheet = self.sheets[config.TRADE_LOG_SHEET]
            trades = sheet.get_all_records()
            
            if symbol:
                trades = [trade for trade in trades if trade['Symbol'] == symbol]
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def clear_old_data(self, days: int = 30) -> bool:
        """
        Clear old data from sheets (keep last N days)
        
        Args:
            days (int): Number of days to keep
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            
            for sheet_name, sheet in self.sheets.items():
                if sheet_name == config.TRADE_LOG_SHEET:
                    # Keep headers
                    all_data = sheet.get_all_values()
                    headers = all_data[0]
                    data_rows = all_data[1:]
                    
                    # Filter by date
                    filtered_rows = []
                    for row in data_rows:
                        try:
                            row_date = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                            if row_date >= cutoff_date:
                                filtered_rows.append(row)
                        except:
                            # Keep rows with invalid dates
                            filtered_rows.append(row)
                    
                    # Update sheet
                    sheet.clear()
                    sheet.append_row(headers)
                    for row in filtered_rows:
                        sheet.append_row(row)
            
            logger.info(f"Cleared data older than {days} days")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing old data: {str(e)}")
            return False

if __name__ == "__main__":
    # Test Google Sheets functionality
    sheets_manager = GoogleSheetsManager()
    
    if sheets_manager.authenticate():
        if sheets_manager.create_or_open_spreadsheet():
            print("Google Sheets setup successful")
            
            # Test logging
            test_trade = {
                'date': datetime.now(),
                'symbol': 'RELIANCE.NS',
                'action': 'BUY',
                'price': 2500.0,
                'shares': 10,
                'value': 25000.0,
                'capital': 100000.0
            }
            
            sheets_manager.log_trade(test_trade)
        else:
            print("Failed to create/open spreadsheet")
    else:
        print("Failed to authenticate with Google Sheets") 