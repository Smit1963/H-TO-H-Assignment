"""
Machine Learning Module
Implements ML models for stock price prediction
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class MLModel:
    """
    Machine Learning model for stock price prediction
    """
    
    def __init__(self, model_type: str = 'decision_tree'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = config.ML_FEATURES
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            pd.DataFrame: Data with prepared features
        """
        try:
            features = data.copy()
            
            # Create target variable (next day price movement)
            features['target'] = np.where(features['Close'].shift(-1) > features['Close'], 1, 0)
            
            # Add lagged features
            features['rsi_lag1'] = features['rsi'].shift(1)
            features['rsi_lag2'] = features['rsi'].shift(2)
            features['macd_lag1'] = features['macd'].shift(1)
            features['volume_ratio_lag1'] = features['volume_ratio'].shift(1)
            
            # Add price momentum features
            features['price_momentum_5'] = features['Close'].pct_change(5)
            features['price_momentum_10'] = features['Close'].pct_change(10)
            
            # Add volatility features
            features['volatility_5'] = features['Close'].rolling(5).std()
            features['volatility_10'] = features['Close'].rolling(10).std()
            
            # Add trend features
            features['trend_5'] = np.where(features['ma_short'] > features['ma_long'], 1, 0)
            features['trend_10'] = np.where(features['Close'] > features['Close'].rolling(10).mean(), 1, 0)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            logger.info(f"Prepared {len(features)} samples with {len(self.feature_columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """
        Train the ML model
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            Dict: Training results and metrics
        """
        try:
            logger.info(f"Training {self.model_type} model")
            
            # Prepare features
            features = self.prepare_features(data)
            
            if features.empty:
                logger.error("No features available for training")
                return {}
            
            # Select features and target
            X = features[self.feature_columns]
            y = features['target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1-config.TRAIN_TEST_SPLIT, 
                random_state=config.RANDOM_STATE, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            if self.model_type == 'decision_tree':
                self.model = DecisionTreeClassifier(
                    max_depth=10, 
                    min_samples_split=5, 
                    min_samples_leaf=2,
                    random_state=config.RANDOM_STATE
                )
            elif self.model_type == 'logistic_regression':
                self.model = LogisticRegression(
                    random_state=config.RANDOM_STATE,
                    max_iter=1000
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='accuracy'
            )
            
            # Feature importance (for decision tree)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_columns):
                    feature_importance[feature] = self.model.feature_importances_[i]
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            results = {
                'model_type': self.model_type,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'classification_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'test_samples': len(X_test),
                'train_samples': len(X_train)
            }
            
            self.is_trained = True
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {}
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Make predictions using trained model
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            Dict: Prediction results
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Model not trained. Please train the model first.")
                return {}
            
            # Prepare features
            features = self.prepare_features(data)
            
            if features.empty:
                logger.error("No features available for prediction")
                return {}
            
            # Check if we have enough data for prediction
            if len(features) < 1:
                logger.error("Insufficient data for prediction")
                return {}
            
            # Use the latest data point for prediction - only select feature columns
            latest_features = features[self.feature_columns].iloc[-1:].values
            
            if latest_features.size == 0:
                logger.error("No valid features for prediction")
                return {}
            
            # Convert to float and handle NaN values properly
            try:
                latest_features = latest_features.astype(float)
                # Replace NaN with 0
                latest_features = np.nan_to_num(latest_features, nan=0.0)
            except Exception as e:
                logger.error(f"Error converting features to float: {str(e)}")
                return {}
            
            # Scale features if scaler is available
            if hasattr(self, 'scaler') and self.scaler is not None:
                try:
                    latest_features = self.scaler.transform(latest_features)
                except Exception as e:
                    logger.error(f"Error scaling features: {str(e)}")
                    return {}
            
            # Make prediction
            try:
                prediction = self.model.predict(latest_features)[0]
                probabilities = self.model.predict_proba(latest_features)[0]
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                return {}
            
            # Calculate confidence
            confidence = max(probabilities) * 100
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probability_up': probabilities[1] if len(probabilities) > 1 else 0,
                'probability_down': probabilities[0] if len(probabilities) > 0 else 0
            }
            
            logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {}
    
    def predict_multiple_days(self, data: pd.DataFrame, days: int = 5) -> List[Dict]:
        """
        Predict multiple days ahead
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            days (int): Number of days to predict
            
        Returns:
            List[Dict]: List of predictions
        """
        try:
            if not self.is_trained or self.model is None:
                logger.error("Model not trained. Please train the model first.")
                return []
            
            # Prepare features
            features = self.prepare_features(data)
            
            if features.empty:
                logger.error("No features available for prediction")
                return []
            
            predictions = []
            
            # Make predictions for the last 'days' data points
            for i in range(min(days, len(features))):
                idx = -(i + 1)
                X = features[self.feature_columns].iloc[idx:idx+1].values
                X_scaled = self.scaler.transform(X)
                
                prediction = self.model.predict(X_scaled)[0]
                probability = self.model.predict_proba(X_scaled)[0]
                
                data_point = features.iloc[idx]
                
                pred_result = {
                    'date': data_point.name,
                    'prediction': int(prediction),
                    'probability_up': float(probability[1]),
                    'probability_down': float(probability[0]),
                    'confidence': float(max(probability)),
                    'actual_price': float(data_point['Close']),
                    'signal': 'BUY' if prediction == 1 else 'SELL'
                }
                
                predictions.append(pred_result)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making multiple predictions: {str(e)}")
            return []
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to file
        
        Args:
            filepath (str): Path to save model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.error("No trained model to save")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from file
        
        Args:
            filepath (str): Path to load model from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_summary(self) -> Dict:
        """
        Get model summary and statistics
        
        Returns:
            Dict: Model summary
        """
        if not self.is_trained:
            return {'status': 'Model not trained'}
        
        summary = {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),
            'is_trained': self.is_trained
        }
        
        if hasattr(self.model, 'feature_importances_'):
            summary['feature_importance'] = dict(zip(
                self.feature_columns, 
                self.model.feature_importances_
            ))
        
        return summary

if __name__ == "__main__":
    # Test ML model
    from data_ingestion import DataIngestion
    
    # Fetch data
    data_ingestion = DataIngestion()
    stocks_data = data_ingestion.fetch_multiple_stocks(config.NIFTY_50_STOCKS[:2])
    
    # Test ML model
    for symbol, data in stocks_data.items():
        print(f"\n=== Training ML Model for {symbol} ===")
        
        ml_model = MLModel('decision_tree')
        training_results = ml_model.train_model(data)
        
        if training_results:
            print(f"Accuracy: {training_results['accuracy']:.4f}")
            print(f"CV Mean: {training_results['cv_mean']:.4f}")
            print(f"CV Std: {training_results['cv_std']:.4f}")
            
            # Make prediction
            prediction = ml_model.predict(data)
            if prediction:
                print(f"Latest Prediction: {prediction['prediction']}")
                print(f"Confidence: {prediction['confidence']:.2f}%")
            
            # Save model
            ml_model.save_model(f'model_{symbol.replace(".NS", "")}.pkl') 