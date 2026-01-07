"""
Inference script for housing prices prediction.

This script provides a command-line interface for making predictions
using trained models.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from utils import validate_file_exists, print_data_summary


class HousingPricePredictor:
    """
    Housing price prediction class for inference.
    """
    
    def __init__(self, model_path: str, feature_engineer_path: str):
        """
        Initialize the predictor with trained model and feature engineer.
        
        Args:
            model_path: Path to trained model file
            feature_engineer_path: Path to feature engineer object
        """
        self.model = None
        self.feature_engineer = None
        self.data_processor = DataProcessor()
        
        # Load model and feature engineer
        self.load_model(model_path)
        self.load_feature_engineer(feature_engineer_path)
    
    def load_model(self, model_path: str) -> None:
        """Load trained model from disk."""
        if not validate_file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from: {model_path}")
    
    def load_feature_engineer(self, feature_engineer_path: str) -> None:
        """Load feature engineer from disk."""
        if not validate_file_exists(feature_engineer_path):
            raise FileNotFoundError(f"Feature engineer file not found: {feature_engineer_path}")
        
        self.feature_engineer = joblib.load(feature_engineer_path)
        print(f"✓ Feature engineer loaded from: {feature_engineer_path}")
    
    def preprocess_single_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess a single housing record for prediction.
        
        Args:
            record: Dictionary containing housing features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([record])
        
        # Apply data processing
        df_processed = self.data_processor.handle_missing_values(df)
        df_processed = self.data_processor.correct_data_types(df_processed)
        
        # Apply feature engineering
        df_engineered, _, _ = self.feature_engineer.engineer_features(
            df_processed, 
            target_column='SalePrice' if 'SalePrice' in df_processed.columns else 'dummy_target',
            apply_log_target=False,  # Don't transform target for inference
            scale_features=True
        )
        
        # Remove target column if it exists
        if 'SalePrice' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['SalePrice'])
        if 'dummy_target' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['dummy_target'])
        if 'SalePrice_log' in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=['SalePrice_log'])
        
        return df_engineered
    
    def predict(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single housing record.
        
        Args:
            record: Dictionary containing housing features
            
        Returns:
            Dictionary with prediction and metadata
        """
        # Preprocess the record
        X = self.preprocess_single_record(record)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Apply inverse log transform if needed
        if hasattr(self.feature_engineer, 'inverse_log_transform'):
            try:
                prediction = self.feature_engineer.inverse_log_transform([prediction])[0]
            except:
                pass  # If transformation fails, use raw prediction
        
        return {
            'predicted_price': float(prediction),
            'prediction_confidence': 'medium',  # Could be calculated if needed
            'model_type': type(self.model.named_steps['model']).__name__ if hasattr(self.model, 'named_steps') else type(self.model).__name__
        }
    
    def predict_batch(self, records_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Make predictions for multiple records from a CSV file.
        
        Args:
            records_file: Path to CSV file with housing records
            output_file: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(records_file)
        print(f"Loaded {len(df)} records from {records_file}")
        
        # Preprocess all records
        df_processed = self.data_processor.handle_missing_values(df)
        df_processed = self.data_processor.correct_data_types(df_processed)
        
        # Apply feature engineering
        df_engineered, _, _ = self.feature_engineer.engineer_features(
            df_processed,
            target_column='SalePrice' if 'SalePrice' in df_processed.columns else 'dummy_target',
            apply_log_target=False,
            scale_features=True
        )
        
        # Prepare features for prediction
        X = df_engineered.copy()
        if 'SalePrice' in X.columns:
            X = X.drop(columns=['SalePrice'])
        if 'dummy_target' in X.columns:
            X = X.drop(columns=['dummy_target'])
        if 'SalePrice_log' in X.columns:
            X = X.drop(columns=['SalePrice_log'])
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Apply inverse log transform if needed
        if hasattr(self.feature_engineer, 'inverse_log_transform'):
            try:
                predictions = self.feature_engineer.inverse_log_transform(predictions)
            except:
                pass  # If transformation fails, use raw predictions
        
        # Create results DataFrame
        results = df.copy()
        results['predicted_price'] = predictions
        
        # Save if output file specified
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"✓ Predictions saved to: {output_file}")
        
        return results


def main():
    """Main inference script."""
    # Get base directory for absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description='Housing Price Prediction Inference')
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=os.path.join(base_dir, 'models', 'ridge_model.pkl'),
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--feature-engineer-path',
        type=str,
        default=os.path.join(base_dir, 'models', 'feature_engineer.pkl'),
        help='Path to feature engineer object'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to CSV file with housing records for batch prediction'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to save predictions (for batch prediction)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode for single predictions'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = HousingPricePredictor(args.model_path, args.feature_engineer_path)
        
        if args.interactive:
            # Interactive mode
            print("\n🏠 Housing Price Prediction - Interactive Mode")
            print("=" * 50)
            print("Enter housing features (press Enter to use default values):")
            
            # Example features - you can expand this list
            sample_record = {
                'OverallQual': 7,
                'GrLivArea': 2000,
                'TotalBsmtSF': 1000,
                'GarageCars': 2,
                'GarageArea': 500,
                'YearBuilt': 2000,
                'Neighborhood': 'CollgCr',
                'HouseStyle': '2Story',
                'ExterQual': 'Gd',
                'KitchenQual': 'TA',
                'BsmtQual': 'Gd',
                'BsmtCond': 'TA',
                'GarageType': 'Attchd',
                'GarageFinish': 'RFn',
                'GarageQual': 'TA',
                'GarageCond': 'TA'
            }
            
            # Get user input
            record = {}
            for key, default_value in sample_record.items():
                user_input = input(f"{key} [{default_value}]: ").strip()
                if user_input:
                    try:
                        # Try to convert to appropriate type
                        if isinstance(default_value, int):
                            record[key] = int(user_input)
                        elif isinstance(default_value, float):
                            record[key] = float(user_input)
                        else:
                            record[key] = user_input
                    except ValueError:
                        record[key] = default_value
                else:
                    record[key] = default_value
            
            # Make prediction
            result = predictor.predict(record)
            
            print(f"\n🎯 Prediction Result:")
            print(f"Predicted Price: ${result['predicted_price']:,.2f}")
            print(f"Model Type: {result['model_type']}")
            print(f"Confidence: {result['prediction_confidence']}")
        
        elif args.input_file:
            # Batch prediction mode
            print(f"\n📊 Batch Prediction Mode")
            print("=" * 50)
            
            results = predictor.predict_batch(args.input_file, args.output_file)
            
            print(f"\n📈 Prediction Summary:")
            print(f"Total predictions: {len(results)}")
            print(f"Average predicted price: ${results['predicted_price'].mean():,.2f}")
            print(f"Price range: ${results['predicted_price'].min():,.2f} - ${results['predicted_price'].max():,.2f}")
            
            if not args.output_file:
                print("\nFirst 5 predictions:")
                print(results[['predicted_price']].head())
        
        else:
            print("Please specify either --interactive or --input-file")
            print("Use --help for more information")
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
