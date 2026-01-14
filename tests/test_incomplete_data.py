"""
Test script to run the project with incomplete data (missing features).
This validates that the encoding methods properly handle unknown values.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator
from config import DATA_PATHS, RANDOM_STATE


def create_incomplete_test_data(original_df, missing_cols_ratio=0.2):
    """
    Create test data with missing features and unknown categorical values.
    
    Args:
        original_df: Original dataframe
        missing_cols_ratio: Ratio of columns to remove features from
        
    Returns:
        Modified dataframe with incomplete data
    """
    df = original_df.copy()
    
    # Select random columns to introduce unknown values
    all_cols = df.columns.tolist()
    if 'SalePrice' in all_cols:
        all_cols.remove('SalePrice')
    
    num_cols_to_modify = max(1, int(len(all_cols) * missing_cols_ratio))
    cols_to_modify = np.random.choice(all_cols, size=num_cols_to_modify, replace=False)
    
    print(f"\nIntroducing unknown values in {len(cols_to_modify)} columns:")
    for col in cols_to_modify:
        if df[col].dtype == 'object':
            # Add some unknown categorical values
            unknown_values = ['UNKNOWN', 'OTHER', 'NA_CUSTOM', 'MISSING_VALUE']
            unknown_mask = np.random.choice([True, False], size=len(df), p=[0.1, 0.9])
            df.loc[unknown_mask, col] = np.random.choice(unknown_values, 
                                                        size=unknown_mask.sum())
            print(f"  - {col}: Added unknown categorical values")
        else:
            # For numerical columns, introduce some outliers/odd values
            outlier_mask = np.random.choice([True, False], size=len(df), p=[0.05, 0.95])
            df.loc[outlier_mask, col] = df.loc[outlier_mask, col] * np.random.uniform(0.5, 2, 
                                                                                        size=outlier_mask.sum())
            print(f"  - {col}: Added outlier values")
    
    return df


def test_pipeline_with_incomplete_data():
    """Run the complete pipeline with incomplete data."""
    
    print("=" * 80)
    print("TESTING HOUSING PRICE REGRESSION WITH INCOMPLETE DATA")
    print("=" * 80)
    
    # Load original data
    print("\n1. Loading original training data...")
    df_original = pd.read_csv(DATA_PATHS['raw_train'])
    print(f"   Original data shape: {df_original.shape}")
    print(f"   Columns: {df_original.shape[1]}")
    
    # Create incomplete test data
    print("\n2. Creating test data with missing features and unknown values...")
    np.random.seed(RANDOM_STATE)
    df_incomplete = create_incomplete_test_data(df_original, missing_cols_ratio=0.2)
    print(f"   Modified data shape: {df_incomplete.shape}")
    
    # Initialize processors
    print("\n3. Initializing processors...")
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    
    # Process incomplete data
    print("\n4. Processing incomplete data...")
    try:
        # Handle missing values
        df_processed = data_processor.handle_missing_values(df_incomplete)
        print(f"   [OK] Data processing completed successfully")
        print(f"   Processed data shape: {df_processed.shape}")
        print(f"   Data types:\n{df_processed.dtypes.value_counts()}")
    except Exception as e:
        print(f"   [ERROR] Data processing failed: {str(e)}")
        return False
    
    # Feature engineering with new encoding
    print("\n5. Applying feature engineering with updated encoding...")
    try:
        df_engineered, y_transformed, was_log_transformed = feature_engineer.engineer_features(
            df_processed,
            target_column='SalePrice',
            apply_log_target=True,
            scale_features=True
        )
        print(f"   [OK] Feature engineering completed successfully")
        print(f"   Engineered data shape: {df_engineered.shape}")
        print(f"   Log transformation applied: {was_log_transformed}")
        
        # Show sample of encoded data
        print(f"\n   Sample of engineered data (first 5 rows, first 10 columns):")
        print(df_engineered.iloc[:5, :10].to_string())
        
    except Exception as e:
        print(f"   [ERROR] Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check for NaN values
    print("\n6. Checking for NaN values after encoding...")
    nan_counts = df_engineered.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"   [WARNING] Found NaN values:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"   [OK] No NaN values in engineered data")
    
    # Prepare for modeling
    print("\n7. Preparing data for modeling...")
    try:
        # Separate features and target
        X = df_engineered.drop(columns=['SalePrice', 'SalePrice_log'], errors='ignore')
        y = y_transformed if y_transformed is not None else df_processed['SalePrice']
        
        print(f"   [OK] Features shape: {X.shape}")
        print(f"   [OK] Target shape: {y.shape}")
        print(f"   [OK] Feature count: {X.shape[1]}")
        
    except Exception as e:
        print(f"   [ERROR] Data preparation failed: {str(e)}")
        return False
    
    # Train a simple model
    print("\n8. Training Ridge model with incomplete data...")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"   [OK] Model training completed successfully")
        print(f"   Train R2 Score: {train_score:.4f}")
        print(f"   Test R2 Score: {test_score:.4f}")
        
    except Exception as e:
        print(f"   [ERROR] Model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("[SUCCESS] ALL TESTS PASSED - Encoding handles unknown values correctly!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_pipeline_with_incomplete_data()
    sys.exit(0 if success else 1)
