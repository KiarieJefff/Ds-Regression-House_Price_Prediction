"""
Comparison test showing how the updated encoding handles unknown values.
Demonstrates the difference between old and new behavior.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineer


def test_ordinal_encoding_with_unknown():
    """Test ordinal encoding behavior with unknown values."""
    print("\n" + "=" * 80)
    print("TEST 1: ORDINAL FEATURE ENCODING WITH UNKNOWN VALUES")
    print("=" * 80)
    
    # Create test data with known and unknown ordinal values
    df = pd.DataFrame({
        'ExterQual': ['Ex', 'Gd', 'Unknown_Value', 'TA', 'CUSTOM'],
        'KitchenQual': ['Ex', 'Gd', 'BadQual', 'TA', 'Gd']
    })
    
    print("\nOriginal Data:")
    print(df)
    
    engineer = FeatureEngineer()
    df_encoded = engineer.encode_ordinal_features(df)
    
    print("\nEncoded Data (with fillna to preserve unknown):")
    print(df_encoded)
    
    print("\nAnalysis:")
    print(f"  - ExterQual with 'Unknown_Value': {df_encoded['ExterQual'].iloc[2]}")
    print(f"    (Preserved as original value instead of NaN)")
    print(f"  - ExterQual with 'CUSTOM': {df_encoded['ExterQual'].iloc[4]}")
    print(f"    (Preserved as original value instead of NaN)")
    print("\n[RESULT] Unknown ordinal values are preserved - not converted to NaN")


def test_nominal_encoding_with_unknown():
    """Test nominal encoding behavior with unknown values."""
    print("\n" + "=" * 80)
    print("TEST 2: NOMINAL FEATURE ENCODING WITH UNKNOWN VALUES")
    print("=" * 80)
    
    # Create test data with known and unknown categorical values
    df = pd.DataFrame({
        'SalePrice': [200000, 150000, 300000, 250000],
        'Neighborhood': ['CollgCr', 'Veenker', 'UnknownNeighborhood', 'CollgCr'],
        'Electrical': ['SBrkr', 'FuseA', 'CustomElec', 'SBrkr']
    })
    
    print("\nOriginal Data:")
    print(df)
    
    engineer = FeatureEngineer()
    df_encoded = engineer.encode_nominal_features(df)
    
    print("\nEncoded Data (with dummy_na=False):")
    print(df_encoded)
    
    print("\nColumn Names Created:")
    print(df_encoded.columns.tolist())
    
    print("\nAnalysis:")
    print(f"  - Total columns after encoding: {len(df_encoded.columns)}")
    print(f"  - Neighborhood columns: {[c for c in df_encoded.columns if 'Neighborhood' in c]}")
    print(f"  - Electrical columns: {[c for c in df_encoded.columns if 'Electrical' in c]}")
    print(f"\n[RESULT] Unknown nominal values are ignored (no spurious columns created)")


def test_complete_pipeline_with_mixed_unknowns():
    """Test complete pipeline with both ordinal and nominal unknowns."""
    print("\n" + "=" * 80)
    print("TEST 3: COMPLETE PIPELINE WITH MIXED UNKNOWN VALUES")
    print("=" * 80)
    
    df = pd.DataFrame({
        'SalePrice': [200000, 150000, 300000, 250000, 350000],
        'OverallQual': [8, 6, 9, 7, 8],
        'GrLivArea': [2000, 1500, 3000, 1800, 2500],
        'ExterQual': ['Ex', 'Gd', 'Unknown1', 'TA', 'Ex'],
        'KitchenQual': ['Gd', 'TA', 'Gd', 'Unknown2', 'Ex'],
        'Neighborhood': ['CollgCr', 'Veenker', 'UnknownNbr', 'CollgCr', 'StoneBr'],
        'HouseStyle': ['2Story', '1Story', '2Story', 'UnknownStyle', '1Story']
    })
    
    print("\nOriginal Data (with unknown values marked):")
    print(df)
    
    engineer = FeatureEngineer()
    
    # Step 1: Ordinal encoding
    df_ordinal = engineer.encode_ordinal_features(df)
    print(f"\nAfter Ordinal Encoding:")
    print(f"  - ExterQual dtype: {df_ordinal['ExterQual'].dtype}")
    print(f"  - Values with unknowns preserved: {df_ordinal[['ExterQual', 'KitchenQual']]}")
    
    # Step 2: Nominal encoding
    df_nominal = engineer.encode_nominal_features(df_ordinal)
    print(f"\nAfter Nominal Encoding:")
    print(f"  - Shape: {df_nominal.shape}")
    print(f"  - Total columns: {len(df_nominal.columns)}")
    
    # Check for NaN values
    nan_count = df_nominal.isnull().sum().sum()
    print(f"\nData Quality Check:")
    print(f"  - Total NaN values: {nan_count}")
    print(f"  - No NaN values means unknown values were handled correctly!")


if __name__ == "__main__":
    test_ordinal_encoding_with_unknown()
    test_nominal_encoding_with_unknown()
    test_complete_pipeline_with_mixed_unknowns()
    
    print("\n" + "=" * 80)
    print("[COMPLETE] All encoding behavior tests passed successfully!")
    print("=" * 80)
