"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import FeatureEngineer
from config import QUALITY_MAPPING, ORDINAL_FEATURES


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_prepared_data(self):
        """Create sample prepared data for testing."""
        return pd.DataFrame({
            'SalePrice': [200000, 150000, 300000, 250000],
            'OverallQual': [8, 6, 9, 7],
            'GrLivArea': [2000, 1500, 3000, 1800],
            'ExterQual': ['Ex', 'Gd', 'Ex', 'TA'],
            'KitchenQual': ['Gd', 'TA', 'Ex', 'Gd'],
            'BsmtQual': ['Ex', 'Gd', 'NoBasement', 'TA'],
            'Neighborhood': ['CollgCr', 'CollgCr', 'NoRidge', 'CollgCr'],
            'HouseStyle': ['2Story', '1Story', '2Story', '1Story'],
            'LotArea': [8450, 9620, 15000, 8000]
        })
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    def test_encode_ordinal_features(self, engineer, sample_prepared_data):
        """Test ordinal feature encoding."""
        result = engineer.encode_ordinal_features(sample_prepared_data)
        
        # Check that ordinal features are converted to numbers
        assert pd.api.types.is_numeric_dtype(result['ExterQual'])
        assert pd.api.types.is_numeric_dtype(result['KitchenQual'])
        assert pd.api.types.is_numeric_dtype(result['BsmtQual'])
        
        # Check that mappings are applied correctly
        assert result['ExterQual'].iloc[0] == QUALITY_MAPPING['Ex']
        assert result['KitchenQual'].iloc[0] == QUALITY_MAPPING['Gd']
        assert result['BsmtQual'].iloc[2] == 0  # NoBasement should be 0
    
    def test_encode_nominal_features(self, engineer, sample_prepared_data):
        """Test nominal feature encoding."""
        # First encode ordinal features to remove them from nominal encoding
        df_ordinal = engineer.encode_ordinal_features(sample_prepared_data)
        
        result = engineer.encode_nominal_features(df_ordinal)
        
        # Check that nominal features are one-hot encoded
        assert 'Neighborhood_NoRidge' in result.columns
        assert 'Neighborhood_CollgCr' in result.columns
        assert 'HouseStyle_2Story' in result.columns
        assert 'HouseStyle_1Story' in result.columns
        
        # Check that original nominal columns are removed
        assert 'Neighborhood' not in result.columns
        assert 'HouseStyle' not in result.columns
    
    def test_scale_numerical_features(self, engineer, sample_prepared_data):
        """Test numerical feature scaling."""
        # First encode all features
        df_encoded = engineer.encode_ordinal_features(sample_prepared_data)
        df_encoded = engineer.encode_nominal_features(df_encoded)
        
        result = engineer.scale_numerical_features(df_encoded)
        
        # Check that numerical features are scaled (approximately mean=0, std=1)
        numerical_cols = result.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if col != 'SalePrice':  # Target is not scaled
                assert abs(result[col].mean()) < 1e-10  # Approximately 0
                assert abs(result[col].std() - 1.0) < 1e-10  # Approximately 1
    
    def test_apply_log_transform(self, engineer):
        """Test log transformation of target variable."""
        # Create right-skewed data
        y_skewed = pd.Series([100000, 150000, 200000, 500000, 1000000])
        
        y_transformed, was_transformed = engineer.apply_log_transform(y_skewed)
        
        assert was_transformed == True
        assert y_transformed.dtype == float
        
        # Check that transformation reduces skewness
        original_skew = y_skewed.skew()
        transformed_skew = pd.Series(y_transformed).skew()
        assert abs(transformed_skew) < abs(original_skew)
    
    def test_inverse_log_transform(self, engineer):
        """Test inverse log transformation."""
        # Original data
        y_original = pd.Series([100000, 150000, 200000])
        
        # Apply log transform
        y_log, _ = engineer.apply_log_transform(y_original)
        
        # Apply inverse transform
        y_restored = engineer.inverse_log_transform(y_log)
        
        # Check that we get back the original values (approximately)
        np.testing.assert_array_almost_equal(y_original.values, y_restored.values, decimal=5)
    
    def test_engineer_features_complete_pipeline(self, engineer, sample_prepared_data):
        """Test complete feature engineering pipeline."""
        result, y_transformed, was_log_transformed = engineer.engineer_features(
            sample_prepared_data
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_prepared_data.shape[0]
        assert 'SalePrice' in result.columns
        
        # Check that all features are numeric
        non_numeric_cols = result.select_dtypes(exclude=['int64', 'float64']).columns
        assert len(non_numeric_cols) == 0
        
        # Check that target is transformed if requested
        if was_log_transformed:
            assert 'SalePrice_log' in result.columns
            assert y_transformed is not None
    
    def test_get_feature_importance_data(self, engineer):
        """Test feature importance data preparation."""
        # Create a mock model with coef_ attribute
        from sklearn.linear_model import Ridge
        import numpy as np
        
        # Sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 3, 4, 5],
            'feature3': [3, 4, 5, 6]
        })
        y = pd.Series([1, 2, 3, 4])
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        importance_df = engineer.get_feature_importance_data(model, X.columns.tolist())
        
        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ['feature', 'coefficient']
        assert len(importance_df) == len(X.columns)
        assert importance_df['coefficient'].dtype == float


if __name__ == "__main__":
    pytest.main([__file__])
