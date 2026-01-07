"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor
from config import BASEMENT_FEATURES, GARAGE_FEATURES


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'Id': [1, 2, 3, 4],
            'SalePrice': [200000, 150000, 300000, 250000],
            'BsmtQual': ['Ex', 'Gd', np.nan, 'TA'],
            'BsmtCond': ['TA', np.nan, 'TA', 'Gd'],
            'BsmtExposure': ['Gd', 'No', np.nan, 'Mn'],
            'BsmtFinType1': ['GLQ', 'ALQ', np.nan, 'Unf'],
            'BsmtFinType2': ['Unf', np.nan, 'GLQ', 'Unf'],
            'GarageType': ['Attchd', np.nan, 'BuiltIn', 'Detchd'],
            'GarageYrBlt': [2000, np.nan, 2010, 1995],
            'GarageFinish': ['RFn', np.nan, 'Unf', 'RFn'],
            'GarageQual': ['TA', np.nan, 'TA', 'TA'],
            'GarageCond': ['TA', np.nan, 'TA', 'TA'],
            'LotFrontage': [60, np.nan, 80, np.nan],
            'Neighborhood': ['CollgCr', 'CollgCr', 'NoRidge', 'CollgCr'],
            'Electrical': ['SBrkr', 'SBrkr', np.nan, 'SBrkr'],
            'MSSubClass': [60, 20, 60, 50],
            'MasVnrType': ['BrkFace', np.nan, 'None', 'Stone'],
            'MasVnrArea': [200, np.nan, 0, 300],
            'FireplaceQu': ['Gd', np.nan, 'TA', 'Ex'],
            'PoolQC': [np.nan, np.nan, np.nan, np.nan],
            'Fence': [np.nan, 'MnPrv', np.nan, 'GdWo'],
            'MiscFeature': [np.nan, np.nan, 'Shed', np.nan],
            'Alley': [np.nan, np.nan, 'Grvl', np.nan]
        })
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance."""
        return DataProcessor()
    
    def test_load_data(self, processor, tmp_path):
        """Test data loading functionality."""
        # Create temporary CSV file
        sample_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        temp_file = tmp_path / "test_data.csv"
        sample_df.to_csv(temp_file, index=False)
        
        # Test loading
        loaded_df = processor.load_data(str(temp_file))
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (2, 2)
        assert list(loaded_df.columns) == ['A', 'B']
    
    def test_handle_missing_values_basement(self, processor, sample_data):
        """Test basement missing value handling."""
        result = processor.handle_missing_values(sample_data)
        
        # Check that basement NaN values are filled
        assert result['BsmtQual'].isnull().sum() == 0
        assert result['BsmtCond'].isnull().sum() == 0
        
        # Check that 'NoBasement' is used for missing values
        assert 'NoBasement' in result['BsmtQual'].values
        assert 'NoBasement' in result['BsmtCond'].values
    
    def test_handle_missing_values_garage(self, processor, sample_data):
        """Test garage missing value handling."""
        result = processor.handle_missing_values(sample_data)
        
        # Check that garage NaN values are filled
        assert result['GarageType'].isnull().sum() == 0
        assert result['GarageYrBlt'].isnull().sum() == 0
        
        # Check that 'NoGarage' is used for categorical garage features
        assert 'NoGarage' in result['GarageType'].values
        assert 0 in result['GarageYrBlt'].values  # Should be filled with 0
    
    def test_handle_missing_values_lot_frontage(self, processor, sample_data):
        """Test LotFrontage neighborhood-based imputation."""
        result = processor.handle_missing_values(sample_data)
        
        # Check that LotFrontage NaN values are filled
        assert result['LotFrontage'].isnull().sum() == 0
        
        # Check that imputed values are reasonable (should use neighborhood median)
        collgcr_data = result[result['Neighborhood'] == 'CollgCr']
        assert len(collgcr_data) == 3  # Should have 3 CollgCr entries
    
    def test_handle_missing_values_electrical(self, processor, sample_data):
        """Test electrical missing value handling."""
        result = processor.handle_missing_values(sample_data)
        
        # Check that electrical NaN values are filled
        assert result['Electrical'].isnull().sum() == 0
        
        # Should be filled with mode (most common value)
        assert result['Electrical'].value_counts().index[0] == 'SBrkr'
    
    def test_correct_data_types(self, processor, sample_data):
        """Test data type corrections."""
        result = processor.correct_data_types(sample_data)
        
        # Check that MSSubClass is converted to object (categorical)
        assert result['MSSubClass'].dtype == 'object'
    
    def test_get_missing_value_summary(self, processor, sample_data):
        """Test missing value summary generation."""
        summary = processor.get_missing_value_summary(sample_data)
        
        assert isinstance(summary, dict)
        assert len(summary) > 0  # Should have some missing values
        
        # Check that percentages are reasonable
        for col, percentage in summary.items():
            assert 0 <= percentage <= 100
    
    def test_validate_data_quality(self, processor, sample_data):
        """Test data quality validation."""
        # Process the data first
        processed_data = processor.handle_missing_values(sample_data)
        corrected_data = processor.correct_data_types(processed_data)
        
        # Validate quality
        is_valid = processor.validate_data_quality(corrected_data)
        
        assert isinstance(is_valid, bool)
    
    def test_prepare_data_complete_pipeline(self, processor, sample_data, tmp_path):
        """Test complete data preparation pipeline."""
        # Save sample data to temporary file
        temp_file = tmp_path / "sample_data.csv"
        sample_data.to_csv(temp_file, index=False)
        
        # Run complete pipeline
        result = processor.prepare_data(str(temp_file))
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == sample_data.shape[0]  # Same number of rows
        assert 'SalePrice' in result.columns
        
        # Check that missing values are handled
        assert result.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__])
