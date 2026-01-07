"""
Unit tests for utility functions.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    setup_project_structure, validate_file_exists, load_data_safely,
    save_data_safely, get_data_summary, print_data_summary,
    detect_outliers, print_outlier_summary, save_results_to_json,
    load_results_from_json, validate_model_inputs, print_validation_issues
)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_setup_project_structure(self, tmp_path):
        """Test project structure setup."""
        base_path = str(tmp_path)
        
        setup_project_structure(base_path)
        
        # Check that directories are created
        expected_dirs = [
            'data/raw', 'data/processed', 'models', 'reports',
            'notebooks', 'src'
        ]
        
        for dir_path in expected_dirs:
            full_path = os.path.join(base_path, dir_path)
            assert os.path.exists(full_path)
            assert os.path.isdir(full_path)
    
    def test_validate_file_exists(self, tmp_path):
        """Test file existence validation."""
        # Test existing file
        temp_file = tmp_path / "test.txt"
        temp_file.write_text("test content")
        
        assert validate_file_exists(str(temp_file)) == True
        
        # Test non-existing file
        non_existent = tmp_path / "non_existent.txt"
        assert validate_file_exists(str(non_existent)) == False
    
    def test_load_data_safely(self, tmp_path):
        """Test safe data loading."""
        # Create test CSV file
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        temp_file = tmp_path / "test_data.csv"
        test_data.to_csv(temp_file, index=False)
        
        # Test loading existing file
        loaded_df = load_data_safely(str(temp_file))
        
        assert loaded_df is not None
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 2)
        assert list(loaded_df.columns) == ['A', 'B']
        
        # Test loading non-existing file
        non_existent = tmp_path / "non_existent.csv"
        loaded_df = load_data_safely(str(non_existent))
        assert loaded_df is None
    
    def test_save_data_safely(self, tmp_path):
        """Test safe data saving."""
        test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        temp_file = tmp_path / "subdir" / "test_save.csv"
        
        # Test saving
        result = save_data_safely(test_data, str(temp_file))
        
        assert result == True
        assert os.path.exists(temp_file)
        
        # Verify saved data
        loaded_data = pd.read_csv(temp_file)
        pd.testing.assert_frame_equal(test_data, loaded_data)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B'],
            'missing_col': [1, np.nan, 3, 4, np.nan]
        })
        
        summary = get_data_summary(test_data)
        
        assert isinstance(summary, dict)
        assert summary['shape'] == (5, 3)
        assert summary['columns'] == 3
        assert 'dtypes' in summary
        assert 'missing_values' in summary
        assert 'numeric_columns' in summary
        assert 'categorical_columns' in summary
        assert 'memory_usage' in summary
        
        # Check missing values
        assert summary['missing_values']['missing_col'] == 2
        assert summary['missing_values']['numeric_col'] == 0
    
    def test_print_data_summary(self, capsys):
        """Test data summary printing."""
        test_data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'categorical_col': ['A', 'B', 'A', 'C', 'B']
        })
        
        print_data_summary(test_data, "Test Summary")
        
        captured = capsys.readouterr()
        assert "Test Summary" in captured.out
        assert "Shape:" in captured.out
        assert "Data Types:" in captured.out
    
    def test_detect_outliers_iqr(self):
        """Test outlier detection using IQR method."""
        # Create data with outliers
        normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        outliers = [50, -30]  # Clear outliers
        data = normal_data + outliers
        
        df = pd.DataFrame({'test_col': data})
        
        outlier_info = detect_outliers(df, method='iqr', threshold=1.5)
        
        assert 'test_col' in outlier_info
        assert outlier_info['test_col']['count'] >= 2  # Should detect at least the 2 outliers
        assert outlier_info['test_col']['percentage'] > 0
    
    def test_detect_outliers_zscore(self):
        """Test outlier detection using Z-score method."""
        # Create data with outliers
        normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        outliers = [50, -30]  # Clear outliers
        data = normal_data + outliers
        
        df = pd.DataFrame({'test_col': data})
        
        outlier_info = detect_outliers(df, method='zscore', threshold=2.0)
        
        assert 'test_col' in outlier_info
        assert outlier_info['test_col']['count'] >= 1  # Should detect at least 1 outlier
    
    def test_print_outlier_summary(self, capsys):
        """Test outlier summary printing."""
        df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'outlier_col': [1, 2, 3, 4, 100]  # Has outlier
        })
        
        print_outlier_summary(df)
        
        captured = capsys.readouterr()
        assert "OUTLIER ANALYSIS" in captured.out
        assert "outlier_col" in captured.out
    
    def test_save_and_load_results_json(self, tmp_path):
        """Test JSON saving and loading of results."""
        test_results = {
            'model_name': 'Test Model',
            'metrics': {
                'RMSE': 1000.5,
                'MAE': 800.2,
                'R2': 0.85
            },
            'timestamp': '2023-01-01T00:00:00'
        }
        
        # Test saving
        temp_file = tmp_path / "results.json"
        result = save_results_to_json(test_results, str(temp_file))
        
        assert result == True
        assert os.path.exists(temp_file)
        
        # Test loading
        loaded_results = load_results_from_json(str(temp_file))
        
        assert loaded_results is not None
        assert loaded_results['model_name'] == test_results['model_name']
        assert loaded_results['metrics']['RMSE'] == test_results['metrics']['RMSE']
    
    def test_validate_model_inputs(self):
        """Test model input validation."""
        # Valid inputs
        X_valid = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        })
        y_valid = pd.Series([10, 20, 30, 40])
        
        is_valid, issues = validate_model_inputs(X_valid, y_valid)
        
        assert is_valid == True
        assert len(issues) == 0
        
        # Shape mismatch
        X_invalid = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [5, 6, 7]
        })
        y_valid = pd.Series([10, 20, 30, 40])
        
        is_valid, issues = validate_model_inputs(X_invalid, y_valid)
        
        assert is_valid == False
        assert len(issues) > 0
        assert "Shape mismatch" in issues[0]
        
        # Missing values
        X_missing = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [5, 6, 7, 8]
        })
        y_valid = pd.Series([10, 20, 30, 40])
        
        is_valid, issues = validate_model_inputs(X_missing, y_valid)
        
        assert is_valid == False
        assert any("Missing values" in issue for issue in issues)
        
        # Constant column
        X_constant = pd.DataFrame({
            'feature1': [1, 1, 1, 1],  # Constant
            'feature2': [5, 6, 7, 8]
        })
        y_valid = pd.Series([10, 20, 30, 40])
        
        is_valid, issues = validate_model_inputs(X_constant, y_valid)
        
        assert is_valid == False
        assert any("Constant columns" in issue for issue in issues)
    
    def test_print_validation_issues(self, capsys):
        """Test validation issues printing."""
        issues = [
            "Shape mismatch: X has 3 rows, y has 4 rows",
            "Missing values found in columns: ['feature1']"
        ]
        
        print_validation_issues(issues)
        
        captured = capsys.readouterr()
        assert "VALIDATION ISSUES FOUND" in captured.out
        assert "Shape mismatch" in captured.out
        assert "Missing values" in captured.out
        
        # Test with no issues
        print_validation_issues([])
        captured = capsys.readouterr()
        assert "All validations passed!" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
