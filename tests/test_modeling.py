"""
Unit tests for modeling module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling import ModelTrainer, prepare_training_data


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'SalePrice': np.random.normal(200000, 50000, n_samples)
        })
    
    @pytest.fixture
    def trainer(self):
        """Create ModelTrainer instance."""
        return ModelTrainer(random_state=42)
    
    def test_split_data(self, trainer, sample_data):
        """Test train-test split functionality."""
        X = sample_data.drop('SalePrice', axis=1)
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        # Check shapes
        assert X_train.shape[0] == int(0.8 * len(sample_data))
        assert X_test.shape[0] == int(0.2 * len(sample_data))
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check that no data is leaked between train and test
        assert len(set(X_train.index) & set(X_test.index)) == 0
    
    def test_train_baseline_model(self, trainer, sample_data):
        """Test baseline linear regression training."""
        X = sample_data[['feature1', 'feature2']]  # Use only numeric features
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_baseline_model(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert 'Linear Regression' in trainer.models
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert all(isinstance(pred, (int, float)) for pred in y_pred)
    
    def test_train_ridge_model(self, trainer, sample_data):
        """Test Ridge regression training."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_ridge_model(X_train, y_train, alpha=1.0)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert 'Ridge' in trainer.models
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_train_lasso_model(self, trainer, sample_data):
        """Test Lasso regression training."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_lasso_model(X_train, y_train, alpha=0.001)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert 'Lasso' in trainer.models
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_train_elasticnet_model(self, trainer, sample_data):
        """Test Elastic Net regression training."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        model = trainer.train_elasticnet_model(X_train, y_train, alpha=0.001, l1_ratio=0.5)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert 'Elastic Net' in trainer.models
        
        # Test prediction
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        # Train a model
        model = trainer.train_ridge_model(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate_model(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'RMSE' in metrics
        assert 'MAE' in metrics
        assert 'R2' in metrics
        
        # Check that metrics are reasonable
        assert metrics['RMSE'] > 0
        assert metrics['MAE'] > 0
        assert 0 <= metrics['R2'] <= 1
    
    def test_train_all_models(self, trainer, sample_data):
        """Test training all models."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        results = trainer.train_all_models(X, y, tune_hyperparameters=False)
        
        assert isinstance(results, dict)
        assert len(results) >= 4  # Should have at least 4 baseline models
        
        # Check that all expected models are present
        expected_models = ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net']
        for model_name in expected_models:
            assert model_name in results
            assert 'RMSE' in results[model_name]
            assert 'MAE' in results[model_name]
            assert 'R2' in results[model_name]
    
    def test_get_best_model(self, trainer, sample_data):
        """Test getting the best model."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        # Train models
        trainer.train_all_models(X, y, tune_hyperparameters=False)
        
        # Get best model by RMSE
        best_name, best_model = trainer.get_best_model(metric='RMSE')
        
        assert best_name in trainer.models
        assert best_model is not None
        
        # Get best model by R2
        best_name_r2, best_model_r2 = trainer.get_best_model(metric='R2')
        
        assert best_name_r2 in trainer.models
        assert best_model_r2 is not None
    
    def test_get_model_comparison_table(self, trainer, sample_data):
        """Test model comparison table generation."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        # Train models
        trainer.train_all_models(X, y, tune_hyperparameters=False)
        
        # Get comparison table
        comparison_df = trainer.get_model_comparison_table()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) >= 4
        assert 'RMSE' in comparison_df.columns
        assert 'MAE' in comparison_df.columns
        assert 'R2' in comparison_df.columns
        
        # Check that table is sorted by RMSE (ascending)
        assert comparison_df['RMSE'].is_monotonic_increasing
    
    def test_save_and_load_model(self, trainer, sample_data, tmp_path):
        """Test model saving and loading."""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['SalePrice']
        
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        # Train a model
        model = trainer.train_ridge_model(X_train, y_train)
        
        # Save model
        model_path = trainer.save_model(model, 'test_model.pkl', str(tmp_path))
        
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = trainer.load_model(model_path)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test that loaded model gives same predictions
        original_pred = model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestUtilityFunctions:
    """Test utility functions in modeling module."""
    
    def test_prepare_training_data(self, tmp_path):
        """Test training data preparation utility."""
        # Create sample engineered data
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [2, 3, 4, 5],
            'SalePrice': [100000, 150000, 200000, 250000],
            'SalePrice_log': [11.51, 11.92, 12.20, 12.43]
        })
        
        # Save to temporary file
        temp_file = tmp_path / "engineered_data.csv"
        sample_data.to_csv(temp_file, index=False)
        
        # Test loading
        X, y = prepare_training_data(str(temp_file))
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert 'SalePrice' not in X.columns
        assert 'SalePrice_log' not in X.columns
        
        # Should prefer log-transformed target if available
        assert len(y) == 4


if __name__ == "__main__":
    pytest.main([__file__])
