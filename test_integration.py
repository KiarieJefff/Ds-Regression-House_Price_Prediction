"""
Integration test for the complete housing prices prediction pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_integration():
    """Test the complete integration of all components."""
    print("🧪 Running Integration Tests...")
    print("=" * 50)
    
    try:
        # Test 1: Import all modules
        print("1. Testing imports...")
        from config import DATA_PATHS, RANDOM_STATE
        from data_processing import DataProcessor
        from feature_engineering import FeatureEngineer
        from modeling import ModelTrainer
        from evaluation import ModelEvaluator
        from utils import setup_project_structure, validate_model_inputs
        print("✅ All imports successful")
        
        # Test 2: Configuration
        print("2. Testing configuration...")
        print(f"   Base directory: {os.path.dirname(DATA_PATHS['raw_train'])}")
        print(f"   Random state: {RANDOM_STATE}")
        print("✅ Configuration working")
        
        # Test 3: Project structure
        print("3. Testing project structure...")
        setup_project_structure('.')
        print("✅ Project structure created")
        
        # Test 4: Create sample data and test pipeline
        print("4. Testing data processing pipeline...")
        
        # Create sample data
        np.random.seed(RANDOM_STATE)
        sample_data = pd.DataFrame({
            'Id': range(100),
            'SalePrice': np.random.normal(200000, 50000, 100),
            'OverallQual': np.random.randint(1, 11, 100),
            'GrLivArea': np.random.randint(800, 4000, 100),
            'TotalBsmtSF': np.random.randint(500, 2500, 100),
            'GarageCars': np.random.randint(0, 4, 100),
            'GarageArea': np.random.randint(0, 1000, 100),
            'YearBuilt': np.random.randint(1950, 2020, 100),
            'Neighborhood': np.random.choice(['CollgCr', 'NoRidge', 'Crawfor', 'Somerst'], 100),
            'HouseStyle': np.random.choice(['2Story', '1Story', 'SLvl', 'SFoyer'], 100),
            'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], 100),
            'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], 100),
            'BsmtQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'NoBasement'], 100),
            'BsmtCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'NoBasement'], 100),
            'GarageType': np.random.choice(['Attchd', 'Detchd', 'BuiltIn', 'NoGarage'], 100),
            'GarageFinish': np.random.choice(['Fin', 'RFn', 'Unf', 'NoGarage'], 100),
            'GarageQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'NoGarage'], 100),
            'GarageCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'NoGarage'], 100),
        })
        
        # Test data processing
        processor = DataProcessor()
        processed_data = processor.handle_missing_values(sample_data)
        processed_data = processor.correct_data_types(processed_data)
        print(f"   Processed data shape: {processed_data.shape}")
        print("✅ Data processing working")
        
        # Test 5: Feature engineering
        print("5. Testing feature engineering...")
        engineer = FeatureEngineer()
        
        # Add dummy target for feature engineering
        processed_data['dummy_target'] = processed_data['SalePrice']
        
        engineered_data, y_transformed, was_log_transformed = engineer.engineer_features(
            processed_data,
            target_column='dummy_target',
            apply_log_target=True,
            scale_features=True
        )
        
        print(f"   Engineered data shape: {engineered_data.shape}")
        print(f"   Log transformation applied: {was_log_transformed}")
        print("✅ Feature engineering working")
        
        # Test 6: Model training
        print("6. Testing model training...")
        trainer = ModelTrainer(random_state=RANDOM_STATE)
        
        # Prepare data for modeling
        X = engineered_data.drop(columns=['dummy_target', 'SalePrice', 'SalePrice_log'], errors='ignore')
        y = engineered_data['SalePrice_log'] if 'SalePrice_log' in engineered_data.columns else engineered_data['dummy_target']
        
        # Validate inputs
        is_valid, issues = validate_model_inputs(X, y)
        if not is_valid:
            print(f"   Validation issues: {issues}")
            return False
        
        # Train a simple model
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.3)
        model = trainer.train_ridge_model(X_train, y_train, alpha=1.0)
        
        # Test prediction
        y_pred = model.predict(X_test)
        print(f"   Model trained successfully")
        print(f"   Predictions shape: {y_pred.shape}")
        print("✅ Model training working")
        
        # Test 7: Model evaluation
        print("7. Testing model evaluation...")
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   R²: {metrics['R2']:.3f}")
        print("✅ Model evaluation working")
        
        # Test 8: Deployment imports
        print("8. Testing deployment components...")
        sys.path.append('deployment')
        from predict import HousingPricePredictor
        print("✅ Deployment imports working")
        
        print("\n🎉 All Integration Tests Passed!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ Integration Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
