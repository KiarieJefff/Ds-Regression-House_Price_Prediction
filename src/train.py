"""
Main training script for housing prices prediction.

This script orchestrates the complete machine learning pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and hyperparameter tuning
4. Model evaluation
5. Model persistence

Usage:
    python src/train.py [--config CONFIG_FILE] [--output OUTPUT_DIR]
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from modeling import ModelTrainer
from evaluation import ModelEvaluator, evaluate_model_comprehensive
from utils import (
    setup_project_structure, validate_file_exists, print_data_summary,
    save_results_to_json, log_experiment, validate_model_inputs,
    print_validation_issues
)
from config import DATA_PATHS, RANDOM_STATE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train housing prices prediction model')
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=DATA_PATHS['raw_train'],
        help='Path to training data file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models',
        help='Directory to save trained models and results'
    )
    
    parser.add_argument(
        '--tune-hyperparams', 
        action='store_true',
        help='Whether to perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip data preprocessing (use existing prepared data)'
    )
    
    parser.add_argument(
        '--save-plots', 
        action='store_true',
        help='Save evaluation plots'
    )
    
    parser.add_argument(
        '--experiment-name', 
        type=str, 
        default=f'housing_prices_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        help='Name for experiment logging'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    print("🏠 Housing Prices Prediction - Training Pipeline")
    print("=" * 60)
    
    # Setup project structure
    setup_project_structure()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer(random_state=RANDOM_STATE)
    evaluator = ModelEvaluator()
    
    experiment_config = {
        'data_path': args.data_path,
        'tune_hyperparams': args.tune_hyperparams,
        'random_state': RANDOM_STATE,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Step 1: Data Preprocessing
        print("\n📊 STEP 1: Data Preprocessing")
        print("-" * 40)
        
        if args.skip_preprocessing:
            print("Skipping preprocessing, loading existing prepared data...")
            if not validate_file_exists(DATA_PATHS['prepared_train']):
                print("❌ No prepared data found. Please run preprocessing first.")
                return
            
            df_prepared = pd.read_csv(DATA_PATHS['prepared_train'])
        else:
            print("Loading and preprocessing data...")
            df_prepared = data_processor.prepare_data(
                file_path=args.data_path,
                save_path=DATA_PATHS['prepared_train']
            )
        
        print_data_summary(df_prepared, "Prepared Data Summary")
        
        # Validate data quality
        if not data_processor.validate_data_quality(df_prepared):
            print("⚠️  Data quality issues detected. Proceeding with caution...")
        
        # Step 2: Feature Engineering
        print("\n🔧 STEP 2: Feature Engineering")
        print("-" * 40)
        
        df_engineered, y_transformed, was_log_transformed = feature_engineer.prepare_modeling_data(
            save_path=DATA_PATHS['prepared_scaled']
        )
        
        print(f"✓ Feature engineering completed")
        print(f"✓ Log transformation applied: {was_log_transformed}")
        print(f"✓ Final dataset shape: {df_engineered.shape}")
        
        # Step 3: Model Training
        print("\n🤖 STEP 3: Model Training")
        print("-" * 40)
        
        # Prepare training data
        X, y = model_trainer.prepare_training_data()
        
        # Validate model inputs
        is_valid, issues = validate_model_inputs(X, y)
        if not is_valid:
            print_validation_issues(issues)
            print("❌ Cannot proceed with training due to validation issues")
            return
        
        print(f"Training data shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        # Train models
        print("Training baseline models...")
        results = model_trainer.train_all_models(X, y, tune_hyperparameters=args.tune_hyperparams)
        
        # Display model comparison
        comparison_df = model_trainer.get_model_comparison_table()
        print("\n📈 Model Comparison:")
        print(comparison_df.to_string())
        
        # Step 4: Model Evaluation
        print("\n📊 STEP 4: Model Evaluation")
        print("-" * 40)
        
        # Get best model
        best_model_name, best_model = model_trainer.get_best_model(metric='RMSE')
        print(f"Best model: {best_model_name}")
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = model_trainer.split_data(X, y)
        
        # Comprehensive evaluation of best model
        if args.save_plots:
            plot_dir = os.path.join('reports', 'plots')
            os.makedirs(plot_dir, exist_ok=True)
        else:
            plot_dir = None
        
        # Get original scale target for evaluation
        if was_log_transformed:
            y_test_original = feature_engineer.inverse_log_transform(y_test)
        else:
            y_test_original = y_test
        
        evaluation_report = evaluate_model_comprehensive(
            best_model, X_test, y_test_original,
            model_name=best_model_name,
            feature_engineer=feature_engineer if was_log_transformed else None,
            avg_house_price=180000,
            save_dir=plot_dir
        )
        
        # Step 5: Feature Importance Analysis
        print("\n🎯 STEP 5: Feature Importance Analysis")
        print("-" * 40)
        
        if hasattr(best_model.named_steps['model'], 'coef_'):
            feature_importance_df = feature_engineer.get_feature_importance_data(
                best_model.named_steps['model'], X.columns.tolist()
            )
            
            print("Top 10 Most Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))
            
            # Plot feature importance if requested
            if args.save_plots:
                evaluator.plot_feature_importance(
                    feature_importance_df,
                    save_path=os.path.join(plot_dir, f'{best_model_name}_feature_importance.png')
                )
        else:
            print("Feature importance not available for this model type")
        
        # Step 6: Model Persistence
        print("\n💾 STEP 6: Model Persistence")
        print("-" * 40)
        
        # Save best model
        model_filename = f"{best_model_name.lower().replace(' ', '_')}_model.pkl"
        model_path = model_trainer.save_model(best_model, model_filename, args.output_dir)
        print(f"✓ Model saved: {model_path}")
        
        # Save feature engineer (for preprocessing during inference)
        import joblib
        feature_engineer_path = os.path.join(args.output_dir, 'feature_engineer.pkl')
        joblib.dump(feature_engineer, feature_engineer_path)
        print(f"✓ Feature engineer saved: {feature_engineer_path}")
        
        # Step 7: Experiment Logging
        print("\n📝 STEP 7: Experiment Logging")
        print("-" * 40)
        
        experiment_results = {
            'best_model': best_model_name,
            'model_comparison': comparison_df.to_dict(),
            'evaluation_metrics': evaluation_report['metrics'],
            'business_impact': evaluation_report['business_impact'],
            'feature_importance': feature_importance_df.head(10).to_dict('records') if 'feature_importance_df' in locals() else None,
            'data_info': {
                'training_shape': X_train.shape,
                'test_shape': X_test.shape,
                'features_count': X.shape[1],
                'log_transformed': was_log_transformed
            }
        }
        
        # Log experiment
        log_path = log_experiment(
            experiment_config, 
            experiment_results, 
            args.experiment_name,
            log_dir='experiments'
        )
        
        # Save final results
        results_path = os.path.join(args.output_dir, 'training_results.json')
        save_results_to_json(experiment_results, results_path)
        
        print(f"\n🎉 Training Pipeline Completed Successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Experiment logged: {log_path}")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📈 Best RMSE: ${evaluation_report['metrics']['RMSE']:,.2f}")
        print(f"🎯 Best R²: {evaluation_report['metrics']['R2']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error during training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
