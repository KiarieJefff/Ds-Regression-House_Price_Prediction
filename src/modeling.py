"""
Modeling module for housing prices prediction.

Contains regression models, hyperparameter tuning, and model training
functionality following machine learning best practices.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Any, Optional
import joblib
from config import (
    RIDGE_PARAMS, LASSO_PARAMS, ELASTICNET_PARAMS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, DATA_PATHS
)


class ModelTrainer:
    """
    Handles model training, hyperparameter tuning, and model management
    for housing price prediction.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.model_results = {}
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = TEST_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Train a baseline linear regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained pipeline with linear regression
        """
        lin_reg = Pipeline(steps=[
            ('model', LinearRegression())
        ])
        
        lin_reg.fit(X_train, y_train)
        self.models['Linear Regression'] = lin_reg
        
        return lin_reg
    
    def train_ridge_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         alpha: float = 1.0) -> Pipeline:
        """
        Train a Ridge regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            
        Returns:
            Trained pipeline with Ridge regression
        """
        ridge = Pipeline(steps=[
            ('model', Ridge(alpha=alpha, random_state=self.random_state))
        ])
        
        ridge.fit(X_train, y_train)
        self.models['Ridge'] = ridge
        
        return ridge
    
    def train_lasso_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         alpha: float = 0.001, max_iter: int = 10000) -> Pipeline:
        """
        Train a Lasso regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            max_iter: Maximum iterations
            
        Returns:
            Trained pipeline with Lasso regression
        """
        lasso = Pipeline(steps=[
            ('model', Lasso(alpha=alpha, max_iter=max_iter, random_state=self.random_state))
        ])
        
        lasso.fit(X_train, y_train)
        self.models['Lasso'] = lasso
        
        return lasso
    
    def train_elasticnet_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              alpha: float = 0.001, l1_ratio: float = 0.5,
                              max_iter: int = 10000) -> Pipeline:
        """
        Train an Elastic Net regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            alpha: Regularization strength
            l1_ratio: Mixing parameter between L1 and L2
            max_iter: Maximum iterations
            
        Returns:
            Trained pipeline with Elastic Net regression
        """
        elastic = Pipeline(steps=[
            ('model', ElasticNet(
                alpha=alpha, 
                l1_ratio=l1_ratio, 
                max_iter=max_iter, 
                random_state=self.random_state
            ))
        ])
        
        elastic.fit(X_train, y_train)
        self.models['Elastic Net'] = elastic
        
        return elastic
    
    def tune_ridge_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  param_grid: Optional[Dict] = None) -> GridSearchCV:
        """
        Tune Ridge regression hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            
        Returns:
            Fitted GridSearchCV object
        """
        if param_grid is None:
            param_grid = RIDGE_PARAMS
        
        ridge = Ridge(random_state=self.random_state)
        
        grid_search = GridSearchCV(
            ridge, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=CV_FOLDS,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_models['Ridge'] = grid_search.best_estimator_
        
        return grid_search
    
    def tune_lasso_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  param_grid: Optional[Dict] = None) -> GridSearchCV:
        """
        Tune Lasso regression hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            
        Returns:
            Fitted GridSearchCV object
        """
        if param_grid is None:
            param_grid = LASSO_PARAMS
        
        lasso = Lasso(max_iter=10000, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            lasso, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=CV_FOLDS,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_models['Lasso'] = grid_search.best_estimator_
        
        return grid_search
    
    def tune_elasticnet_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                       param_grid: Optional[Dict] = None) -> GridSearchCV:
        """
        Tune Elastic Net hyperparameters using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            
        Returns:
            Fitted GridSearchCV object
        """
        if param_grid is None:
            param_grid = ELASTICNET_PARAMS
        
        elastic = ElasticNet(max_iter=10000, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            elastic, param_grid,
            scoring='neg_root_mean_squared_error',
            cv=CV_FOLDS,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_models['Elastic Net'] = grid_search.best_estimator_
        
        return grid_search
    
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, 
                      y_test: pd.Series, y_transform_func=None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test target
            y_transform_func: Function to transform predictions back to original scale
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Apply inverse transformation if needed
        if y_transform_func is not None:
            y_pred = y_transform_func(y_pred)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series,
                        tune_hyperparameters: bool = False) -> Dict[str, Dict]:
        """
        Train all models and evaluate their performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary of model results
        """
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        results = {}
        
        # Train baseline models
        self.train_baseline_model(X_train, y_train)
        self.train_ridge_model(X_train, y_train)
        self.train_lasso_model(X_train, y_train)
        self.train_elasticnet_model(X_train, y_train)
        
        # Evaluate baseline models
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results[name] = metrics
        
        # Hyperparameter tuning if requested
        if tune_hyperparameters:
            print("Tuning Ridge hyperparameters...")
            ridge_grid = self.tune_ridge_hyperparameters(X_train, y_train)
            ridge_metrics = self.evaluate_model(ridge_grid.best_estimator_, X_test, y_test)
            results['Ridge (Tuned)'] = ridge_metrics
            
            print("Tuning Lasso hyperparameters...")
            lasso_grid = self.tune_lasso_hyperparameters(X_train, y_train)
            lasso_metrics = self.evaluate_model(lasso_grid.best_estimator_, X_test, y_test)
            results['Lasso (Tuned)'] = lasso_metrics
            
            print("Tuning Elastic Net hyperparameters...")
            elastic_grid = self.tune_elasticnet_hyperparameters(X_train, y_train)
            elastic_metrics = self.evaluate_model(elastic_grid.best_estimator_, X_test, y_test)
            results['Elastic Net (Tuned)'] = elastic_metrics
        
        self.model_results = results
        return results
    
    def get_best_model(self, metric: str = 'RMSE') -> Tuple[str, Any]:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison ('RMSE', 'MAE', 'R2')
            
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.model_results:
            raise ValueError("No model results available. Train models first.")
        
        # For R2, higher is better; for RMSE and MAE, lower is better
        if metric == 'R2':
            best_model_name = max(self.model_results.keys(), 
                               key=lambda x: self.model_results[x][metric])
        else:
            best_model_name = min(self.model_results.keys(), 
                               key=lambda x: self.model_results[x][metric])
        
        # Get the actual model object
        if best_model_name in self.models:
            best_model = self.models[best_model_name]
        elif best_model_name in self.best_models:
            best_model = self.best_models[best_model_name.replace(' (Tuned)', '')]
        else:
            raise ValueError(f"Model {best_model_name} not found")
        
        return best_model_name, best_model
    
    def save_model(self, model: Any, filename: str, 
                   model_dir: str = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            filename: Name for the saved model file
            model_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        if model_dir is None:
            model_dir = DATA_PATHS['model_output']
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        filepath = os.path.join(model_dir, filename)
        joblib.dump(model, filepath)
        
        return filepath
    
    def load_model(self, filepath: str) -> Any:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded model object
        """
        return joblib.load(filepath)
    
    def get_model_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all model results.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.model_results:
            raise ValueError("No model results available. Train models first.")
        
        comparison_df = pd.DataFrame(self.model_results).T
        comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df


def prepare_training_data(file_path: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for training models.
    
    Args:
        file_path: Path to engineered data file
        
    Returns:
        Tuple of (features, target)
    """
    if file_path is None:
        file_path = DATA_PATHS['prepared_scaled']
    
    df = pd.read_csv(file_path)
    
    # Check if log-transformed target exists
    if 'SalePrice_log' in df.columns:
        X = df.drop(columns=['SalePrice', 'SalePrice_log'])
        y = df['SalePrice_log']
        return X, y
    else:
        X = df.drop(columns=['SalePrice'])
        y = df['SalePrice']
        return X, y
