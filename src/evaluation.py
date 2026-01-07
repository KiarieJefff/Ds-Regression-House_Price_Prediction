"""
Evaluation module for housing prices prediction.

Provides comprehensive model evaluation including metrics, visualization,
and business impact assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics, visualizations,
    and business impact analysis.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.feature_importance_data = None
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Median_AE': np.median(np.abs(y_true - y_pred))
        }
        
        # Calculate percentage errors for business context
        metrics['Mean_Percentage_Error'] = np.mean((y_pred - y_true) / y_true) * 100
        
        return metrics
    
    def evaluate_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Analyze model residuals for diagnostic purposes.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of residual analysis results
        """
        residuals = y_true - y_pred
        
        residual_stats = {
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals),
            'Skewness_Residuals': pd.Series(residuals).skew(),
            'Kurtosis_Residuals': pd.Series(residuals).kurtosis(),
            'Residuals': residuals
        }
        
        return residual_stats
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  title: str = "Predictions vs Actual Values",
                                  save_path: Optional[str] = None) -> None:
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Create residual analysis plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Residual Analysis', fontsize=16)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[1, 1].scatter(y_true, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Actual')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance from linear model coefficients.
        
        Args:
            feature_importance_df: DataFrame with features and coefficients
            top_n: Number of top features to display
            save_path: Path to save the plot
        """
        self.feature_importance_data = feature_importance_df
        
        # Get top features by absolute coefficient
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Create color map for positive/negative coefficients
        colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
        
        sns.barplot(x='coefficient', y='feature', data=top_features, palette=colors)
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at x=0
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def business_impact_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                avg_house_price: float) -> Dict[str, Any]:
        """
        Analyze business impact of model predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            avg_house_price: Average house price in the market
            
        Returns:
            Dictionary of business impact metrics
        """
        # Calculate prediction errors in dollar terms
        absolute_errors = np.abs(y_true - y_pred)
        percentage_errors = (absolute_errors / y_true) * 100
        
        business_metrics = {
            'Avg_Absolute_Error_Dollars': np.mean(absolute_errors),
            'Median_Absolute_Error_Dollars': np.median(absolute_errors),
            'Avg_Percentage_Error': np.mean(percentage_errors),
            'Median_Percentage_Error': np.median(percentage_errors),
            'Pct_Predictions_Within_5pct': np.mean(percentage_errors <= 5) * 100,
            'Pct_Predictions_Within_10pct': np.mean(percentage_errors <= 10) * 100,
            'Pct_Predictions_Within_15pct': np.mean(percentage_errors <= 15) * 100,
        }
        
        # Revenue impact estimation
        # Assuming underpricing leads to revenue loss
        underpricing_loss = np.maximum(0, y_true - y_pred)
        overpricing_risk = np.maximum(0, y_pred - y_true)
        
        business_metrics.update({
            'Est_Annual_Revenue_Loss': np.sum(underpricing_loss),
            'Est_Overpricing_Risk': np.sum(overpricing_risk),
            'Pct_Underpriced': np.mean(y_pred < y_true) * 100,
            'Pct_Overpriced': np.mean(y_pred > y_true) * 100
        })
        
        return business_metrics
    
    def create_evaluation_report(self, model_name: str, y_true: np.ndarray, 
                               y_pred: np.ndarray, avg_house_price: float = 180000,
                               save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report.
        
        Args:
            model_name: Name of the model being evaluated
            y_true: True target values
            y_pred: Predicted target values
            avg_house_price: Average house price for business context
            save_dir: Directory to save evaluation plots
            
        Returns:
            Dictionary containing all evaluation results
        """
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Residual analysis
        residual_analysis = self.evaluate_residuals(y_true, y_pred)
        
        # Business impact
        business_impact = self.business_impact_analysis(y_true, y_pred, avg_house_price)
        
        # Compile results
        evaluation_report = {
            'model_name': model_name,
            'metrics': metrics,
            'residual_analysis': residual_analysis,
            'business_impact': business_impact
        }
        
        # Store results
        self.evaluation_results[model_name] = evaluation_report
        
        # Generate visualizations
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_predictions_vs_actual(
                y_true, y_pred, 
                title=f'{model_name} - Predictions vs Actual',
                save_path=os.path.join(save_dir, f'{model_name}_predictions_vs_actual.png')
            )
            
            self.plot_residuals(
                y_true, y_pred,
                save_path=os.path.join(save_dir, f'{model_name}_residuals.png')
            )
        
        return evaluation_report
    
    def compare_models(self, results_dict: Dict[str, Dict[str, float]],
                      metric: str = 'RMSE') -> pd.DataFrame:
        """
        Compare multiple models based on evaluation metrics.
        
        Args:
            results_dict: Dictionary of model results
            metric: Primary metric for ranking
            
        Returns:
            DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results_dict).T
        
        # Sort by specified metric
        if metric in ['RMSE', 'MAE', 'MAPE']:
            comparison_df = comparison_df.sort_values(metric)
        elif metric == 'R2':
            comparison_df = comparison_df.sort_values(metric, ascending=False)
        
        return comparison_df
    
    def print_evaluation_summary(self, model_name: str) -> None:
        """
        Print a formatted summary of evaluation results.
        
        Args:
            model_name: Name of the model to summarize
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for model: {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {model_name}")
        print(f"{'='*60}")
        
        print("\n📊 PERFORMANCE METRICS:")
        print(f"  RMSE: ${results['metrics']['RMSE']:,.2f}")
        print(f"  MAE:  ${results['metrics']['MAE']:,.2f}")
        print(f"  R²:   {results['metrics']['R2']:.3f}")
        print(f"  MAPE: {results['metrics']['MAPE']:.2f}%")
        
        print("\n💼 BUSINESS IMPACT:")
        print(f"  Avg Absolute Error: ${results['business_impact']['Avg_Absolute_Error_Dollars']:,.2f}")
        print(f"  Predictions within 5%: {results['business_impact']['Pct_Predictions_Within_5pct']:.1f}%")
        print(f"  Predictions within 10%: {results['business_impact']['Pct_Predictions_Within_10pct']:.1f}%")
        print(f"  Underpriced predictions: {results['business_impact']['Pct_Underpriced']:.1f}%")
        
        print("\n📈 RESIDUAL ANALYSIS:")
        print(f"  Mean residual: ${results['residual_analysis']['Mean_Residual']:,.2f}")
        print(f"  Residual std: ${results['residual_analysis']['Std_Residual']:,.2f}")
        print(f"{'='*60}\n")


def evaluate_model_comprehensive(model, X_test: pd.DataFrame, y_test: pd.Series,
                               model_name: str, feature_engineer=None,
                               avg_house_price: float = 180000,
                               save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for comprehensive model evaluation.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        feature_engineer: FeatureEngineer object for inverse transformations
        avg_house_price: Average house price for business context
        save_dir: Directory to save evaluation outputs
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = ModelEvaluator()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Apply inverse transformation if needed
    if feature_engineer and hasattr(feature_engineer, 'inverse_log_transform'):
        y_pred = feature_engineer.inverse_log_transform(y_pred)
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(
        model_name, y_test, y_pred, avg_house_price, save_dir
    )
    
    # Print summary
    evaluator.print_evaluation_summary(model_name)
    
    return report
