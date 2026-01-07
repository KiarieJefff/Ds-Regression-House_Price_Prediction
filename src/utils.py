"""
Utility functions for housing prices prediction.

Contains helper functions for common operations, data validation,
and project management tasks.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')


def setup_project_structure(base_path: str = ".") -> None:
    """
    Create necessary directories for the project.
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports',
        'notebooks',
        'src'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Created directory: {full_path}")


def validate_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file exists, False otherwise
    """
    exists = os.path.exists(file_path)
    if not exists:
        print(f"❌ File not found: {file_path}")
    return exists


def load_data_safely(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load data safely with error handling.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if not validate_file_exists(file_path):
            return None
        
        df = pd.read_csv(file_path)
        print(f"✓ Successfully loaded data: {file_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data from {file_path}: {str(e)}")
        return None


def save_data_safely(df: pd.DataFrame, file_path: str, 
                    index: bool = False) -> bool:
    """
    Save data safely with error handling.
    
    Args:
        df: DataFrame to save
        file_path: Path to save the file
        index: Whether to save the index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=index)
        print(f"✓ Successfully saved data to: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving data to {file_path}: {str(e)}")
        return False


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a comprehensive summary of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'shape': df.shape,
        'columns': len(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Add basic statistics for numeric columns
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    return summary


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary") -> None:
    """
    Print a formatted data summary.
    
    Args:
        df: Input DataFrame
        title: Title for the summary
    """
    summary = get_data_summary(df)
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Shape: {summary['shape'][0]:,} rows × {summary['shape'][1]} columns")
    print(f"Memory usage: {summary['memory_usage']:.2f} MB")
    
    print(f"\n📊 Data Types:")
    for dtype, count in summary['dtypes'].items():
        print(f"  {dtype}: {count} columns")
    
    print(f"\n📈 Numeric Columns: {len(summary['numeric_columns'])}")
    print(f"📝 Categorical Columns: {len(summary['categorical_columns'])}")
    
    # Missing values
    missing_cols = [col for col, missing in summary['missing_values'].items() if missing > 0]
    if missing_cols:
        print(f"\n⚠️  Missing Values Found in {len(missing_cols)} columns:")
        for col in missing_cols[:5]:  # Show first 5
            missing = summary['missing_values'][col]
            print(f"  {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
        if len(missing_cols) > 5:
            print(f"  ... and {len(missing_cols)-5} more columns")
    else:
        print(f"\n✅ No missing values found")
    
    print(f"{'='*60}\n")


def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Dict]:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to check (if None, check all numeric columns)
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        Dictionary with outlier information for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > threshold]
        
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'values': outliers.tolist()[:10]  # First 10 outliers
        }
    
    return outlier_info


def print_outlier_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """
    Print a summary of outliers in the dataset.
    
    Args:
        df: Input DataFrame
        columns: Columns to check
    """
    outlier_info = detect_outliers(df, columns)
    
    print(f"\n{'='*60}")
    print("OUTLIER ANALYSIS")
    print(f"{'='*60}")
    
    total_outliers = 0
    for col, info in outlier_info.items():
        if info['count'] > 0:
            print(f"📊 {col}:")
            print(f"  Outliers: {info['count']} ({info['percentage']:.1f}%)")
            total_outliers += info['count']
    
    if total_outliers == 0:
        print("✅ No outliers detected")
    else:
        print(f"\n📈 Total outliers detected: {total_outliers:,}")
    
    print(f"{'='*60}\n")


def save_results_to_json(results: Dict[str, Any], file_path: str) -> bool:
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary to save
        file_path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Results saved to: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")
        return False


def load_results_from_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load results from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Results dictionary if successful, None otherwise
    """
    try:
        if not validate_file_exists(file_path):
            return None
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        print(f"✓ Results loaded from: {file_path}")
        return results
        
    except Exception as e:
        print(f"❌ Error loading results: {str(e)}")
        return None


def create_feature_importance_report(feature_importance_df: pd.DataFrame,
                                   top_n: int = 20) -> Dict[str, Any]:
    """
    Create a detailed feature importance report.
    
    Args:
        feature_importance_df: DataFrame with feature importances
        top_n: Number of top features to include in detailed report
        
    Returns:
        Dictionary containing feature importance analysis
    """
    top_features = feature_importance_df.head(top_n)
    
    report = {
        'total_features': len(feature_importance_df),
        'top_features': top_features.to_dict('records'),
        'positive_features': top_features[top_features['coefficient'] > 0].to_dict('records'),
        'negative_features': top_features[top_features['coefficient'] < 0].to_dict('records'),
        'summary': {
            'top_positive_feature': top_features[top_features['coefficient'] > 0].iloc[0]['feature'] if len(top_features[top_features['coefficient'] > 0]) > 0 else None,
            'top_negative_feature': top_features[top_features['coefficient'] < 0].iloc[0]['feature'] if len(top_features[top_features['coefficient'] < 0]) > 0 else None,
            'avg_coefficient': feature_importance_df['coefficient'].mean(),
            'std_coefficient': feature_importance_df['coefficient'].std()
        }
    }
    
    return report


def log_experiment(config: Dict[str, Any], results: Dict[str, Any],
                  experiment_name: str, log_dir: str = "experiments") -> str:
    """
    Log experiment configuration and results.
    
    Args:
        config: Experiment configuration
        results: Experiment results
        experiment_name: Name of the experiment
        log_dir: Directory to save experiment logs
        
    Returns:
        Path to the experiment log file
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    log_path = os.path.join(log_dir, filename)
    
    experiment_log = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'results': results
    }
    
    save_results_to_json(experiment_log, log_path)
    return log_path


def validate_model_inputs(X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
    """
    Validate inputs for model training.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check shapes
    if len(X) != len(y):
        issues.append(f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows")
    
    # Check for missing values
    if X.isnull().sum().sum() > 0:
        missing_cols = X.columns[X.isnull().sum() > 0].tolist()
        issues.append(f"Missing values found in columns: {missing_cols}")
    
    if y.isnull().sum() > 0:
        issues.append(f"Missing values found in target: {y.isnull().sum()} values")
    
    # Check for constant columns
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        issues.append(f"Constant columns found: {constant_cols}")
    
    return len(issues) == 0, issues


def print_validation_issues(issues: List[str]) -> None:
    """
    Print validation issues in a formatted way.
    
    Args:
        issues: List of validation issues
    """
    if issues:
        print(f"\n{'='*60}")
        print("VALIDATION ISSUES FOUND")
        print(f"{'='*60}")
        for i, issue in enumerate(issues, 1):
            print(f"❌ {i}. {issue}")
        print(f"{'='*60}\n")
    else:
        print("✅ All validations passed!")
