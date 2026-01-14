"""
Feature engineering module for housing prices prediction.

Handles ordinal encoding, one-hot encoding, and feature scaling
to prepare data for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from config import (
    QUALITY_MAPPING, BASEMENT_EXPOSURE_MAPPING, BASEMENT_FINISH_MAPPING,
    ORDINAL_FEATURES, DATA_PATHS
)


class FeatureEngineer:
    """
    Handles feature engineering including encoding and scaling
    for the housing prices dataset.
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = None
        self.ordinal_mappings = {
            'quality': QUALITY_MAPPING,
            'basement_exposure': BASEMENT_EXPOSURE_MAPPING,
            'basement_finish': BASEMENT_FINISH_MAPPING
        }
    
    def encode_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ordinal categorical features using domain-consistent mappings.
        Unknown values are ignored (kept as-is).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded ordinal features
        """
        df_encoded = df.copy()
        
        # Apply quality mapping to quality-related features
        quality_features = ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
                          'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
                          'BsmtQual', 'BsmtCond']
        
        for col in quality_features:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(
                    self.ordinal_mappings['quality']
                ).fillna(df_encoded[col])
        
        # Apply basement exposure mapping
        if 'BsmtExposure' in df_encoded.columns:
            df_encoded['BsmtExposure'] = df_encoded['BsmtExposure'].map(
                self.ordinal_mappings['basement_exposure']
            ).fillna(df_encoded['BsmtExposure'])
        
        # Apply basement finish mapping
        basement_finish_features = ['BsmtFinType1', 'BsmtFinType2']
        for col in basement_finish_features:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(
                    self.ordinal_mappings['basement_finish']
                ).fillna(df_encoded[col])
        
        return df_encoded
    
    def encode_nominal_features(self, df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Encode nominal categorical features using one-hot encoding.
        Unknown values are ignored (treated as a separate category).
        
        Args:
            df: Input DataFrame
            drop_first: Whether to drop first category to avoid multicollinearity
            
        Returns:
            DataFrame with one-hot encoded nominal features
        """
        df_encoded = df.copy()
        
        # Identify nominal features (object type columns that aren't already encoded)
        nominal_cols = df_encoded.select_dtypes(include='object').columns.tolist()
        
        if nominal_cols:
            df_encoded = pd.get_dummies(
                df_encoded, 
                columns=nominal_cols, 
                drop_first=drop_first,
                dummy_na=False
            )
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                               target_column: str = 'SalePrice',
                               fit_scaler: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column to exclude from scaling
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_scaled = df.copy()
        
        # Identify numerical features (excluding target)
        numerical_cols = df_scaled.drop(columns=[target_column]).select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        
        if numerical_cols:
            if fit_scaler:
                self.scaler = StandardScaler()
                df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
            else:
                if self.scaler is None:
                    # Fallback: fit on this data if scaler not available
                    # This handles inference case where scaler wasn't loaded
                    self.scaler = StandardScaler()
                    df_scaled[numerical_cols] = self.scaler.fit_transform(df_scaled[numerical_cols])
                else:
                    df_scaled[numerical_cols] = self.scaler.transform(df_scaled[numerical_cols])
        
        # Store feature columns for later use
        self.feature_columns = df_scaled.columns.tolist()
        
        return df_scaled
    
    def apply_log_transform(self, y: pd.Series) -> Tuple[pd.Series, bool]:
        """
        Apply log transformation to target variable if needed.
        
        Args:
            y: Target variable
            
        Returns:
            Tuple of (transformed_target, was_transformed)
        """
        # Check if log transformation is beneficial (right-skewed data)
        skewness = y.skew()
        
        if skewness > 0.5:  # Right-skewed
            y_transformed = np.log1p(y)  # log(1 + y) to handle zeros
            return y_transformed, True
        else:
            return y, False
    
    def inverse_log_transform(self, y_transformed: pd.Series) -> pd.Series:
        """
        Inverse log transformation.
        
        Args:
            y_transformed: Log-transformed target variable
            
        Returns:
            Original scale target variable
        """
        return np.expm1(y_transformed)
    
    def engineer_features(self, df: pd.DataFrame, 
                         target_column: str = 'SalePrice',
                         apply_log_target: bool = True,
                         scale_features: bool = True,
                         fit_scaler: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series], bool]:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            apply_log_target: Whether to apply log transformation to target
            scale_features: Whether to scale numerical features
            fit_scaler: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            Tuple of (engineered_df, transformed_target, was_log_transformed)
        """
        # Separate target
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Encode ordinal features
        X_encoded = self.encode_ordinal_features(X)
        
        # Encode nominal features
        X_encoded = self.encode_nominal_features(X_encoded)
        
        # Re-attach target for scaling
        df_encoded = X_encoded.copy()
        df_encoded[target_column] = y
        
        # Scale numerical features
        if scale_features:
            df_scaled = self.scale_numerical_features(df_encoded, target_column, fit_scaler=fit_scaler)
        else:
            df_scaled = df_encoded
        
        # Apply log transformation to target if requested
        y_transformed = df_scaled[target_column].copy()
        was_log_transformed = False
        
        if apply_log_target:
            y_transformed, was_log_transformed = self.apply_log_transform(y_transformed)
            df_scaled[f'{target_column}_log'] = y_transformed
        
        return df_scaled, y_transformed if apply_log_target else None, was_log_transformed
    
    def prepare_modeling_data(self, file_path: str = None, 
                            save_path: str = None,
                            target_column: str = 'SalePrice') -> Tuple[pd.DataFrame, pd.Series, bool]:
        """
        Prepare data for modeling with complete feature engineering.
        
        Args:
            file_path: Path to prepared data file
            save_path: Path to save engineered data
            target_column: Name of target column
            
        Returns:
            Tuple of (engineered_df, transformed_target, was_log_transformed)
        """
        # Load prepared data
        if file_path is None:
            file_path = DATA_PATHS['prepared_train']
        
        df = pd.read_csv(file_path)
        
        # Apply feature engineering
        df_engineered, y_transformed, was_log_transformed = self.engineer_features(
            df, target_column=target_column
        )
        
        # Save if path provided
        if save_path is None:
            save_path = DATA_PATHS['prepared_scaled']
        
        df_engineered.to_csv(save_path, index=False)
        
        return df_engineered, y_transformed, was_log_transformed
    
    def get_feature_importance_data(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Prepare feature importance data for visualization.
        
        Args:
            model: Trained model with coef_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': model.coef_
            }).sort_values(by='coefficient', key=abs, ascending=False)
            
            return importance_df
        else:
            raise ValueError("Model does not have coef_ attribute for feature importance")


def load_engineered_data(file_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load engineered data.
    
    Args:
        file_path: Path to engineered data file
        
    Returns:
        Loaded DataFrame
    """
    if file_path is None:
        file_path = DATA_PATHS['prepared_scaled']
    
    return pd.read_csv(file_path)
