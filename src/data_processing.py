"""
Data processing module for housing prices prediction.

Handles data loading, missing value treatment, and initial data preparation
following CRISP-DM methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from config import (
    BASEMENT_FEATURES, GARAGE_FEATURES, OUTDOOR_FEATURES,
    DATA_PATHS
)


class DataProcessor:
    """
    Handles data preprocessing including missing value treatment and
    data type corrections for the housing prices dataset.
    """
    
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load raw housing data from CSV file.
        
        Args:
            file_path: Path to the CSV file. Defaults to raw training data.
            
        Returns:
            Loaded DataFrame
        """
        if file_path is None:
            file_path = DATA_PATHS['raw_train']
        
        self.data = pd.read_csv(file_path)
        return self.data
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values according to domain knowledge.
        
        Structural missing values (absence of features) are encoded explicitly,
        while true missing values are imputed appropriately.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df_processed = df.copy()
        
        # Handle basement features (structural missingness)
        basement_cols_present = [col for col in BASEMENT_FEATURES if col in df_processed.columns]
        if basement_cols_present:
            df_processed[basement_cols_present] = df_processed[basement_cols_present].fillna('NoBasement')
        
        # Handle garage features (structural missingness)
        garage_categorical = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        garage_cols_present = [col for col in garage_categorical if col in df_processed.columns]
        if garage_cols_present:
            df_processed[garage_cols_present] = df_processed[garage_cols_present].fillna('NoGarage')
        
        if 'GarageYrBlt' in df_processed.columns:
            df_processed['GarageYrBlt'] = df_processed['GarageYrBlt'].fillna(0)
        
        # Handle outdoor features (structural missingness)
        outdoor_features = ['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
        outdoor_cols_present = [col for col in outdoor_features if col in df_processed.columns]
        if outdoor_cols_present:
            if 'FireplaceQu' in df_processed.columns:
                df_processed['FireplaceQu'] = df_processed['FireplaceQu'].fillna('NoFireplace')
            if 'PoolQC' in df_processed.columns:
                df_processed['PoolQC'] = df_processed['PoolQC'].fillna('NoPool')
            if 'Fence' in df_processed.columns:
                df_processed['Fence'] = df_processed['Fence'].fillna('NoFence')
            if 'MiscFeature' in df_processed.columns:
                df_processed['MiscFeature'] = df_processed['MiscFeature'].fillna('None')
        
        # Handle alley access (structural missingness)
        if 'Alley' in df_processed.columns:
            df_processed['Alley'] = df_processed['Alley'].fillna('NoAlley')
        
        # Handle masonry veneer (semi-structural missingness)
        if 'MasVnrType' in df_processed.columns:
            df_processed['MasVnrType'] = df_processed['MasVnrType'].fillna('None')
        if 'MasVnrArea' in df_processed.columns:
            df_processed['MasVnrArea'] = df_processed['MasVnrArea'].fillna(0)
        
        # Handle electrical (true missing value)
        if 'Electrical' in df_processed.columns and df_processed['Electrical'].isnull().sum() > 0:
            df_processed['Electrical'] = df_processed['Electrical'].fillna(
                df_processed['Electrical'].mode()[0]
            )
        
        # Handle LotFrontage (neighborhood-based imputation)
        if 'LotFrontage' in df_processed.columns and 'Neighborhood' in df_processed.columns:
            df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage']\
                .transform(lambda x: x.fillna(x.median()))
        
        return df_processed
    
    def correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct data types to ensure proper variable semantics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with corrected data types
        """
        df_corrected = df.copy()
        
        # MSSubClass should be treated as categorical
        if 'MSSubClass' in df_corrected.columns:
            df_corrected['MSSubClass'] = df_corrected['MSSubClass'].astype(str)
        
        return df_corrected
    
    def get_missing_value_summary(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate a summary of missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of column names and their missing percentages
        """
        missing_summary = (df.isnull().mean() * 100).sort_values(ascending=False)
        return {col: round(perc, 2) for col, perc in missing_summary.items() if perc > 0}
    
    def prepare_data(self, file_path: str = None, save_path: str = None) -> pd.DataFrame:
        """
        Complete data preparation pipeline.
        
        Args:
            file_path: Path to raw data file
            save_path: Path to save processed data
            
        Returns:
            Fully processed DataFrame
        """
        # Load data
        df = self.load_data(file_path)
        
        # Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Correct data types
        df_final = self.correct_data_types(df_processed)
        
        # Store processed data
        self.processed_data = df_final
        
        # Save if path provided
        if save_path is None:
            save_path = DATA_PATHS['prepared_train']
        
        df_final.to_csv(save_path, index=False)
        
        return df_final
    
    def validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate that data quality meets requirements for modeling.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data passes quality checks, False otherwise
        """
        missing_summary = self.get_missing_value_summary(df)
        
        # Check for any remaining missing values that represent data quality issues
        problematic_missing = [col for col, perc in missing_summary.items() 
                             if perc > 0 and col not in ['LotFrontage']]
        
        if problematic_missing:
            print(f"Warning: Problematic missing values found in: {problematic_missing}")
            return False
        
        # Check for reasonable data types
        if 'MSSubClass' in df.columns and df['MSSubClass'].dtype != 'object':
            print("Warning: MSSubClass not properly converted to categorical")
            return False
        
        print("Data quality validation passed!")
        return True


def load_prepared_data(file_path: str = None) -> pd.DataFrame:
    """
    Convenience function to load prepared data.
    
    Args:
        file_path: Path to prepared data file
        
    Returns:
        Loaded DataFrame
    """
    if file_path is None:
        file_path = DATA_PATHS['prepared_train']
    
    return pd.read_csv(file_path)
