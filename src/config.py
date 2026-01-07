"""
Configuration module for housing prices prediction.

Contains constants, mappings, and configuration parameters used throughout
the data science pipeline.
"""

import os

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ordinal quality and condition mappings
QUALITY_MAPPING = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    'NoBasement': 0,
    'NoGarage': 0,
    'NoFireplace': 0,
    'NoPool': 0
}

BASEMENT_EXPOSURE_MAPPING = {
    'Gd': 4,
    'Av': 3,
    'Mn': 2,
    'No': 1,
    'NoBasement': 0
}

BASEMENT_FINISH_MAPPING = {
    'GLQ': 6,
    'ALQ': 5,
    'BLQ': 4,
    'Rec': 3,
    'LwQ': 2,
    'Unf': 1,
    'NoBasement': 0
}

# Feature groups for preprocessing
BASEMENT_FEATURES = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2'
]

GARAGE_FEATURES = [
    'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond'
]

OUTDOOR_FEATURES = ['FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

ORDINAL_FEATURES = [
    'ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual',
    'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
    'BsmtQual', 'BsmtCond'
]

# Model hyperparameters
RIDGE_PARAMS = {
    'alpha': [0.01, 0.1, 1, 10, 50, 100]
}

LASSO_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
}

ELASTICNET_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Data paths
DATA_PATHS = {
    'raw_train': os.path.join(BASE_DIR, 'data', 'raw', 'train.csv'),
    'raw_test': os.path.join(BASE_DIR, 'data', 'raw', 'test.csv'),
    'prepared_train': os.path.join(BASE_DIR, 'data', 'processed', 'train_prepared.csv'),
    'prepared_scaled': os.path.join(BASE_DIR, 'data', 'processed', 'train_prepared_scaled.csv'),
    'model_output': os.path.join(BASE_DIR, 'models')
}

# Random state for reproducibility
RANDOM_STATE = 42

# Test split ratio
TEST_SIZE = 0.2

# Cross-validation folds
CV_FOLDS = 5
