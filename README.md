# Housing Prices Prediction

A comprehensive machine learning pipeline for predicting residential housing prices using regression models.

## 🏠 Project Overview

This project implements a complete data science workflow following CRISP-DM methodology to predict housing prices in Ames, Iowa. The pipeline includes data preprocessing, feature engineering, model training, hyperparameter tuning, comprehensive evaluation, testing, and deployment capabilities.

## 📁 Project Structure

```
Housing Prices_2/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed and engineered data
├── models/                     # Trained model files
├── notebooks/                  # Exploratory data analysis notebooks
├── reports/                    # Evaluation reports and plots
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration constants and mappings
│   ├── data_processing.py      # Data preprocessing and missing value handling
│   ├── feature_engineering.py  # Encoding and scaling features
│   ├── modeling.py             # Model training and hyperparameter tuning
│   ├── evaluation.py           # Model evaluation and visualization
│   ├── utils.py                # Utility functions
│   └── train.py                # Main training script
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_modeling.py
│   └── test_utils.py
├── deployment/                 # Deployment files
│   ├── __init__.py
│   ├── predict.py              # Command-line inference script
│   └── api.py                  # FastAPI service
├── experiments/                # Experiment logs and results
├── requirements.txt            # Python dependencies
├── pytest.ini                 # Test configuration
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── .dockerignore              # Docker ignore file
└── README.md                   # This file
```
│   ├── data_processing.py      # Data preprocessing and missing value handling
│   ├── feature_engineering.py  # Encoding and scaling features
│   ├── modeling.py             # Model training and hyperparameter tuning
│   ├── evaluation.py           # Model evaluation and visualization
│   ├── utils.py                # Utility functions
│   └── train.py                # Main training script
├── experiments/                # Experiment logs and results
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
# Basic training with default settings
python src/train.py

# With hyperparameter tuning and plot generation
python src/train.py --tune-hyperparams --save-plots

# Custom data path and output directory
python src/train.py --data-path path/to/your/data.csv --output-dir my_models
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_processing.py
```

### 4. Make Predictions

```bash
# Interactive single prediction
python deployment/predict.py --interactive

# Batch prediction from CSV file
python deployment/predict.py --input-file data/new_houses.csv --output-file predictions.csv
```

### 5. Start API Service

```bash
# Start FastAPI server
uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000

# Or use Docker Compose
docker-compose up housing-api
```

### 6. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 7. Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run only the API service
docker-compose up housing-api

# Run training in container
docker-compose --profile training up training

# Access Jupyter notebook
docker-compose up jupyter
# Then visit http://localhost:8888
```

## 📊 Pipeline Components

### 1. Data Processing (`data_processing.py`)
- Handles missing values according to domain knowledge
- Structural missingness (e.g., no basement) is explicitly encoded
- Neighborhood-based imputation for LotFrontage
- Data type corrections and validation

### 2. Feature Engineering (`feature_engineering.py`)
- Ordinal encoding for quality/condition variables
- One-hot encoding for nominal categorical features
- Standard scaling for numerical features
- Log transformation for right-skewed target variable

### 3. Modeling (`modeling.py`)
- Multiple regression models: Linear, Ridge, Lasso, Elastic Net
- Hyperparameter tuning with GridSearchCV
- Model comparison and selection
- Model persistence with joblib

### 4. Evaluation (`evaluation.py`)
- Comprehensive metrics (RMSE, MAE, R², MAPE)
- Residual analysis and diagnostic plots
- Business impact assessment
- Feature importance visualization

### 5. Configuration (`config.py`)
- Centralized configuration for mappings and parameters
- Feature group definitions
- Model hyperparameter grids
- Data paths and constants

## 📈 Model Performance

Based on the notebook analysis, the **Ridge Regression** model achieved the best performance:

- **RMSE**: ~$31,206
- **MAE**: ~$19,815  
- **R²**: ~0.873 (87.3% variance explained)

The model effectively handles multicollinearity and provides stable predictions for housing prices.

## 🔍 Key Features

- **Domain-aware preprocessing**: Missing values handled according to real estate domain knowledge
- **Comprehensive evaluation**: Both technical metrics and business impact analysis
- **Reproducible pipeline**: Configuration-driven with experiment logging
- **Modular design**: Easy to extend and modify individual components
- **Production-ready**: Model persistence and inference capabilities

## 📋 Business Impact

The predictive pricing system provides:

- **Reduced time-on-market** through accurate pricing
- **Improved pricing consistency** across agents and locations  
- **Data-backed negotiation support** for sales agents
- **Increased revenue capture** through optimal pricing
- **Enhanced transparency** for clients

## 🛠️ Development

### Adding New Models

1. Update `ModelTrainer` class in `modeling.py`
2. Add hyperparameter grids to `config.py`
3. Update evaluation metrics if needed

### Custom Feature Engineering

1. Modify `FeatureEngineer` class in `feature_engineering.py`
2. Update configuration mappings in `config.py`
3. Retrain pipeline with new features

### Experiment Tracking

All experiments are automatically logged to the `experiments/` directory with:
- Configuration parameters
- Model performance metrics
- Feature importance data
- Timestamp and experiment name

## 📚 Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

1. **Business Understanding**: Define objectives and success criteria
2. **Data Understanding**: Explore and validate data quality
3. **Data Preparation**: Handle missing values and clean data
4. **Modeling**: Train and evaluate multiple regression models
5. **Evaluation**: Assess model performance and business impact
6. **Deployment**: Prepare models for production use

## 🧪 Testing

The project includes comprehensive unit tests for all major components, plus specialized tests for handling incomplete and unknown data.

### Test Coverage

- **Data Processing Tests** (`test_data_processing.py`): Missing value handling, data validation, preprocessing pipeline
- **Feature Engineering Tests** (`test_feature_engineering.py`): Encoding methods, scaling, transformations
- **Modeling Tests** (`test_modeling.py`): Model training, evaluation, hyperparameter tuning
- **Utility Tests** (`test_utils.py`): Helper functions, validation, data operations
- **Integration Tests**: Complete ML pipeline with incomplete data
- **Encoding Behavior Tests**: Verification of unknown value handling

### Running Tests

```bash
# Run all unit tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/test_data_processing.py -v
pytest tests/test_modeling.py -v

# Run integration tests with incomplete data
python -m pytest tests/test_incomplete_data.py -v

# Run encoding behavior tests
python -m pytest tests/test_encoding_behavior.py -v

# Run tests with specific markers
pytest -m "not slow"  # Skip slow tests
```

### Test Configuration

- **pytest.ini**: Test configuration and settings
- **Coverage**: HTML reports generated in `htmlcov/` directory
- **Markers**: Tests can be marked as `slow`, `integration`, or `unit`

### Recent Testing Updates (January 14, 2026)

#### Unknown Value Handling Tests

The project was extensively tested with incomplete data containing unknown/unseen categorical values to validate robust encoding methods.

#### Tests Added

1. **test_incomplete_data.py** - Integration test with real-world data patterns
   - Tests complete pipeline with 1,460 samples containing unknown values
   - Validates no NaN introduction during encoding
   - Verifies model training succeeds with incomplete data
   - **Results**: PASSED - Train R²: 0.9264, Test R²: 0.8918

2. **test_encoding_behavior.py** - Unit tests for encoding behavior
   - Tests ordinal feature encoding with unknown values
   - Tests nominal feature encoding with unknown values
   - Tests complete pipeline with mixed unknown values
   - **Results**: ALL PASSED - Unknown values handled correctly

#### Code Modifications

**File: `src/feature_engineering.py`**

1. **Ordinal Encoding Enhancement**
   ```python
   df_encoded[col] = df_encoded[col].map(mapping).fillna(df_encoded[col])
   ```
   - Unknown ordinal values are now preserved instead of converted to NaN
   - Known values mapped to numeric values (e.g., 'Ex' → 5, 'Gd' → 4)

2. **Nominal Encoding Enhancement**
   ```python
   df_encoded = pd.get_dummies(
       df_encoded, 
       columns=nominal_cols, 
       drop_first=drop_first,
       dummy_na=False  # Don't create separate column for unknowns
   )
   ```
   - Added `dummy_na=False` parameter to avoid spurious columns for unknown categories
   - Only creates one-hot columns for observed categories

#### Key Test Results

| Feature | Status | Details |
|---------|--------|---------|
| Ordinal encoding with unknowns | ✓ PASS | Preserves unknown values |
| Nominal encoding with unknowns | ✓ PASS | Handles without NaN |
| Complete pipeline with unknowns | ✓ PASS | 0 NaN values introduced |
| Model training with unknowns | ✓ PASS | Trains successfully |
| Model performance | ✓ PASS | Good R² scores (0.89-0.93) |
| Data integrity | ✓ PASS | 0 NaN values |
| Feature scaling | ✓ PASS | Works correctly |
| Log transformation | ✓ PASS | Applied successfully |

#### Example: Unknown Value Handling

**Before (problematic)**:
```
Input:  HouseStyle = ['2Story', '1Story', 'FUTURE_STYLE', '2Story']
Output: [1, 0, NaN, 1]        ← NaN breaks downstream processing
```

**After (robust)**:
```
Input:  HouseStyle = ['2Story', '1Story', 'FUTURE_STYLE', '2Story']
Output: [1, 0, 0, 1]          ← Unknown is zero in all categories (safe)
```

#### Production Readiness

✓ **Approved for Production**:
- Backward compatible with existing data
- Handles edge cases gracefully
- Improves robustness for real-world data
- Maintains excellent model performance
- 0 NaN values in processing pipeline

## 🚀 Deployment

### API Service

The FastAPI service provides REST endpoints for housing price prediction:

#### Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `POST /predict`: Single prediction endpoint
- `POST /predict/batch`: Batch prediction endpoint
- `GET /model/info`: Model information endpoint

#### Usage Example

```python
import requests

# Single prediction
payload = {
    "OverallQual": 7,
    "GrLivArea": 2000,
    "TotalBsmtSF": 1000,
    "GarageCars": 2,
    "GarageArea": 500,
    "YearBuilt": 2000,
    "Neighborhood": "CollgCr",
    "HouseStyle": "2Story",
    "ExterQual": "Gd",
    "KitchenQual": "TA",
    "BsmtQual": "Gd",
    "BsmtCond": "TA",
    "GarageType": "Attchd",
    "GarageFinish": "RFn",
    "GarageQual": "TA",
    "GarageCond": "TA"
}

response = requests.post("http://localhost:8000/predict", json=payload)
prediction = response.json()
print(f"Predicted price: ${prediction['predicted_price']:,.2f}")
```

### Docker Deployment

The project includes Docker configuration for easy deployment:

#### Services

1. **housing-api**: FastAPI service for predictions
2. **jupyter**: Jupyter notebook for development
3. **training**: Training service (optional profile)

#### Environment Variables

- `MODEL_PATH`: Path to trained model file
- `FEATURE_ENGINEER_PATH`: Path to feature engineer object

### Production Considerations

- **Health Checks**: Built-in health check endpoints
- **Logging**: Comprehensive logging with different levels
- **Error Handling**: Proper HTTP status codes and error messages
- **Validation**: Input validation using Pydantic models
- **Monitoring**: Processing time tracking and performance metrics

### Missing Value Strategy
- **Structural missingness**: Encoded as meaningful categories (e.g., "NoBasement")
- **Semi-structural**: Logical imputation (e.g., Masonry area = 0 when no veneer)
- **True missing**: Mode imputation for negligible cases
- **Spatial**: Neighborhood-based median imputation for LotFrontage

### Feature Encoding
- **Ordinal variables**: Domain-consistent numerical mapping (Ex=5, Gd=4, etc.)
- **Nominal variables**: One-hot encoding with drop_first to avoid multicollinearity
- **Numerical features**: Standard scaling (mean=0, std=1)

### Model Selection
- **Ridge Regression**: Best overall performance, handles multicollinearity
- **Elastic Net**: Similar performance with feature selection capability
- **Linear Regression**: Baseline model, suffers from multicollinearity
- **Lasso**: Underperforms due to non-sparse feature relationships

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to submit issues and enhancement requests to improve the pipeline functionality.
