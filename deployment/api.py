"""
FastAPI service for housing prices prediction.

Provides REST API endpoints for making predictions with trained models.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor
from feature_engineering import FeatureEngineer
from utils import validate_file_exists

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Housing Prices Prediction API",
    description="API for predicting residential housing prices using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and feature engineer
model = None
feature_engineer = None
data_processor = None

# Pydantic models for API
class HousingFeatures(BaseModel):
    """Pydantic model for housing features."""
    
    # Numeric features
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    GrLivArea: int = Field(..., ge=0, description="Above grade living area in square feet")
    TotalBsmtSF: int = Field(..., ge=0, description="Total square feet of basement area")
    GarageCars: int = Field(..., ge=0, le=4, description="Size of garage in car capacity")
    GarageArea: int = Field(..., ge=0, description="Size of garage in square feet")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Original construction year")
    
    # Categorical features
    Neighborhood: str = Field(..., description="Physical locations within Ames city limits")
    HouseStyle: str = Field(..., description="Style of dwelling")
    ExterQual: str = Field(..., description="Evaluates the quality of the material on the exterior")
    KitchenQual: str = Field(..., description="Kitchen quality")
    BsmtQual: str = Field(..., description="Evaluates the height of the basement")
    BsmtCond: str = Field(..., description="Evaluates the general condition of the basement")
    GarageType: str = Field(..., description="Garage location")
    GarageFinish: str = Field(..., description="Interior finish of the garage")
    GarageQual: str = Field(..., description="Garage quality")
    GarageCond: str = Field(..., description="Garage condition")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    
    predicted_price: float = Field(..., description="Predicted housing price in USD")
    prediction_confidence: str = Field(..., description="Confidence level of prediction")
    model_type: str = Field(..., description="Type of model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchPredictionRequest(BaseModel):
    """Pydantic model for batch prediction request."""
    
    records: List[HousingFeatures] = Field(..., description="List of housing records to predict")


class BatchPredictionResponse(BaseModel):
    """Pydantic model for batch prediction response."""
    
    predictions: List[float] = Field(..., description="List of predicted prices")
    total_records: int = Field(..., description="Total number of records processed")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Pydantic model for health check response."""
    
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    feature_engineer_loaded: bool = Field(..., description="Whether the feature engineer is loaded")
    timestamp: str = Field(..., description="Health check timestamp")


def load_models():
    """Load model and feature engineer on startup."""
    global model, feature_engineer, data_processor
    
    try:
        # Get base directory and create absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths to model files
        model_path = os.environ.get('MODEL_PATH', os.path.join(base_dir, 'models', 'ridge_model.pkl'))
        feature_engineer_path = os.environ.get('FEATURE_ENGINEER_PATH', os.path.join(base_dir, 'models', 'feature_engineer.pkl'))
        
        # Load model
        if validate_file_exists(model_path):
            model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
        else:
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load feature engineer
        if validate_file_exists(feature_engineer_path):
            feature_engineer = joblib.load(feature_engineer_path)
            logger.info(f"Feature engineer loaded from: {feature_engineer_path}")
        else:
            logger.error(f"Feature engineer file not found: {feature_engineer_path}")
            raise FileNotFoundError(f"Feature engineer file not found: {feature_engineer_path}")
        
        # Initialize data processor
        data_processor = DataProcessor()
        logger.info("Data processor initialized")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def preprocess_record(record: Dict[str, Any]) -> pd.DataFrame:
    """Preprocess a single record for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([record])
    
    # Apply data processing
    df_processed = data_processor.handle_missing_values(df)
    df_processed = data_processor.correct_data_types(df_processed)
    
    # Apply feature engineering
    df_engineered, _, _ = feature_engineer.engineer_features(
        df_processed,
        target_column='dummy_target',
        apply_log_target=False,
        scale_features=True
    )
    
    # Remove target columns if they exist
    columns_to_drop = ['dummy_target', 'SalePrice', 'SalePrice_log']
    for col in columns_to_drop:
        if col in df_engineered.columns:
            df_engineered = df_engineered.drop(columns=[col])
    
    return df_engineered


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("Starting Housing Prices Prediction API...")
    load_models()
    logger.info("API startup completed successfully")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Housing Prices Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        feature_engineer_loaded=feature_engineer is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: HousingFeatures):
    """
    Predict housing price for a single record.
    
    Args:
        features: Housing features for prediction
        
    Returns:
        Prediction response with price and metadata
    """
    start_time = datetime.now()
    
    try:
        # Convert Pydantic model to dictionary
        record = features.dict()
        
        # Preprocess the record
        X = preprocess_record(record)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Apply inverse log transform if needed
        if hasattr(feature_engineer, 'inverse_log_transform'):
            try:
                prediction = feature_engineer.inverse_log_transform([prediction])[0]
            except:
                pass  # If transformation fails, use raw prediction
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predicted_price=float(prediction),
            prediction_confidence="medium",
            model_type=type(model.named_steps['model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict housing prices for multiple records.
    
    Args:
        request: Batch prediction request with multiple records
        
    Returns:
        Batch prediction response with all predictions
    """
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for features in request.records:
            # Convert Pydantic model to dictionary
            record = features.dict()
            
            # Preprocess the record
            X = preprocess_record(record)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Apply inverse log transform if needed
            if hasattr(feature_engineer, 'inverse_log_transform'):
                try:
                    prediction = feature_engineer.inverse_log_transform([prediction])[0]
                except:
                    pass  # If transformation fails, use raw prediction
            
            predictions.append(float(prediction))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_records=len(predictions),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_type = type(model.named_steps['model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__
    
    return {
        "model_type": model_type,
        "model_loaded": True,
        "feature_engineer_loaded": feature_engineer is not None,
        "api_version": "1.0.0",
        "supported_features": list(HousingFeatures.__fields__.keys())
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
