from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import numpy as np
import logging
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Predictor", version="1.0.0")

# Load Artifact at Startup (Singleton Pattern)
ARTIFACT_PATH = "artifacts/pipeline_v1.joblib"
if not os.path.exists(ARTIFACT_PATH):
    raise FileNotFoundError(f"Artifact is not found in {ARTIFACT_PATH}. Run save_model.py")

artifact = joblib.load(ARTIFACT_PATH)
preprocessor = artifact['preprocessor']
model = artifact['model']
THRESHOLD = artifact['threshold']
FEATURE_NAMES = artifact['feature_names']

logger.info(f"Model Loaded. Threshold set to: {THRESHOLD}")

# Input Contract (Pydantic Schema)
class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float = None
    SibSp: int = 0
    Parch: int = 0 
    Fare: float = None
    Embarked: str = None

    @field_validator('Sex')
    @classmethod
    def sex_must_be_valid(cls, v):
        if v.lower() not in ['male', 'female']: 
            raise ValueError('Sex must be "male" or "female"')
        return v.lower()

    @field_validator('Embarked')
    @classmethod
    def embarked_must_be_valid(cls, v):
        if v and v.upper() not in ['S', 'C', 'Q']:
            raise ValueError('Embarked must be S, C, or Q')
        return v.upper() if v else None

@app.post("/predict")
async def predict(passenger: PassengerInput):
    """
    Predicts survival probability for a given passenger.
    Uses the tuned threshold from training.
    """
    try:
        # 1. Convert to DataFrame (Single Row)
        input_df = pd.DataFrame([passenger.model_dump()])
        
        # 2. Transform using the EXACT same pipeline as training
        # This handles imputation, scaling, and one-hot encoding automatically
        input_vector = preprocessor.transform(input_df)
        
        # 3. Get Probability
        prob_survive = model.predict_proba(input_vector)[0, 1]
        
        # 4. Apply Tuned Threshold
        prediction = int(prob_survive >= THRESHOLD)
        
        # 5. Log for Drift Monitoring (In Prod, send to Kafka/S3)
        logger.info(f"Prediction: {prediction}, Prob: {prob_survive:.4f}, Input: {input_df.to_dict()}")

        return {
            "prediction": "survived" if prediction == 1 else "died",
            "probability": round(float(prob_survive), 4),
            "threshold_used": THRESHOLD,
            "status": "success"
        }
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Prediction Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": "v1"}

@app.post("/check-drift")
async def check_drift(recent_inputs: list[PassengerInput]):
    if len(recent_inputs) < 10:
        return {"status": "insufficient_data", "message": "Need >10 samples for drift check"}
    
    # Here you would integrate Evidently AI or NannyML
    # For now, we just confirm the endpoint works
    return {
        "status": "drift_check_simulated",
        "samples_analyzed": len(recent_inputs),
        "message": "Integrate Evidently AI here to compare distributions."
    }