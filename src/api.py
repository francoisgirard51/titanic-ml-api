# src/api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1) Define the input schema
class TitanicPassenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# 2) Instantiate FastAPI
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Given passenger features, returns survival prediction & probability",
    version="0.1.0",
)

# 3) Load your trained pipeline on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "titanic_pipeline.joblib")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# 4) Define the predict endpoint
@app.post("/predict")
def predict(passenger: TitanicPassenger):
    # build a single-row DataFrame
    df = pd.DataFrame([passenger.model_dump()])


    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "prediction": int(pred),
        "probability": float(proba),
    }
