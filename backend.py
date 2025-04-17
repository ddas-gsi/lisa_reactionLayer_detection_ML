# backend for batch mode processing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import pandas as pd
from typing import List

app = FastAPI()

# Add CORS middleware to allow requests from any origin (use specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow any origin (frontend) to talk to the backend.. Set specific domains in production
    # allow_origins=["http://localhost:3000"],  # Allow only localhost for development
    allow_credentials=True,
    allow_methods=["*"],           # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],           # Allow all headers (e.g., JSON)
)

# Load your model
model = joblib.load("./models/best_hgb_model.joblib")

# Setup basic logging config_
logging.basicConfig(level=logging.INFO)

if model:
    logging.info("Model loaded successfully.")
else:
    logging.error("Failed to load the model.")

class InputData(BaseModel):
    x1: float
    x2: float
    x3: float

class InputBatch(BaseModel):
    data: List[InputData]

@app.post("/predict_batch")
def predict_batch(batch: InputBatch):
    df = pd.DataFrame([{
        "dE_L0": item.x1,
        "dE_L1": item.x2,
        "dE_Tot": item.x3
    } for item in batch.data])

    logging.info(f"Received batch of size: {len(df)}")

    predictions = model.predict(df)
    logging.info(f"Batch predictions: {predictions.tolist()}")

    return {"predictions": predictions.tolist()}
