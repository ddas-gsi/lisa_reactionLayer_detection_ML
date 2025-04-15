from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import pandas as pd

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

@app.post("/predict")
def predict(data: InputData):
    logging.info(f"Received input: dE_L0={data.x1}, dE_L1={data.x2}, dE_Tot={data.x3}")
    
    # Use correct feature names here
    input_df = pd.DataFrame([{
        "dE_L0": data.x1,
        "dE_L1": data.x2,
        "dE_Tot": data.x3
    }])

    prediction = model.predict(input_df)
    logging.info(f"Prediction result: {prediction[0]}")
    return {"prediction": prediction[0]}