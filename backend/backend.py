from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Add CORS middleware to allow requests from any origin (use specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = joblib.load("models/best_model.joblib")

# Define the input data format
class InputData(BaseModel):
    features: list[float]

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction[0]}
