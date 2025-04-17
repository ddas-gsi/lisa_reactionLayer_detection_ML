# backend for batch mode processing with websocket

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
import pandas as pd
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("./models/best_hgb_model.joblib")

logging.basicConfig(level=logging.INFO)

@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted")

    try:
        while True:
            data = await websocket.receive_text()
            batch = json.loads(data)

            df = pd.DataFrame(batch)
            df["x3"] = df["x1"] + df["x2"]
            df.columns = ["dE_L0", "dE_L1", "dE_Tot"]
            predictions = model.predict(df)

            result = df.copy()
            result["prediction"] = predictions.tolist()

            await websocket.send_text(result.to_json(orient="records"))

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close()

