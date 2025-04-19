# This script generates random samples of energy data and sends them to a backend service for prediction.

import requests
import random
import numpy as np
import time
import os

BACKEND_URL = "http://localhost:8000/predict_batch"

def simulateEnergy():
    dE_L0 = np.random.normal(loc=977.74488, scale=18.02573)    # random.normal(loc=0.0, scale=1.0, size=None)
    dE_L1 = np.random.normal(loc=975.9298, scale=21.795434)   
    dE_Tot = dE_L0 + dE_L1
    return {"x1": dE_L0, "x2": dE_L1, "x3": dE_Tot}

count = 1

while True:
    batch = [simulateEnergy() for _ in range(1000)]
    try:
        response = requests.post(BACKEND_URL, json={"data": batch})
        result = response.json()

        print(f"BatchNO: {count}")
        count = count+1

        # Save batch with predictions to file (or database, Redis, etc.)
        with open("latest_batch.tmp", "w") as f:
            import json
            for i, sample in enumerate(batch):
                sample["prediction"] = result["predictions"][i]
            json.dump(batch, f)
        os.replace("latest_batch.tmp", "latest_batch.json")
    except Exception as e:
        print("Error:", e)
    
    time.sleep(1)  # Wait 2 second before generating the next batch
