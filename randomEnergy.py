# This script generates random samples of energy data and sends them to a backend service for prediction.

import requests
import random
import time
import os

BACKEND_URL = "http://localhost:8000/predict_batch"

def generate_random_sample():
    dE_L0 = random.uniform(910.0, 1025.0)    # random.uniform(lowerLimit, upperLimit)
    dE_L1 = random.uniform(925.0, 1035.0)    # random.uniform(lowerLimit, upperLimit)
    # dE_L0 = random.uniform(500.0, 1500.0)    # random.uniform(lowerLimit, upperLimit)
    # dE_L1 = random.uniform(500.0, 1500.0)    # random.uniform(lowerLimit, upperLimit)
    dE_Tot = dE_L0 + dE_L1
    return {"x1": dE_L0, "x2": dE_L1, "x3": dE_Tot}

count = 1

while True:
    batch = [generate_random_sample() for _ in range(1000)]
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
    
    time.sleep(2)  # Wait 2 second before generating the next batch
