# This script generates random samples of energy data and sends them to a backend service for prediction.

import requests
import random
import numpy as np
import time
import os

BACKEND_URL = "http://localhost:8000/predict_batch"
# BACKEND_URL = "https://56ea-140-181-90-85.ngrok-free.app/predict_batch"

def simulateEnergy():
    dE_L0 = np.random.normal(loc=977.74488, scale=18.02573)    # random.normal(loc=0.0, scale=1.0, size=None)
    dE_L1 = np.random.normal(loc=975.9298, scale=21.795434)   
    dE_Tot = dE_L0 + dE_L1
    return {"x1": dE_L0, "x2": dE_L1, "x3": dE_Tot}

def simulateEnergy2():
    # Means and sigmas for the 3 Gaussian distributions
    means_L0 = [988, 964, 990]
    sigmas_L0 = [11.04, 16.53, 8.60]
    means_L1 = [959.64, 984.83, 1008.97]
    sigmas_L1 = [12.00, 17.25, 10.70]
    
    # Probabilities for choosing each Gaussian
    probabilities_L0 = [0.625, 0.25, 0.125]
    probabilities_L1 = [0.625, 0.25, 0.125]
    
    # Choose index of the Gaussian based on the specified probabilities
    idxL0 = np.random.choice(len(means_L0), p=probabilities_L0)
    idxL1 = np.random.choice(len(means_L1), p=probabilities_L1)
    
    # Sample dE_L0, dE_L1 from the selected Gaussian
    dE_L0 = np.random.normal(loc=means_L0[idxL0], scale=sigmas_L0[idxL0])
    dE_L1 = np.random.normal(loc=means_L1[idxL1], scale=sigmas_L1[idxL1])
    
    # Calculate total energy
    dE_Tot = dE_L0 + dE_L1
    
    return {"x1": dE_L0, "x2": dE_L1, "x3": dE_Tot}


count = 1

while True:
    batch = [simulateEnergy() for _ in range(1000)]
    # batch = [simulateEnergy2() for _ in range(1000)]
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
