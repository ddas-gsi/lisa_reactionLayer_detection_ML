# Dump random Energy data to a websocket server

import asyncio
import websockets
import json
import random

async def send_data():
    uri = "ws://localhost:8000/ws/predict"
    async with websockets.connect(uri) as websocket:
        while True:
            batch = [{"x1": random.uniform(0, 10), "x2": random.uniform(0, 10)} for _ in range(100)]
            await websocket.send(json.dumps(batch))
            response = await websocket.recv()
            with open("latest_batch.json", "w") as f:
                f.write(response)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(send_data())
