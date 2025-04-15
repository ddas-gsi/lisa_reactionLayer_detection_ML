import requests
resp = requests.post("http://localhost:8000/predict", json={"x1": 1050, "x2": 1000, "x3": 2150})
print(resp.status_code)
print(resp.text)
