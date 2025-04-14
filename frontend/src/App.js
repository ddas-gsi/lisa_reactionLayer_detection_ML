import { useState } from "react";
import "./App.css";

function App() {
  const [input, setInput] = useState("");
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const features = input
      .split(",")
      .map((val) => parseFloat(val.trim()))
      .filter((val) => !isNaN(val));

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ features }),
    });

    const data = await response.json();
    setPrediction(data.prediction);
  };

  return (
    <div className="App">
      <div className="container">
        <h1>🧠 ML Model Predictor</h1>
        <form onSubmit={handleSubmit}>
          <label htmlFor="inputField">
            Enter comma-separated feature values:
          </label>
          <input
            id="inputField"
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g., 5.1, 3.2, 1.4, 0.2"
          />
          <button type="submit">Predict</button>
        </form>
        {prediction !== null && (
          <div className="result">
            <h2>🔮 Prediction Result:</h2>
            <p>{prediction}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
