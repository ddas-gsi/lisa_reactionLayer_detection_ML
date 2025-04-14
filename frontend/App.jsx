import { useState } from "react";
import axios from "axios";

function App() {
  const [features, setFeatures] = useState("");
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const featureArray = features.split(",").map(Number);
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        features: featureArray,
      });
      setPrediction(response.data.prediction);
    } catch (err) {
      console.error("Prediction error", err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-6 max-w-md w-full">
        <h1 className="text-2xl font-bold mb-4 text-center">Prediction Dashboard</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            value={features}
            onChange={(e) => setFeatures(e.target.value)}
            placeholder="Enter comma-separated features"
            className="w-full p-2 border border-gray-300 rounded"
          />
          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
          >
            Predict
          </button>
        </form>
        {prediction !== null && (
          <div className="mt-4 text-center">
            <p className="text-lg">Prediction: <strong>{prediction}</strong></p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
