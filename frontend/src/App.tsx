import { useEffect, useState } from "react";
import type { PredictRequest, PredictResponse } from "./types.d.ts";
import "./index.css";
import "./App.css";

const backendPort = 2006;

const defaults: PredictRequest = {
  age: 63,
  sex: 1,
  cp: 3,
  trestbps: 145,
  chol: 233,
  fbs: 1,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 2.3,
  slope: 0,
  ca: 0,
  thal: 1,
};

export default function App() {
  const [input, setInput] = useState<PredictRequest>(defaults);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // model selection state
  const [models, setModels] = useState<string[] | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  useEffect(() => {
    // load available models from backend
    const loadModels = async () => {
      setModelsLoading(true);
      setModelsError(null);
      try {
        const res = await fetch(`http://localhost:${backendPort}/models`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        // expected shape: { available_models: [...], default_model: "..." }
        const available: string[] = data.available_models ?? [];
        const defaultModel: string | undefined = data.default_model;
        setModels(available);
        // if default model is present, pick it; otherwise first available
        setSelectedModel(
          defaultModel && available.includes(defaultModel)
            ? defaultModel
            : available.length > 0
            ? available[0]
            : null
        );
      } catch (e: any) {
        setModelsError(e.message || "Failed to fetch models");
        setModels(null);
      } finally {
        setModelsLoading(false);
      }
    };

    loadModels();
  }, []);

  const handleChange = (k: keyof PredictRequest, v: string) => {
    const isInt = [
      "sex",
      "cp",
      "fbs",
      "restecg",
      "exang",
      "slope",
      "ca",
      "thal",
    ].includes(k);
    setInput((p) => ({
      ...p,
      [k]: (isInt ? parseInt(v || "0") : parseFloat(v || "0")) as any,
    }));
  };

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    // send model_name in body (backend gives body precedence)
    const payload = {
      ...input,
      ...(selectedModel ? { model_name: selectedModel } : {}),
    };

    try {
      const res = await fetch(`http://localhost:${backendPort}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        // try to extract text error
        const txt = await res.text();
        throw new Error(txt || `HTTP ${res.status}`);
      }
      const data: PredictResponse = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex justify-center p-6 text-black">
      <div className="w-full max-w-2xl bg-white rounded-2xl shadow p-6">
        <h1 className="text-2xl font-semibold mb-4">Heart Disease Predictor</h1>

        {/* Model selector */}
        <div className="mb-4">
          <label className="text-sm text-gray-600">Choose model</label>
          <div className="mt-2">
            {modelsLoading ? (
              <div className="text-sm text-gray-500">Loading models...</div>
            ) : modelsError ? (
              <div className="text-sm text-red-600">Error: {modelsError}</div>
            ) : models && models.length > 0 ? (
              <select
                value={selectedModel ?? ""}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="p-2 border rounded w-full"
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            ) : (
              <div className="text-sm text-gray-500">No models available</div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {Object.keys(defaults).map((k) => (
            <div key={k}>
              <label className="text-sm text-gray-600">{k}</label>
              <input
                className="mt-1 p-2 border rounded w-full"
                value={(input as any)[k]}
                onChange={(e) =>
                  handleChange(k as keyof PredictRequest, e.target.value)
                }
              />
            </div>
          ))}
        </div>

        <div className="flex gap-3 mt-6">
          <button
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-60"
            disabled={loading}
            onClick={submit}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            className="px-3 py-2 border rounded text-white"
            onClick={() => {
              setInput(defaults);
              setResult(null);
              setError(null);
            }}
          >
            Reset
          </button>
        </div>

        {error && <div className="mt-4 text-red-600">Error: {error}</div>}

        {result && (
          <div className="mt-6 p-4 border rounded bg-gray-50">
            <div className="text-lg font-medium">Result</div>
            <div className="mt-2">
              Prediction: <strong>{result.prediction}</strong>
            </div>
            <div>
              Probability:{" "}
              <strong>{(result.probability * 100).toFixed(2)}%</strong>
            </div>
            <div className="text-sm text-gray-500">
              Model: {result.model_version}
            </div>
            {result.note && (
              <div className="text-sm text-gray-500 mt-1">
                Note: {result.note}
              </div>
            )}

            {/* Model Performance Metrics */}
            {result.metrics && (
              <div className="mt-4 border-t pt-3">
                <div className="text-sm font-medium mb-2">
                  Model Performance Metrics
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-600">Precision:</span>{" "}
                    <span className="font-medium">
                      {(result.metrics.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Recall:</span>{" "}
                    <span className="font-medium">
                      {(result.metrics.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">F1 Score:</span>{" "}
                    <span className="font-medium">
                      {(result.metrics.f1_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Support:</span>{" "}
                    <span className="font-medium">
                      {result.metrics.support} samples
                    </span>
                  </div>
                </div>
                <div className="text-xs text-gray-500 mt-2">
                  Metrics computed on test set for the positive class (Heart
                  Disease)
                </div>
              </div>
            )}
          </div>
        )}

        {result?.shap_plot && (
          <div className="mt-4">
            <div className="text-lg font-medium mb-2">SHAP Visualization</div>
            <img
              src={`data:image/png;base64,${result.shap_plot}`}
              alt="SHAP Visualization"
              className="w-full rounded shadow"
            />
          </div>
        )}
      </div>
    </div>
  );
}
