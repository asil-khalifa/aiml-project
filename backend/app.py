from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import io
import base64

# ---------- CONFIGURATION ----------
MODEL_FILES: Dict[str, str] = {
    "grid_search": "grid_search.joblib",
    "logistic_reg": "logistic_reg.joblib",
    "knn": "knn.joblib",
    "random_forest": "random_forest.joblib",
}

DEFAULT_MODEL_KEY = "random_forest"

# ---------- LOAD MODELS AND TEST DATA ----------
_loaded_models: Dict[str, Any] = {}
_X_test = None
_y_test = None

# Load test data for metrics computation and SHAP background
try:
    data_path = Path(__file__).parent.parent / "model" / "data.csv"
    if data_path.exists():
        df = pd.read_csv(str(data_path))
        test_size = int(len(df) * 0.2)
        _X_test = df.iloc[-test_size:].drop("target", axis=1)
        _y_test = df.iloc[-test_size:]["target"]
        print(f"✅ Loaded test data: {len(_X_test)} samples")
    else:
        print("⚠️ Test data not found at", data_path)
except Exception as e:
    print(f"❌ Failed to load test data: {e}")

_backend_path = Path(__file__).parent

for key, fname in MODEL_FILES.items():
    fpath = _backend_path / fname
    try:
        if fpath.exists():
            print(f"✅ Loading model '{key}' from {fpath}")
            _loaded_models[key] = joblib.load(str(fpath))
        else:
            print(f"⚠️ Model file for '{key}' not found at {fpath}, skipping.")
    except Exception as e:
        print(f"❌ Failed to load model '{key}': {e}")

if DEFAULT_MODEL_KEY not in _loaded_models:
    print(f"⚠️ WARNING: Default model '{DEFAULT_MODEL_KEY}' not loaded. Loaded models: {_loaded_models.keys()}")

# ---------- FASTAPI APP ----------
app = FastAPI(title="Heart Disease Predictor API (with SHAP Explainability)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REQUEST / RESPONSE MODELS ----------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int
    trestbps: float
    chol: float
    fbs: int = Field(..., ge=0, le=1)
    restecg: int
    thalach: float
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float
    slope: int
    ca: int
    thal: int
    model_name: Optional[str] = None

class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    support: int

class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: Optional[str] = None
    note: Optional[str] = None
    metrics: Optional[MetricsResponse] = None
    shap_image: Optional[str] = None
    shap_summary: Optional[str] = None

# ---------- HELPER ----------
def get_model_by_key(key: Optional[str]):
    if key is None:
        key = DEFAULT_MODEL_KEY
    if key not in _loaded_models:
        raise HTTPException(status_code=400, detail=f"Model '{key}' not found.")
    return key, _loaded_models[key]

# ---------- ENDPOINTS ----------
@app.get("/models")
def list_models():
    return {"available_models": list(_loaded_models.keys()), "default_model": DEFAULT_MODEL_KEY}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, model: Optional[str] = Query(None)):
    selected_key = payload.model_name or model or DEFAULT_MODEL_KEY
    model_key, model_obj = get_model_by_key(selected_key)

    input_data = pd.DataFrame([payload.dict(exclude={"model_name"})])

    # --- Prediction ---
    try:
        if hasattr(model_obj, "predict_proba"):
            proba = float(model_obj.predict_proba(input_data)[:, 1][0])
            pred = int(proba >= 0.5)
            note = None
        else:
            pred = int(model_obj.predict(input_data)[0])
            proba = 1.0 if pred == 1 else 0.0
            note = "Model does not support predict_proba; probability inferred."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # --- SHAP Visualization ---
    shap_image_b64 = None
    shap_summary_b64 = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if _X_test is not None:
            background_data = _X_test.sample(min(100, len(_X_test)), random_state=42)
        else:
            background_data = input_data

        model_type = type(model_obj).__name__

        # --- Use correct explainer ---
        if any(x in model_type for x in ["Forest", "XGB", "GBM", "Tree"]):
            explainer = shap.TreeExplainer(model_obj, background_data, feature_perturbation="interventional")
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):  # classifiers return list
                shap_values = shap_values[1]

            shap_exp = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=input_data.columns,
            )
            shap.plots.waterfall(shap_exp, max_display=len(input_data.columns), show=False)

        else:
            explainer = shap.Explainer(model_obj, background_data)
            shap_values = explainer(input_data)
            shap.plots.waterfall(shap_values[0], max_display=len(input_data.columns), show=False)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        shap_image_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # --- Summary Plot (global view) ---
        plt.figure()
        shap.summary_plot(shap_values, input_data, show=False)
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight")
        buf2.seek(0)
        shap_summary_b64 = base64.b64encode(buf2.read()).decode("utf-8")
        plt.close()

    except Exception as e:
        print(f"⚠️ SHAP plot generation failed: {e}")

    # --- Compute Metrics ---
    metrics = None
    if _X_test is not None and _y_test is not None:
        try:
            y_pred = model_obj.predict(_X_test)
            report = classification_report(_y_test, y_pred, output_dict=True, zero_division=0)
            metrics = MetricsResponse(
                precision=float(report["1"]["precision"]),
                recall=float(report["1"]["recall"]),
                f1_score=float(report["1"]["f1-score"]),
                support=int(report["1"]["support"]),
            )
        except Exception as e:
            print(f"⚠️ Metrics computation failed: {e}")

    return PredictResponse(
        prediction=pred,
        probability=proba,
        model_version=model_key,
        note=note,
        metrics=metrics,
        shap_image=shap_image_b64,
        shap_summary=shap_summary_b64,
    )

@app.get("/")
def root():
    return {"status": "ok", "loaded_models": list(_loaded_models.keys()), "default_model": DEFAULT_MODEL_KEY}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=2006, reload=True)
