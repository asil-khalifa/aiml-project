from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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
        # Prefer a stratified random split so the test set contains both classes when possible
        if 'target' not in df.columns:
            print("⚠️ 'target' column not found in test data; skipping metrics setup")
        elif df['target'].nunique() < 1:
            print("⚠️ No labels found in 'target' column; skipping metrics setup")
        else:
            try:
                if df['target'].nunique() > 1:
                    train_df, test_df = train_test_split(
                        df, test_size=0.2, random_state=42, stratify=df['target']
                    )
                else:
                    # Only a single class present; fallback to simple split without stratify
                    train_df, test_df = train_test_split(
                        df, test_size=0.2, random_state=42
                    )
                _X_test = test_df.drop('target', axis=1)
                _y_test = test_df['target']
                print(f"✅ Loaded test data: {len(_X_test)} samples; label distribution: {_y_test.value_counts().to_dict()}")
            except Exception as e:
                # Fallback: use a simple tail split but warn the user
                test_size = int(len(df) * 0.2)
                _X_test = df.iloc[-test_size:].drop('target', axis=1)
                _y_test = df.iloc[-test_size:]['target']
                print(f"⚠️ Stratified split failed, falling back to tail-slice. Loaded test data: {len(_X_test)} samples; error: {e}")
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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

    # debug prints before computing metrics
    print("DEBUG: _X_test shape:", getattr(_X_test, "shape", None))
    print("DEBUG: _y_test value_counts:\n", _y_test.value_counts() if _y_test is not None else None)

    # check model type and feature info
    print("DEBUG: model type:", type(model_obj))
    if hasattr(model_obj, "feature_names_in_"):
        print("DEBUG: model.feature_names_in_:", model_obj.feature_names_in_)
    else:
        print("DEBUG: _X_test.columns:", list(_X_test.columns))

    # get predictions and show composition
    y_pred = model_obj.predict(_X_test)
    print("DEBUG: unique predictions:", np.unique(y_pred), "counts:", np.bincount(y_pred.astype(int)))

    # if available, show sample probabilities
    if hasattr(model_obj, "predict_proba"):
        try:
            proba_sample = model_obj.predict_proba(_X_test)[:5, 1]
            print("DEBUG: sample predicted probabilities (first 5):", proba_sample)
        except Exception as e:
            print("DEBUG: predict_proba failed:", e)
    
    
    # --- Compute Metrics on Test Set ---
    metrics = None
    if _X_test is not None and _y_test is not None:
        try:
            y_pred = model_obj.predict(_X_test)
            # Compute metrics for positive class (1)
            report = classification_report(
                _y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            )
            pos_report = report.get('1')
            # If there are no positive samples in the test set, pos_report may be missing or have support == 0
            if pos_report and pos_report.get('support', 0) > 0:
                metrics = MetricsResponse(
                    precision=float(pos_report['precision']),
                    recall=float(pos_report['recall']),
                    f1_score=float(pos_report['f1-score']),
                    support=int(pos_report['support']),
                )
            else:
                # No positive class examples in test set — fall back to macro average if available
                print(f"⚠️ No positive samples in test set for '{model_key}' (support=0). report keys: {list(report.keys())}")
                if 'macro avg' in report:
                    ma = report['macro avg']
                    metrics = MetricsResponse(
                        precision=float(ma['precision']),
                        recall=float(ma['recall']),
                        f1_score=float(ma['f1-score']),
                        support=int(len(_y_test)),
                    )
                else:
                    metrics = None
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
