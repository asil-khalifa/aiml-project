from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# ---------- CONFIG: map logical model names -> joblib filenames ----------
MODEL_FILES: Dict[str, str] = {
    "grid_search": "grid_search.joblib",
    "logistic_reg": "logistic_reg.joblib",
    "knn": "knn.joblib",
    "random_forest": "random_forest.joblib",
}

DEFAULT_MODEL_KEY = "random_forest"  # used when client doesn't specify

# ---------- load models at startup ----------
_loaded_models: Dict[str, Any] = {}
_backend_path = Path(__file__).parent

for key, fname in MODEL_FILES.items():
    fpath = _backend_path / fname
    try:
        if fpath.exists():
            print(f"Loading model '{key}' from {fpath}")
            _loaded_models[key] = joblib.load(str(fpath))
        else:
            print(f"Model file for '{key}' not found at {fpath}; skipping.")
    except Exception as e:
        print(f"Failed to load model '{key}' from {fpath}: {e}")

if DEFAULT_MODEL_KEY not in _loaded_models:
    print(f"WARNING: default model '{DEFAULT_MODEL_KEY}' not loaded. Loaded models: {_loaded_models.keys()}")

# ---------- FastAPI app ----------
app = FastAPI(title="Heart Predictor API (multiple models)")

# allow frontend dev origin; change for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request/response schemas ----------
class PredictRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0)
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

    # optional: allow specifying desired model in body (body takes precedence over query param)
    model_name: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: Optional[str] = None
    note: Optional[str] = None


# ---------- helper ----------
def get_model_by_key(key: Optional[str]):
    """Return model object for given logical key, or raise HTTPException if not available."""
    if key is None:
        key = DEFAULT_MODEL_KEY
    if key not in _loaded_models:
        raise HTTPException(status_code=400, detail=f"Requested model '{key}' not available. Use GET /models to see available models.")
    return key, _loaded_models[key]


# ---------- endpoints ----------
@app.get("/models")
def list_models():
    """List available models and which one is the default."""
    return {"available_models": list(_loaded_models.keys()), "default_model": DEFAULT_MODEL_KEY}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, model: Optional[str] = Query(None, description="Optional model key (query param). Body.field 'model_name' overrides this.")):
    """
    Predict endpoint.
    Model selection order:
      1) payload.model_name (if provided)
      2) query param ?model=...
      3) default model
    """
    # determine model key (body takes precedence)
    selected_key = payload.model_name if payload.model_name else (model if model else DEFAULT_MODEL_KEY)

    try:
        model_key, model_obj = get_model_by_key(selected_key)
    except HTTPException as e:
        raise e

    # build dataframe for inference (1-row)
    data = pd.DataFrame([payload.dict(exclude={"model_name"})])

    try:
        # prefer predict_proba when available
        if hasattr(model_obj, "predict_proba"):
            proba = float(model_obj.predict_proba(data)[:, 1][0])
            pred = int(proba >= 0.5)
            note = None
        else:
            # fallback: use predict and return deterministic 1.0/0.0 probability
            pred = int(model_obj.predict(data)[0])
            proba = 1.0 if pred == 1 else 0.0
            note = "model does not support predict_proba; probability is deterministic from predict()"
    except Exception as e:
        # catch inference errors and return 500
        print(f"Inference error using model '{model_key}': {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictResponse(prediction=pred, probability=proba, model_version=model_key, note=note)


@app.get("/")
def root():
    return {"status": "ok", "loaded_models": list(_loaded_models.keys())}
