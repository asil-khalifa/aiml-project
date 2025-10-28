export interface PredictRequest {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
}
export interface MetricsResponse {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface PredictResponse {
  prediction: number;
  probability: number;
  model_version: string;
  note?: string;
  shap_plot?: string;
  metrics?: MetricsResponse;
}
