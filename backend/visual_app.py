import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

st.set_page_config(page_title="💓 Heart Disease Visualizer", layout="wide")

API_URL = "http://localhost:2006/predict"   # FastAPI backend URL

st.markdown("<h1 style='text-align:center;color:red;'>💓 Heart Disease Predictor Dashboard</h1>", unsafe_allow_html=True)
st.write("### Enter Patient Details Below")

# --- Sidebar input form ---
with st.sidebar.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0=Normal, 1=Fixed defect, 2=Reversible defect)", [0, 1, 2])
    model_name = st.selectbox("Select Model", ["random_forest", "logistic_reg", "knn", "grid_search"])
    submitted = st.form_submit_button("🚀 Predict")

# --- When user submits ---
if submitted:
    st.write("### Running prediction...")
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "model_name": model_name,
    }

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()

        if response.status_code != 200:
            st.error(f"❌ Error: {result.get('detail', 'Unknown error')}")
        else:
            st.success("✅ Prediction complete!")

            # --- Prediction result ---
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", "Heart Disease" if result['prediction'] == 1 else "No Disease")
            with col2:
                st.metric("Probability", f"{result['probability'] * 100:.2f}%")

            # --- Metrics ---
            if result.get("metrics"):
                st.subheader("📊 Model Performance on Test Data")
                metrics = result["metrics"]
                df_metrics = pd.DataFrame([metrics])
                st.table(df_metrics)

                # Bar chart
                fig, ax = plt.subplots()
                ax.bar(metrics.keys(), metrics.values(), color=["#FF9999", "#66B3FF", "#99FF99", "#FFD700"])
                ax.set_ylabel("Score")
                ax.set_title("Performance Metrics")
                st.pyplot(fig)

            # --- SHAP Visualization ---
            if result.get("shap_image"):
                st.subheader("🧠 SHAP Explainability (Waterfall Plot)")
                image_data = base64.b64decode(result["shap_image"])
                st.image(BytesIO(image_data), caption="SHAP Waterfall Plot", use_column_width=True)
            else:
                st.info("No SHAP visualization available for this model.")

    except Exception as e:
        st.error(f"⚠️ Request failed: {e}")

# --- Footer visuals ---
st.markdown("---")
st.markdown("Developed with ❤️ using FastAPI + Streamlit + SHAP")
