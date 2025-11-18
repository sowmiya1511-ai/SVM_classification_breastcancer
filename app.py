# app.py

import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer SVM Predictor", page_icon="🎗️", layout="centered")

st.title("🎗️ Breast Cancer Prediction (SVM Model)")
st.write("Enter patient details below and click *Predict* to know if the tumor is **Benign** or **Malignant**.")

@st.cache_resource
def load_model(path='svm_breast_cancer_model.pkl'):
    return joblib.load(path)

bundle = load_model()
pipeline = bundle['pipeline']
feature_names = bundle['feature_names']
defaults = bundle['defaults']
target_mapping = bundle['target_mapping']

st.subheader("Patient Inputs")

cols = st.columns(2)

input_values = []
for i, feat in enumerate(feature_names):
    col = cols[i % 2]
    default = float(defaults[feat])
    value = col.number_input(
        label=feat.replace("mean ", "").title(),
        min_value=0.0,
        value=round(default, 4),
        step=0.01,
        format="%.4f"
    )
    input_values.append(value)

X_input = np.array(input_values).reshape(1, -1)

st.markdown("---")

if st.button("Predict"):
    try:
        result = pipeline.predict(X_input)[0]
        pred_name = target_mapping[int(result)]

        st.subheader("Prediction Result")

        if result == 0:
            st.error("🔮 **Result: MALIGNANT**\n\nThis indicates a high-risk tumor.")
        else:
            st.success("🔮 **Result: BENIGN**\n\nThis indicates a low-risk tumor.")
    except Exception as e:
        st.error("Unexpected error during prediction.")
        st.exception(e)
