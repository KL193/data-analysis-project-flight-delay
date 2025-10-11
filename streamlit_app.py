# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")
st.title("✈ Flight Delay Prediction Web App")

# ----------------------
# LOAD MODELS AND ARTIFACTS
# ----------------------
@st.cache_resource
def load_models():
    # Load trained models
    models = {}
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'neural_network']
    for name in model_names:
        with open(f'models/{name}.pkl', 'rb') as f:
            models[name] = pickle.load(f)

    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load label encoders
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Load feature names
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    return models, scaler, label_encoders, feature_names


models, scaler, label_encoders, feature_names = load_models()

# Identify categorical and numerical features
categorical_features = list(label_encoders.keys())
numerical_features = [f for f in feature_names if f not in categorical_features]

# ----------------------
# USER INPUTS
# ----------------------
st.header("Enter Flight Details:")

input_dict = {}

# Numerical inputs
for feature in numerical_features:
    input_dict[feature] = st.number_input(
        f"{feature} (numerical)",
        value=0,
        key=f"num_{feature}"
    )

# Categorical inputs
for feature in categorical_features:
    le = label_encoders[feature]
    input_dict[feature] = st.selectbox(
        f"{feature} (categorical)",
        options=le.classes_,
        key=f"cat_{feature}"
    )
    # Encode immediately
    input_dict[feature] = le.transform([input_dict[feature]])[0]

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reorder columns to match training data
input_df = input_df[feature_names]

# Scale numerical features
if numerical_features:
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

# ----------------------
# PREDICTION
# ----------------------
st.header("Prediction Results:")

selected_model_name = st.selectbox(
    "Select Model",
    options=list(models.keys())
)

if st.button("Predict"):
    model = models[selected_model_name]
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0, 1]
        st.success(f"Prediction: {'Delayed' if prediction == 1 else 'On-time'}")
        st.info(f"Probability of Delay: {prediction_proba*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
