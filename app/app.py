import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Heart Disease Prediction (XAI)", layout="wide")

model = joblib.load(os.path.join(BASE_DIR, '..', 'model', 'random_forest_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, '..', 'data', 'processed', 'scaler.pkl'))
feature_names = joblib.load(os.path.join(BASE_DIR, '..', 'data', 'processed', 'feature_names.pkl'))
label_encoders = joblib.load(os.path.join(BASE_DIR, '..', 'data', 'processed', 'label_encoders.pkl'))
df_original = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'HeartDiseaseTrain-Test.csv')).drop_duplicates().reset_index(drop=True)

st.title("🫀 Explainable AI: Heart Disease Prediction")
st.markdown("Enter patient details below to get a prediction **with an explanation** of why the model made that decision.")

st.sidebar.header("Patient Information")
user_input = {}

for col in feature_names:
    if col in label_encoders:  # categorical column — use the REAL trained encoder's classes
        options = list(label_encoders[col].classes_)
        user_input[col] = st.sidebar.selectbox(col.replace('_', ' ').title(), options)
    else:  # numeric column
        min_val = float(df_original[col].min())
        max_val = float(df_original[col].max())
        mean_val = float(df_original[col].mean())
        user_input[col] = st.sidebar.slider(col.replace('_', ' ').title(), min_val, max_val, mean_val)

predict_button = st.sidebar.button("🔍 Predict")

if predict_button:
    input_df = pd.DataFrame([user_input])[feature_names]

    # ----------------------------
    # Input validation: flag physiologically implausible combinations
    # ----------------------------
    warnings_list = []
    age_val = user_input['age']
    max_hr_val = user_input['Max_heart_rate']
    expected_max_hr = 220 - age_val

    if max_hr_val > expected_max_hr + 15:
        warnings_list.append(
            f"⚠️ Max Heart Rate ({max_hr_val:.0f}) is unusually high for age {age_val:.0f}. "
            f"Expected maximum is roughly {expected_max_hr:.0f} bpm (220 − age)."
        )
    if user_input['cholestoral'] > 400:
        warnings_list.append(f"⚠️ Cholesterol level ({user_input['cholestoral']:.0f}) is extremely high (normal range is typically 125–200).")
    if user_input['resting_blood_pressure'] > 180:
        warnings_list.append(f"⚠️ Resting Blood Pressure ({user_input['resting_blood_pressure']:.0f}) is in a hypertensive crisis range.")
    if user_input['oldpeak'] > 4:
        warnings_list.append(f"⚠️ Oldpeak value ({user_input['oldpeak']:.2f}) is unusually extreme.")

    if warnings_list:
        st.warning(
            "**Input Validation Notice:** This combination of values includes one or more "
            "physiologically unusual readings, which are rare or absent in the training data. "
            "The model's prediction below may be unreliable for this input.\n\n"
            + "\n\n".join(warnings_list)
        )

    # Encode categorical columns using the REAL saved label encoders (no manual guessing)
    input_encoded = input_df.copy()
    for col, le in label_encoders.items():
        input_encoded[col] = le.transform(input_encoded[col])

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error(f"⚠️ **Prediction: Disease Present**")
        else:
            st.success(f"✅ **Prediction: No Disease**")
        st.metric("Probability of Disease", f"{probability*100:.1f}%")
    with col2:
        st.write("**Patient Input Summary**")
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

    # ----------------------------
    # SHAP Explanation for THIS prediction
    # ----------------------------
    st.subheader("🔎 Why did the model make this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(input_encoded)

    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]
        expected_value = explainer.expected_value[1]
    elif shap_values_raw.ndim == 3:
        shap_values = shap_values_raw[:, :, 1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values = shap_values_raw
        expected_value = explainer.expected_value

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=input_encoded.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)

    st.markdown("""
    **How to read this chart:** Features in **red** push the prediction toward *Disease*,
    features in **blue** push it toward *No Disease*. The length of each bar shows how strong that feature's influence was for this specific patient.
    """)
else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict** to see the result.")