import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="vyasmax9/tourism-predict-app", filename="best_tourism_app_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Prediction App")
st.write("""
Predict whether a customer will purchase the Wellness Tourism Package""".
""")

age = st.number_input("Age", 18, 70, 30)
income = st.number_input("Monthly Income", 1000, 200000, 50000)

typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
citytier = st.selectbox("City Tier", [1, 2, 3])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "Luxury"])

children = st.number_input("Number of Children Visiting", 0, 5, 0)

# Create DataFrame (IMPORTANT)
input_df = pd.DataFrame([{
        'Age': age,
        'NumberOfChildrenVisiting': children,
        'MonthlyIncome': income,
        'TypeofContact': typeofcontact,
        'Occupation': occupation,
        'Gender': gender,
        'CityTier': citytier,
        'MaritalStatus': maritalstatus,
        'PreferredPropertyStar': preferredpropertystar,
        'Designation': designation,
        'ProductPitched': productpitched
    }])

# MODEL PREDICTION
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Customer will purchase the Wellness Tourism Package" if prediction == 1 else "Customer will not purchase the Wellness Tourism Package"
    st.success(result)
    st.subheader("Prediction Probability")
    prediction_proba = model.predict_proba(input_df)
    st.write(prediction_proba)
    if prediction == 1:
      st.subheader("Prediction")
      st.write(f"Prediction: {prediction}")
      st.subheader("Prediction Probability")
      st.write(f"Probability of Purchase: {prediction_proba[0][1]}")

    else:
      st.subheader("Prediction")
      st.write(f"Prediction: {prediction}")
      st.subheader("Prediction Probability")
      st.write(f"Probability of Purchase: {prediction_proba[0][
