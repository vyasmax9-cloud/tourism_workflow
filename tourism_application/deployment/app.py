import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ------------------ LOAD MODEL ------------------
model_path = hf_hub_download(
    repo_id="vyasmax9/tourism-predict-app",
    filename="best_tourism_app_v1.joblib"
)
model = joblib.load(model_path)

# ------------------ UI ------------------
st.title("Tourism Prediction App")
st.write("Predict whether a customer will purchase the Wellness Tourism Package")

# ------------------ INPUTS ------------------
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

# ----------- ADDITIONAL FEATURES -----------
duration = st.number_input("Duration Of Pitch", 1, 60, 10)
persons = st.number_input("Number Of Persons Visiting", 1, 10, 2)
followups = st.number_input("Number Of Followups", 0, 10, 2)
trips = st.number_input("Number Of Trips", 0, 20, 3)
pitchscore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)

passport = st.selectbox("Passport", ["Yes", "No"])
owncar = st.selectbox("Own Car", ["Yes", "No"])

# ------------------ ENCODING ------------------
typeofcontact_map = {"Company Invited": 0, "Self Inquiry": 1}
occupation_map = {"Salaried": 0, "Freelancer": 1}
gender_map = {"Male": 0, "Female": 1}
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
designation_map = {"Executive": 0, "Manager": 1, "Senior Manager": 2}
product_map = {
    "Basic": 0,
    "Standard": 1,
    "Deluxe": 2,
    "Super Deluxe": 3,
    "Luxury": 4
}

# ------------------ DATAFRAME ------------------
input_df = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofcontact_map[typeofcontact],
    'CityTier': citytier,
    'DurationOfPitch': duration,
    'Occupation': occupation_map[occupation],
    'Gender': gender_map[gender],
    'NumberOfPersonVisiting': persons,
    'NumberOfFollowups': followups,
    'ProductPitched': product_map[productpitched],
    'PreferredPropertyStar': preferredpropertystar,
    'MaritalStatus': marital_map[maritalstatus],
    'NumberOfTrips': trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitchscore,
    'OwnCar': 1 if owncar == "Yes" else 0,
    'NumberOfChildrenVisiting': children,
    'Designation': designation_map[designation],
    'MonthlyIncome': income
}])

# ----------- FORCE COLUMN ORDER -----------
input_df = input_df[[
    'Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender',
    'NumberOfPersonVisiting','NumberOfFollowups','ProductPitched',
    'PreferredPropertyStar','MaritalStatus','NumberOfTrips','Passport',
    'PitchSatisfactionScore','OwnCar','NumberOfChildrenVisiting',
    'Designation','MonthlyIncome'
]]

# ------------------ PREDICTION ------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)

    result = "Customer WILL purchase the package" if prediction == 1 else "Customer will NOT purchase the package"
    st.success(result)

    st.subheader("Prediction Probability")
    st.write(f"Probability of Purchase: {prediction_proba[0][1]:.2f}")


