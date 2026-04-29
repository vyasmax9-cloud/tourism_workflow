import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Helper function to load the model
@st.cache_resource
def load_model():
    return joblib.load(hf_hub_download(
        repo_id="vyasmax9/tourism-predict-app",
        filename="model.pkl",
        repo_type="space"
    ))

# Helper function to get feature types from the original dataset (for dynamic form generation)
@st.cache_data
def get_feature_info():
    # Load a sample of the raw data to infer types and categories
    try:
        df_raw = pd.read_csv("tourism_application/data/tourism.csv")
    except FileNotFoundError:
        df_raw = pd.read_csv(f"https://huggingface.co/spaces/vyasmax9/tourism-predict-app/resolve/main/data/tourism.csv")

    df_raw.drop(columns=["Unnamed: 0", "CustomerID", "ProdTaken"], inplace=True, errors="ignore")
    return df_raw.columns, df_raw.dtypes, {col: df_raw[col].unique().tolist() for col in df_raw.select_dtypes(include='object').columns}

st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter customer details to predict their likelihood of purchasing the Wellness Tourism Package.")

model = load_model()
feature_cols, dtypes, categories = get_feature_info()

# Input form for all features
input_data_dict = {}

for col in feature_cols:
    if col == 'Age':
        input_data_dict[col] = st.slider("Age", 18, 70, 35, key=col)
    elif col == 'MonthlyIncome':
        input_data_dict[col] = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0, step=100.0, key=col)
    elif col == 'DurationOfPitch':
        input_data_dict[col] = st.slider("Duration of Pitch (minutes)", 5, 60, 15, key=col)
    elif col == 'NumberOfPersonVisiting':
        input_data_dict[col] = st.slider("Number of Persons Visiting", 1, 5, 2, key=col)
    elif col == 'NumberOfFollowups':
        input_data_dict[col] = st.slider("Number of Follow-ups", 1, 6, 3, key=col)
    elif col == 'NumberOfTrips':
        input_data_dict[col] = st.slider("NumberOfTrips (Annually)", 1, 15, 3, key=col)
    elif col == 'NumberOfChildrenVisiting':
        input_data_dict[col] = st.slider("Number of Children Visiting (below 5)", 0, 3, 0, key=col)
    elif col == 'PitchSatisfactionScore':
        input_data_dict[col] = st.slider("Pitch Satisfaction Score", 1, 5, 3, key=col)
    elif col in ['Passport', 'OwnCar']:
        input_data_dict[col] = st.selectbox(f"Do they have {col.replace('NumberOf', '')}?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key=col)
    elif dtypes[col] == 'object': # Categorical features (strings)
        if col in categories:
            input_data_dict[col] = st.selectbox(col, categories[col], key=col)
        else:
            # Fallback for unexpected categorical (should not happen if get_feature_info is robust)s
            input_data_dict[col] = st.text_input(col, key=col) # Generic text input if categories not found
    elif dtypes[col] in ['int64', 'float64']: # Other numeric features
        input_data_dict[col] = st.number_input(col, value=df_raw[col].mean() if pd.api.types.is_numeric_dtype(df_raw[col]) else 0.0, key=col)

# Convert input dictionary to DataFrame, ensuring correct order
input_df = pd.DataFrame([input_data_dict], columns=feature_cols)

if st.button("Predict Package Purchase"):
    # The model expects the preprocessor to handle scaling/encoding, which is part of the Pipeline
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if prob >= 0.5: # Example threshold, can be tuned
        st.success(f"The customer is likely to purchase the package (Probability: {prob:.2%})")
    else:
        st.info(f"The customer is unlikely to purchase the package (Probability: {prob:.2%})")

    st.metric(label="Purchase Probability", value=f"{prob:.2%}")

st.markdown(" preconceived notion---")
st.info("Note: This prediction is based on the trained Random Forest model and the features provided.")
