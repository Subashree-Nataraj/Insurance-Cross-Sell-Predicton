import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgb_model.pkl")
columns = joblib.load("models/columns.pkl")

st.set_page_config(page_title="Insurance Predictor", layout="wide")

st.title("🚗 Insurance Cross-Sell Prediction App")

# ---------------- INPUT UI ----------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region_code = st.number_input("Region Code", value=28)

with col2:
    previously_insured = st.selectbox("Previously Insured", [0, 1])
    annual_premium = st.number_input("Annual Premium", value=40000)
    vintage = st.number_input("Vintage", value=100)

with col3:
    vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    policy_channel = st.number_input("Policy Sales Channel", value=26)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict"):

    # Step 1: create empty row
    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # ---------------- NUMERIC ----------------
    input_df.loc[0, "Age"] = age
    input_df.loc[0, "Region_Code"] = region_code
    input_df.loc[0, "Previously_Insured"] = previously_insured
    input_df.loc[0, "Annual_Premium"] = annual_premium
    input_df.loc[0, "Policy_Sales_Channel"] = policy_channel
    input_df.loc[0, "Vintage"] = vintage

    # IMPORTANT (you missed this earlier)
    if "Driving_License" in columns:
        input_df.loc[0, "Driving_License"] = 1

    # ---------------- CATEGORICAL ----------------

    # Gender
    if "Gender_Male" in columns:
        input_df.loc[0, "Gender_Male"] = 1 if gender == "Male" else 0

    # Vehicle Damage
    if "Vehicle_Damage_Yes" in columns:
        input_df.loc[0, "Vehicle_Damage_Yes"] = 1 if vehicle_damage == "Yes" else 0

    # Vehicle Age (MATCH CLEANED NAMES)
    if "Vehicle_Age__1_Year" in columns:
        input_df.loc[0, "Vehicle_Age__1_Year"] = 1 if vehicle_age == "< 1 Year" else 0

    if "Vehicle_Age__2_Years" in columns:
        input_df.loc[0, "Vehicle_Age__2_Years"] = 1 if vehicle_age == "> 2 Years" else 0

    # ---------------- AGE GROUPS ----------------
    if 26 <= age <= 35 and "Age_Groups_26_35" in columns:
        input_df.loc[0, "Age_Groups_26_35"] = 1

    elif 36 <= age <= 45 and "Age_Groups_36_45" in columns:
        input_df.loc[0, "Age_Groups_36_45"] = 1

    elif 46 <= age <= 55 and "Age_Groups_46_55" in columns:
        input_df.loc[0, "Age_Groups_46_55"] = 1

    elif 56 <= age <= 65 and "Age_Groups_56_65" in columns:
        input_df.loc[0, "Age_Groups_56_65"] = 1

    elif age >= 66 and "Age_Groups_66_" in columns:
        input_df.loc[0, "Age_Groups_66_"] = 1

    # ---------------- FINAL ALIGNMENT ----------------
    input_df = input_df[columns]

    # ---------------- PREDICT ----------------
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write(f"Probability: {prob:.2f}")

    if prob > 0.3:
        st.success("Customer is likely to respond ✅")
    else:
        st.error("Customer is not likely to respond ❌")

    # DEBUG (keep this while testing)
    st.subheader("🧪 Active Features")
    st.write(input_df.T[input_df.T[0] != 0])