import streamlit as st
import pandas as pd
import joblib

# Load model + columns
model = joblib.load("xgb_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Insurance Cross-Sell Predictor", layout="wide")

st.title("🚗 Insurance Cross-Sell Prediction App")
st.write("Enter customer details to predict response probability")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Prediction Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3)

# ---------------- MAIN LAYOUT ----------------

col1, col2, col3 = st.columns(3)

# --------- COLUMN 1: CUSTOMER INFO ---------
with col1:
    st.subheader("👤 Customer Info")
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region_code = st.number_input("Region Code", min_value=0, value=1)

# --------- COLUMN 2: INSURANCE INFO ---------
with col2:
    st.subheader("🛡️ Insurance Info")
    previously_insured = st.selectbox("Previously Insured", [0, 1])
    annual_premium = st.number_input("Annual Premium", value=30000)
    vintage = st.number_input("Vintage (days)", value=100)

# --------- COLUMN 3: VEHICLE INFO ---------
with col3:
    st.subheader("🚗 Vehicle Info")
    vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    policy_channel = st.number_input("Policy Sales Channel", value=1)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Response"):

    # ✅ Step 1: Create empty dataframe with ALL training columns
    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # ✅ Step 2: Fill numeric features
    input_df.loc[0, "Age"] = age
    input_df.loc[0, "Previously_Insured"] = previously_insured
    input_df.loc[0, "Annual_Premium"] = annual_premium
    input_df.loc[0, "Region_Code"] = region_code
    input_df.loc[0, "Policy_Sales_Channel"] = policy_channel
    input_df.loc[0, "Vintage"] = vintage

    # ✅ Step 3: Handle categorical (VERY IMPORTANT)

    # Gender
    if "Gender_Male" in columns:
        input_df.loc[0, "Gender_Male"] = 1 if gender == "Male" else 0

    # Vehicle Damage
    if "Vehicle_Damage_Yes" in columns:
        input_df.loc[0, "Vehicle_Damage_Yes"] = 1 if vehicle_damage == "Yes" else 0

    # Vehicle Age
    if "Vehicle_Age_1-2 Year" in columns:
        input_df.loc[0, "Vehicle_Age_1-2 Year"] = 1 if vehicle_age == "1-2 Year" else 0

    if "Vehicle_Age_> 2 Years" in columns:
        input_df.loc[0, "Vehicle_Age_> 2 Years"] = 1 if vehicle_age == "> 2 Years" else 0

    # ✅ Step 4: Ensure correct column order (CRITICAL)
    input_df = input_df[columns]

    # ✅ Step 5: Prediction
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= threshold else 0

    # ✅ Step 6: Output
    st.subheader("Result")
    st.write(f"Probability: {prob:.2f}")

    if prediction == 1:
        st.success("Customer is likely to respond ✅")
    else:
        st.error("Customer is not likely to respond ❌")