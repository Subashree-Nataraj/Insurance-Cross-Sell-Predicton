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

    # Create full input dataframe with zeros
    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # Fill numerical features
    input_df["Age"] = age
    input_df["Region_Code"] = region_code
    input_df["Previously_Insured"] = previously_insured
    input_df["Annual_Premium"] = annual_premium
    input_df["Vintage"] = vintage
    input_df["Policy_Sales_Channel"] = policy_channel

    # Gender encoding
    if gender == "Male" and "Gender_Male" in columns:
        input_df["Gender_Male"] = 1

    # Vehicle Damage
    if vehicle_damage == "Yes" and "Vehicle_Damage_Yes" in columns:
        input_df["Vehicle_Damage_Yes"] = 1

    # Vehicle Age encoding
    if vehicle_age == "1-2 Year":
        input_df["Vehicle_Age_1-2 Year"] = 1
    elif vehicle_age == "> 2 Years":
        input_df["Vehicle_Age_> 2 Years"] = 1

    # ---------------- PREDICTION ----------------
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= threshold else 0

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Result")

    st.metric(label="Probability of Response", value=f"{prob:.2f}")

    if prediction == 1:
        st.success("✅ Customer is LIKELY to respond")
    else:
        st.error("❌ Customer is NOT likely to respond")

    st.write(f"Threshold used: {threshold}")