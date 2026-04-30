import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("xgb_model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Insurance Cross-Sell Predictor", layout="wide")

st.title("🚗 Insurance Cross-Sell Prediction App")
st.write("Enter customer details to predict response probability")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.3)

# ---------------- INPUT UI ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Info")
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    region_code = st.number_input("Region Code", min_value=0, value=1)

with col2:
    st.subheader("🛡️ Insurance Info")
    previously_insured = st.selectbox("Previously Insured", [0, 1])
    annual_premium = st.number_input("Annual Premium", value=30000)
    vintage = st.number_input("Vintage (days)", value=100)

with col3:
    st.subheader("🚗 Vehicle Info")
    vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    policy_channel = st.number_input("Policy Sales Channel", value=1)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Response"):

    # Step 1: create empty dataframe with model columns
    input_df = pd.DataFrame(0, index=[0], columns=columns)

    # Step 2: numeric features
    if "Age" in columns:
        input_df.loc[0, "Age"] = age

    if "Previously_Insured" in columns:
        input_df.loc[0, "Previously_Insured"] = previously_insured

    if "Annual_Premium" in columns:
        input_df.loc[0, "Annual_Premium"] = annual_premium

    if "Region_Code" in columns:
        input_df.loc[0, "Region_Code"] = region_code

    if "Policy_Sales_Channel" in columns:
        input_df.loc[0, "Policy_Sales_Channel"] = policy_channel

    if "Vintage" in columns:
        input_df.loc[0, "Vintage"] = vintage

    # Step 3: categorical features

    # Gender
    if "Gender_Male" in columns:
        input_df.loc[0, "Gender_Male"] = 1 if gender == "Male" else 0

    # Vehicle Damage
    if "Vehicle_Damage_Yes" in columns:
        input_df.loc[0, "Vehicle_Damage_Yes"] = 1 if vehicle_damage == "Yes" else 0

    # Vehicle Age
    if "Vehicle_Age_< 1 Year" in columns:
        input_df.loc[0, "Vehicle_Age_< 1 Year"] = 1 if vehicle_age == "< 1 Year" else 0

    if "Vehicle_Age_1-2 Year" in columns:
        input_df.loc[0, "Vehicle_Age_1-2 Year"] = 1 if vehicle_age == "1-2 Year" else 0

    if "Vehicle_Age_> 2 Years" in columns:
        input_df.loc[0, "Vehicle_Age_> 2 Years"] = 1 if vehicle_age == "> 2 Years" else 0

    # ---------------- FIX: AGE GROUPS ----------------
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

    # Step 4: ensure correct column order
    input_df = input_df[columns]

    # Step 5: prediction
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= threshold else 0

    # Step 6: debug view (VERY useful)
    st.subheader("🧪 Debug Input (active features only)")
    st.write(input_df[input_df.sum() > 0])

    # Step 7: output
    st.subheader("Result")
    st.write(f"Probability: {prob:.2f}")

    if prediction == 1:
        st.success("Customer is likely to respond ✅")
    else:
        st.error("Customer is not likely to respond ❌")