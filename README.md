# 🚗 Insurance Cross-Sell Prediction Project

## 📌 Project Overview
This project predicts whether a customer will respond positively to an insurance cross-sell offer based on demographic, vehicle, and policy-related features. The goal is to help optimize marketing campaigns by targeting the right customers and reducing unnecessary outreach.

---

## 🎯 Objective
- Improve marketing efficiency
- Reduce cost of customer acquisition
- Identify potential customers likely to respond to insurance offers

---

## 📊 Dataset
- Source: Insurance Cross-Sell Prediction Dataset (Kaggle)
- Features include:
  - Age, Gender
  - Vehicle Age, Vehicle Damage
  - Previously Insured
  - Annual Premium
  - Policy Sales Channel, Region Code, Vintage

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit (for deployment)

---

## 🔍 Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering (One-hot encoding, transformations)  
4. Model Building:
   - Logistic Regression (Baseline)
   - Random Forest
   - Gradient Boosting
   - XGBoost (Final Model)
5. Threshold Tuning (Optimal threshold = 0.3)  
6. Model Evaluation (ROC-AUC, Precision, Recall)  
7. Deployment using Streamlit  

---

## 📈 Model Performance

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | 0.83 |
| Random Forest | 0.84 |
| Gradient Boosting | 0.855 |
| XGBoost | **0.857** |

---

## 🎯 Final Model
- Model: XGBoost  
- Threshold: 0.3  
- Selected based on best balance between precision and recall  

---

## 🚀 Streamlit App
The model is deployed using Streamlit where users can input customer details and get real-time predictions.

### Run locally:
```bash
streamlit run app.py
