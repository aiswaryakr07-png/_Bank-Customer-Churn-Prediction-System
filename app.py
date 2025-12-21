import streamlit as st
import pickle
import pandas as pd

st.title("🏦 Bank Customer Churn Prediction")

credit_score = st.number_input("Credit Score", 350, 850, 650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Account Balance", 0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, value=60000.0)

with open("churn.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
geo_encoder = data["onehot"]
feature_names = data["feature_names"]

gender_mapping = {"Male": 1, "Female": 0}

if st.button("Predict Churn"):

    gender_val = gender_mapping[gender]

    num_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }])

    geo_encoded = geo_encoder.transform([[geography]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=geo_encoder.get_feature_names_out(["Geography"])
    )

    final_df = pd.concat([num_df, geo_df], axis=1)
    final_df = final_df[feature_names]

    scaled_input = scaler.transform(final_df)

    probability = model.predict_proba(scaled_input)[0][1]
    prediction = 1 if probability >= 0.6 else 0

    if prediction == 1:
        st.error(f"❌ Customer likely to CHURN\n\n🔥 Risk: {probability:.2%}")
    else:
        st.success(f"✅ Customer NOT likely to churn\n\n📉 Risk: {probability:.2%}")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.sanity.io/images/b7pblshe/marketing-prod/8a5c51fa72084f767822b62a4f93bf726a7c9b22-1200x628.png");
        background-size: cover;
    }
    div.stButton > button {
        background-color: #4A00E0 !important;
        color: white !important;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 25px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #6A00FF !important;
    }
    div[data-testid="stAlert"] {
        background-color: #000000 !important;
        font-size: 18px !important;
        border-radius: 12px !important;
    }
    div[data-testid="stAlert"].stAlert-success {
        color: #3CFF3C !important;
        border-left: 6px solid #3CFF3C !important;
    }
    div[data-testid="stAlert"].stAlert-error {
        color: #FF4C4C !important;
        border-left: 6px solid #FF4C4C !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



