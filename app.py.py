
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Customer Engagement Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Engagement & Product Utilization Analytics")
st.subheader("Retention Strategy Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_customer_data.csv")

df = load_data()

st.sidebar.header("Customer Filters")

engagement = st.sidebar.selectbox(
    "Engagement Status",
    ["All", "Active", "Inactive"]
)

min_balance = st.sidebar.number_input(
    "Minimum Balance",
    min_value=0.0,
    value=0.0
)

product_count = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    (1, 4)
)

filtered = df.copy()

if engagement != "All":
    filtered = filtered[
        filtered["Engagement_Status"] == engagement
    ]

filtered = filtered[
    (filtered["Balance"] >= min_balance) &
    (filtered["NumOfProducts"].between(
        product_count[0], product_count[1]
    ))
]

total = len(filtered)

churn = (
    filtered["Exited"].mean() * 100
    if total else 0
)

active = (
    filtered["IsActiveMember"].mean() * 100
    if total else 0
)

c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", total)
c2.metric("Churn Rate", f"{churn:.2f}%")
c3.metric("Active Customers", f"{active:.2f}%")

st.divider()

st.header("1. Engagement vs Churn")

if total:
    engagement_data = (
        filtered.groupby("Engagement_Status")["Exited"]
        .mean().reset_index()
    )

    fig1 = px.bar(
        engagement_data,
        x="Engagement_Status",
        y="Exited",
        color="Engagement_Status",
        title="Engagement vs Churn"
    )

    st.plotly_chart(fig1, use_container_width=True)

st.header("2. Product Utilization Analysis")

if total:
    product_data = (
        filtered.groupby("NumOfProducts")["Exited"]
        .mean().reset_index()
    )

    fig2 = px.bar(
        product_data,
        x="NumOfProducts",
        y="Exited",
        title="Product Count vs Churn"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.header("3. High Value Disengaged Customers")

if total:
    threshold = filtered["Balance"].quantile(0.75)

    risk = filtered[
        (filtered["Balance"] >= threshold) &
        (filtered["IsActiveMember"] == 0)
    ]

    st.metric("High Value Disengaged Customers", len(risk))

    st.dataframe(
        risk[[
            "Geography",
            "Age",
            "Balance",
            "NumOfProducts",
            "Exited"
        ]],
        use_container_width=True
    )

st.header("4. Relationship Strength")

if total:
    tier = (
        filtered["Relationship_Tier"]
        .value_counts().reset_index()
    )

    tier.columns = ["Relationship Tier", "Customers"]

    fig3 = px.pie(
        tier,
        names="Relationship Tier",
        values="Customers"
    )

    st.plotly_chart(fig3, use_container_width=True)

st.header("5. Churn Prediction")

model = joblib.load("churn_model.pkl")

with st.form("prediction_form"):

    credit_score = st.number_input("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0)
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    card = st.selectbox("Has Credit Card", [0, 1])
    active_member = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", min_value=0.0)

    submit = st.form_submit_button("Predict Churn")

if submit:

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.metric("Churn Probability", f"{probability:.2%}")

    if prediction == 1:
        st.warning("Customer is predicted to be at risk of churn.")
    else:
        st.success("Customer is predicted to be likely to stay.")

st.caption("Customer Engagement Analytics | Retention Strategy")
