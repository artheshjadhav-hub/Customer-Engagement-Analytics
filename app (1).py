
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="Customer Engagement Analytics",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    return pd.read_csv(
        os.path.join(BASE_DIR, "cleaned_customer_data.csv")
    )

df = load_data()

st.title("Customer Engagement & Product Utilization Analytics")

st.header("Customer Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Customers", len(df))
c2.metric("Churn Rate", f"{df['Exited'].mean()*100:.2f}%")
c3.metric("Active Customers", int(df["IsActiveMember"].sum()))

st.header("Engagement vs Churn")

engagement_data = (
    df.groupby("Engagement_Status")["Exited"]
    .mean().reset_index()
)

fig = px.bar(
    engagement_data,
    x="Engagement_Status",
    y="Exited",
    color="Engagement_Status"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Product Utilization")

product_data = (
    df.groupby("NumOfProducts")["Exited"]
    .mean().reset_index()
)

fig2 = px.bar(
    product_data,
    x="NumOfProducts",
    y="Exited"
)

st.plotly_chart(fig2, use_container_width=True)

st.header("Customer Data")

st.dataframe(df.head(100), use_container_width=True)
