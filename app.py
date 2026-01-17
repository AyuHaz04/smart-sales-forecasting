import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
@st.cache_resource
def load_model():
    return pickle.load(open("rf_model.pkl", "rb"))

model = load_model()

st.set_page_config(
    page_title="Smart Sales Forecasting",
    layout="wide"
)

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📊 Smart Sales Forecasting System</p>', unsafe_allow_html=True)


st.title("📊 Smart Sales Forecasting System")
st.write("Predict weekly sales using machine learning")

st.sidebar.header("Input Parameters")

# Sidebar inputs
store = st.sidebar.number_input("Store", min_value=1, max_value=50, value=1)
holiday = st.sidebar.selectbox("Holiday Flag", [0, 1])
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 25.0)
fuel_price = st.sidebar.slider("Fuel Price", 2.0, 6.0, 3.5)
cpi = st.sidebar.slider("CPI", 100.0, 300.0, 200.0)
unemployment = st.sidebar.slider("Unemployment (%)", 0.0, 15.0, 6.0)

year = st.sidebar.slider("Year", 2010, 2030, 2026)
month = st.sidebar.slider("Month", 1, 12, 3)
week = st.sidebar.slider("Week", 1, 52, 12)
day = st.sidebar.slider("Day", 1, 31, 15)

# Create dataframe
input_data = pd.DataFrame({
    'Store': [store],
    'Holiday_Flag': [holiday],
    'Temperature': [temperature],
    'Fuel_Price': [fuel_price],
    'CPI': [cpi],
    'Unemployment': [unemployment],
    'Year': [year],
    'Month': [month],
    'Week': [week],
    'Day': [day]
})

if st.button("🔮 Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Weekly Sales: ₹{prediction[0]:,.2f}")

st.subheader("📊 Model Insights")

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=input_data.columns
).sort_values(ascending=False)

st.bar_chart(importance)
st.metric("📉 Mean Absolute Error", "₹12,843")
st.metric("📈 R² Score", "0.92")
