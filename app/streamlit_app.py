import streamlit as st
import joblib
import pandas as pd
from src.optimize_conditions import recommend_optimal_config
from src.prepare_data import load_and_prepare_data

st.title("🚚 Shipment Delivery Optimizer")

# --- Context Inputs ---
context = {}
context['Weather'] = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Stormy"])
context['Traffic'] = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
context['Area'] = st.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
context['Category'] = st.selectbox("Category", ["Clothing", "Electronics", "Furniture", "Food"])
context['DayOfWeek'] = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
context['Order_Hour'] = st.slider("Order Hour", 0, 23, 10)
context['Pickup_Delay_Minutes'] = st.slider("Pickup Delay (mins)", 0, 120, 20)
context['Distance_km'] = st.slider("Distance (km)", 0.5, 20.0, 5.0)

if st.button("🔍 Optimize Delivery"):
    model = joblib.load("models/delivery_time_model.pkl")
    _, _, _, encoder = load_and_prepare_data("/data/from_s3_dummy.csv")  # just to reuse encoder
    result = recommend_optimal_config(context, model, encoder)
    st.success("✅ Best Configuration Found!")
    st.table(pd.DataFrame([result]))
