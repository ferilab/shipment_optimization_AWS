import streamlit as st
import pandas as pd
import numpy as np
import boto3
import joblib
from itertools import product
import io
import warnings

warnings.filterwarnings("ignore")

# --- S3 Config ---
BUCKET = "shipment-optimization-bucket"
MODEL_KEY = "models/delivery_time_model.pkl"
ENCODER_KEY = "models/encoder.pkl"

# --- Load model and encoder from S3 ---
@st.cache_resource
def load_model_and_encoder():
    s3 = boto3.client(
    's3',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
)
    
    model_obj = s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    model = joblib.load(io.BytesIO(model_obj['Body'].read()))

    encoder_obj = s3.get_object(Bucket=BUCKET, Key=ENCODER_KEY)
    encoder = joblib.load(io.BytesIO(encoder_obj['Body'].read()))
    
    return model, encoder

model, encoder = load_model_and_encoder()

# --- UI Elements ---
st.title("🚚 Shipment Optimization Tool")
st.markdown("The tool is trained using Amazon delivery data and suggests the required number of \
            optimum delivery configurations. Define the number of desired configurations as well \
            as delivery conditions (context) of the problem.")
st.markdown("Define the **delivery context** below to get the best delivery configurations.")

opt_num = st.selectbox("How many top options would you like to see?", options=[1, 2, 3, 4, 5], index=2)
opt_num= int(opt_num)  # Ensure opt_num is an integer

weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Windy", "Stormy", "Fog"])
traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
area = st.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
category = st.selectbox("Package Category", ["Electronics", "Grocery", "Clothing", "Furniture"])
dayofweek = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
order_hour = st.slider("Order Hour (0-23)", 0, 23, 10)
pickup_delay = st.slider("Pickup Delay (minutes)", 0, 120, 20)
distance_km = st.slider("Distance (km)", 0.0, 50.0, 5.0)

# --- Optimization ---
def recommend_top_k_configs(context_dict, model, encoder, opt_num=3):
    opt_num= int(opt_num)  # Ensure opt_num is an integer
    vehicles = ['Bike', 'Car', 'Scooter']
    agent_ages = list(range(20, 60, 5))
    agent_ratings = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    combos = list(product(vehicles, agent_ages, agent_ratings))
    df = pd.DataFrame(combos, columns=["Vehicle", "Agent_Age", "Agent_Rating"])

    for k, v in context_dict.items():
        df[k] = v

    cat_cols = ['Vehicle', 'Weather', 'Traffic', 'Area', 'Category', 'DayOfWeek']
    encoded = encoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    model_input = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    df['Predicted_Delivery_Time'] = model.predict(model_input)

    # st.write(df.columns)
    return df.sort_values("Predicted_Delivery_Time").head(opt_num)

# --- Button Trigger ---
# Lets users select context variables via dropdowns/sliders,
if st.button("🔍 Optimize"):
    context = {
        "Weather": weather,
        "Traffic": traffic,
        "Area": area,
        "Category": category,
        "DayOfWeek": dayofweek,
        "Order_Hour": order_hour,
        "Pickup_Delay_Minutes": pickup_delay,
        "Distance_km": distance_km
    }

# Returns the top optimized configurations based on predicted delivery time.
    result_df = recommend_top_k_configs(context, model, encoder, opt_num)
    st.success("Top delivery configurations new4:")
    st.dataframe(result_df[["Vehicle", "Agent_Age", "Agent_Rating", "Predicted_Delivery_Time"]])

