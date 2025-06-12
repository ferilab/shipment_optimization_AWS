
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from geopy.distance import geodesic
import os
import boto3
import io
import warnings
warnings.filterwarnings("ignore")

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# The Haversine distance between pick up and delivery points needs to be measured 
def haversine_distance(row):
    loc1 = (row['Store_Latitude'], row['Store_Longitude'])
    loc2 = (row['Drop_Latitude'], row['Drop_Longitude'])
    return geodesic(loc1, loc2).km

# Feature engineering. We create new features that could be of importance in delivery
def extract_temporal_features(df):
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Order_Time'] = pd.to_datetime(df['Order_Time'])
    df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'])

    df['DayOfWeek'] = df['Order_Date'].dt.day_name()
    df['Order_Hour'] = df['Order_Time'].dt.hour
    df['Pickup_Delay_Minutes'] = (df['Pickup_Time'] - df['Order_Time']).dt.total_seconds() / 60
    return df

def load_and_prepare_data(s3_path):
    s3 = boto3.client('s3')
    bucket = 'shipment-optimization-bucket'
    key = 'amazon_delivery.csv'
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    # Drop missing or corrupted rows
    df = df.dropna(subset=['Agent_Age', 'Agent_Rating', 'Vehicle', 'Weather', 'Traffic',
                           'Area', 'Category', 'Delivery_Time',
                           'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude',
                           'Order_Date', 'Order_Time', 'Pickup_Time'])

    df = extract_temporal_features(df)
    df['Distance_km'] = df.apply(haversine_distance, axis=1)

    # Select features and target
    # Note: The reqiired features are: ['Vehicle', 'Agent_Age', 'Agent_Rating', 'Weather', 'Traffic', 
    #              'Area', 'Category', 'DayOfWeek', 'Order_Hour', 'Pickup_Delay_Minutes', 'Distance_km']
    target_col = 'Delivery_Time'

    # Encode categorical features and make a dataframe from them with corresponding appropriate column names
    categorical_cols = ['Vehicle', 'Weather', 'Traffic', 'Area', 'Category', 'DayOfWeek']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    # GradientBoostingRegressor (and most models) can’t handle string — it expects purely numeric input.
    # Here we replace uncoded categorical, raw datetime and geo columns with their encoded and numeric versions 
    # before model training.
    df_final = pd.concat([df.drop(columns=categorical_cols + ['Order_Date', 'Order_Time', 'Pickup_Time',
                    'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']), encoded_df], axis=1)

    # Of course, the order_id and target shouldn't be in input variable.
    final_feature_cols = [col for col in df_final.columns if col not in ['Order_ID', target_col]]

    return df_final, final_feature_cols, target_col, encoder
