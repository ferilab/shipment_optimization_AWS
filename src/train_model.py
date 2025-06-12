
import argparse
import pandas as pd
import joblib
from shipment_optimization_AWS.prepare_data import load_and_prepare_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import boto3
import os


def download_file_from_s3(s3_path, local_path):
    s3 = boto3.client('s3')
    bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket_name, key, local_path)


def main(args):
    # Download CSV from S3
    local_data_path = "./temp_data.csv"
    if args.input_s3_path.startswith("s3://"):
        download_file_from_s3(args.input_s3_path, local_data_path)
        input_path = local_data_path
    else:
        input_path = args.input_s3_path

    df_final, feature_cols, target_col, encoder = load_and_prepare_data(input_path)
    X = df_final[feature_cols]
    y = df_final[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    rmse = sqrt(mean_squared_error(y, y_pred))
    print(f"Training RMSE: {rmse:.2f}")

    # Save model and encoder
    joblib.dump(model, args.output_model_path)
    joblib.dump(encoder, args.output_encoder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train delivery model")
    parser.add_argument("--input_s3_path", type=str, required=True, help="S3 path to input data or local path")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save trained model")
    parser.add_argument("--output_encoder_path", type=str, required=True, help="Path to save encoder")
    args = parser.parse_args()
    main(args)
