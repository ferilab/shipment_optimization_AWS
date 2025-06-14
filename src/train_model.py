
# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import joblib, os, warnings, io
import boto3

warnings.filterwarnings("ignore")

def train_delivery_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"\nModel RMSE on test set: {rmse:.2f} minutes")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/delivery_time_model.pkl")

    return model


def upload_to_s3(model_path, encoder, bucket="shipment-optimization-bucket", prefix="models/"):
    s3 = boto3.client("s3")

    # Upload model file
    s3.upload_file(model_path, bucket, f"{prefix}delivery_time_model.pkl")

    # Serialize encoder to memory and upload
    encoder_buffer = io.BytesIO()
    joblib.dump(encoder, encoder_buffer)
    encoder_buffer.seek(0)
    s3.upload_fileobj(encoder_buffer, bucket, f"{prefix}encoder.pkl")

    print("✅ Model and encoder uploaded to S3.")