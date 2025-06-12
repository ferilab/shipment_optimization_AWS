
# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
import joblib, os, warnings

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
