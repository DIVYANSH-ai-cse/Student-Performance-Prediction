"""
train_model.py
---------------
Trains a RandomForestRegressor to predict a student's final score (0-100)
from academic and lifestyle features, evaluates it, and saves the
trained pipeline (scaler + model) to model/student_model.pkl.

Run:
    python generate_dataset.py     # first, to create data/student_data.csv
    python train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_dataset import generate_dataset

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_scores",
    "assignments_completed",
    "sleep_hours",
    "extracurricular",
    "parental_support",
    "internet_access",
]
TARGET_COLUMN = "final_score"

DATA_PATH = os.path.join("data", "student_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "student_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CHART_PATH = os.path.join("static", "feature_importance.png")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("Dataset not found, generating a new one...")
        os.makedirs("data", exist_ok=True)
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH)


def train():
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model evaluation on test set:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R^2  : {r2:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")

    # Feature importance chart
    os.makedirs("static", exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(FEATURE_COLUMNS)[order], importances[order], color="#4f6df5")
    plt.xlabel("Importance")
    plt.title("Feature Importance - Student Performance Model")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120)
    print(f"Saved feature importance chart to {CHART_PATH}")


if __name__ == "__main__":
    train()
