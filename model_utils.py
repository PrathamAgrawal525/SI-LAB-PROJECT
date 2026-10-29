"""
AutoVision AI - Core Machine Learning & Valuation Utilities
Shared model loading, training, and prediction logic for Gradio and Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "car_data.csv"
MODEL_PATH = BASE_DIR / "car_price_model.pkl"

FEATURE_NAMES = [
    "Present_Price",
    "Kms_Driven",
    "Age",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
]

EXPECTED_DUMMY_COLUMNS = [
    "Present_Price",
    "Kms_Driven",
    "Age",
    "Fuel_Type_Diesel",
    "Fuel_Type_Petrol",
    "Seller_Type_Individual",
    "Transmission_Manual",
]


def load_car_data(csv_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load and prepare raw car data."""
    path = Path(csv_path) if csv_path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    if "Age" not in df.columns and "Year" in df.columns:
        df["Age"] = 2026 - df["Year"]
    return df


def train_model(
    csv_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Multiple Linear Regression model and export bundle."""
    csv_file = Path(csv_path) if csv_path else DATA_PATH
    model_file = Path(model_path) if model_path else MODEL_PATH

    df = load_car_data(csv_file)
    X_raw = df[FEATURE_NAMES]
    y = df["Selling_Price"]

    X = pd.get_dummies(X_raw, drop_first=True)
    # Ensure expected dummy columns match standard order
    for col in EXPECTED_DUMMY_COLUMNS:
        if col not in X.columns:
            X[col] = 0.0
    X = X[EXPECTED_DUMMY_COLUMNS].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))

    bundle = {
        "model": regressor,
        "feature_columns": list(X.columns),
        "metrics": {
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mse": round(mse, 4),
        },
    }

    joblib.dump(bundle, model_file)
    return bundle


def load_model(
    model_path: Optional[Path | str] = None,
    csv_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Load existing model bundle or train a new one if missing."""
    model_file = Path(model_path) if model_path else MODEL_PATH

    if model_file.exists():
        try:
            bundle = joblib.load(model_file)
            if isinstance(bundle, dict) and "model" in bundle and "feature_columns" in bundle:
                return bundle
        except Exception:
            pass

    return train_model(csv_path=csv_path, model_path=model_path)


def predict_price(bundle: Dict[str, Any], values: Dict[str, Any]) -> float:
    """Predict selling price (in ₹ Lakhs) for a given car configuration."""
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", EXPECTED_DUMMY_COLUMNS)

    row = pd.DataFrame([values])
    encoded = pd.get_dummies(row).reindex(columns=feature_columns, fill_value=0).astype(float)

    raw_prediction = float(model.predict(encoded)[0])
    return max(0.0, raw_prediction)


def get_feature_importances(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Extract model feature coefficients sorted by absolute impact."""
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", EXPECTED_DUMMY_COLUMNS)

    if hasattr(model, "coef_"):
        coeffs = model.coef_
    elif hasattr(model, "named_steps") and "regressor" in model.named_steps:
        coeffs = model.named_steps["regressor"].coef_
    else:
        coeffs = np.zeros(len(feature_columns))

    df_coef = pd.DataFrame({"feature": feature_columns, "impact": coeffs})
    df_coef["abs_impact"] = df_coef["impact"].abs()
    return df_coef.sort_values("abs_impact", ascending=False).reset_index(drop=True)
