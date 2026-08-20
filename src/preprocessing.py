"""
preprocessing.py

Defines the scikit-learn preprocessing pipeline shared by training
(src/train_model.py) and inference (src/predict.py, app/app.py).

Feature groups
--------------
ML_NUMERIC_FEATURES:
    car_age, km_driven, mileage_per_year
    -> SimpleImputer(median) + StandardScaler

ML_CATEGORICAL_FEATURES:
    brand, fuel, seller_type, transmission, owner
    -> SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")

Why `model` (the free-text remainder of the car name, e.g. "Wagon R LXI
Minor") is NOT an ML feature: it is close to a unique identifier per row
(thousands of distinct strings across ~3.5k rows). One-hot encoding it would
create thousands of near-useless sparse columns that mostly memorize
individual listings instead of generalizing, and any unseen model string at
prediction time would carry zero learned signal anyway. `brand` captures the
generalizable part of that information instead. `model`/`name` are kept in
the processed dataset purely for display in the Market Analytics page.

`km_driven_category` is also excluded from the ML features: it is a coarser,
binned copy of `km_driven`, which the model already receives at full
numeric resolution, so including both would be redundant.

`OneHotEncoder(handle_unknown="ignore")` is critical for a small dataset
like this one: several brands (Kia, Isuzu, Daewoo, Force, ...) have only a
single row. If that row lands in the test split (or a user later asks for a
prediction on a brand the training data never saw), `handle_unknown="ignore"`
prevents a hard crash - the model simply falls back to the information in
the other features instead of raising an error.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ML_NUMERIC_FEATURES: list[str] = ["car_age", "km_driven", "mileage_per_year"]
ML_CATEGORICAL_FEATURES: list[str] = ["brand", "fuel", "seller_type", "transmission", "owner"]
ML_FEATURES: list[str] = ML_NUMERIC_FEATURES + ML_CATEGORICAL_FEATURES
TARGET = "selling_price"

# Single source of truth for every path shared between train_model.py,
# evaluate_model.py, predict.py and app/app.py. Every one of those files
# imports these constants instead of hard-coding the path again, so the
# filename the trainer *saves* and the filename the app *loads* can never
# drift apart.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_car_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "car_price_pipeline.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
TEST_PREDICTIONS_PATH = REPORTS_DIR / "test_predictions.csv"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor() -> ColumnTransformer:
    """Build the (unfitted) ColumnTransformer used inside every model pipeline."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, ML_NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, ML_CATEGORICAL_FEATURES),
    ])
    return preprocessor


if __name__ == "__main__":
    pre = build_preprocessor()
    print(pre)
    print("\nNumeric features:", ML_NUMERIC_FEATURES)
    print("Categorical features:", ML_CATEGORICAL_FEATURES)
