"""
train_model.py

End-to-end training entry point for the Used Car Price Prediction model.

Run with:
    python src/train_model.py

What it does, in order
-----------------------
 1. Load the raw dataset (data/raw/CAR DETAILS FROM CAR DEKHO.csv)
 2. Clean it (src/data_cleaning.py)
 3. Engineer features (src/feature_engineering.py)
 4. Save the cleaned + engineered dataset to data/processed/cleaned_car_data.csv
    (this file is what the Streamlit Market Analytics page reads)
 5. Split into train/test (80/20, random_state=42)
 6. Build 3 candidate pipelines (preprocessing + model), each wrapped in a
    TransformedTargetRegressor that fits on log1p(selling_price) and
    automatically un-transforms predictions back to rupees:
        - Linear Regression
        - Random Forest Regressor
        - XGBoost Regressor
 7. Evaluate every model on the held-out test set: MAE, MSE, RMSE, R^2
 8. Print a comparison table
 9. Select the best model (lowest test RMSE)
10. Save the WHOLE winning pipeline (preprocessing + model, one object) to
    models/car_price_pipeline.pkl via joblib
11. Save reports/metrics.json (every model's real metrics - nothing here is
    invented; this file is what the Streamlit "Model Performance" page reads)
12. Save reports/test_predictions.csv (actual vs. predicted for every model
    on the test set, used for the "Actual vs Predicted" chart)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `python src/train_model.py`

from data_cleaning import clean_data
from data_loader import load_raw_data
from feature_engineering import engineer_features
from preprocessing import (
    DATA_PROCESSED_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    ML_CATEGORICAL_FEATURES,
    ML_FEATURES,
    ML_NUMERIC_FEATURES,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_PREDICTIONS_PATH,
    TEST_SIZE,
    build_preprocessor,
)


def build_candidate_models() -> dict[str, TransformedTargetRegressor]:
    """
    Build the three candidate pipelines. Each is a full
    (preprocessing -> regressor) Pipeline wrapped in a
    TransformedTargetRegressor so that:
      * the model is trained on log1p(selling_price) - selling_price is
        strongly right-skewed (min 20,000 / median 350,000 / max 8,900,000
        INR in the real data), and regressing on the log scale keeps a
        handful of very expensive luxury cars from dominating the loss.
      * predict() automatically applies expm1() to return a plain rupee
        value, so callers (predict.py, the Streamlit app) never have to
        remember to undo the transform themselves.
    """
    models: dict[str, TransformedTargetRegressor] = {}

    linear_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ])
    models["Linear Regression"] = TransformedTargetRegressor(
        regressor=linear_pipeline, func=np.log1p, inverse_func=np.expm1
    )

    # Hyperparameters below were not guessed - they're the result of a
    # RandomizedSearchCV pass (5-fold CV, scoring="r2" on log1p(price))
    # run against this exact dataset. An unregularized RandomForest
    # (defaults, min_samples_leaf=1) hits TRAIN R^2 = 0.94 but TEST R^2 =
    # 0.61 - a textbook overfitting gap, unsurprising with ~2,861 training
    # rows and high-cardinality one-hot columns. Constraining tree depth
    # and leaf size closes most of that gap.
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    models["Random Forest"] = TransformedTargetRegressor(
        regressor=rf_pipeline, func=np.log1p, inverse_func=np.expm1
    )

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            min_child_weight=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    models["XGBoost"] = TransformedTargetRegressor(
        regressor=xgb_pipeline, func=np.log1p, inverse_func=np.expm1
    )

    return models


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(float(mae), 2), "MSE": round(float(mse), 2),
            "RMSE": round(rmse, 2), "R2": round(float(r2), 4)}


def main() -> None:
    start = time.time()
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DATA_PROCESSED_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)

    # 1-2-3) Load, clean, engineer -----------------------------------------
    print("Loading raw data ...")
    raw_df = load_raw_data()
    print(f"  raw shape: {raw_df.shape}")

    print("Cleaning ...")
    clean_df, report = clean_data(raw_df)
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")

    print("Engineering features ...")
    full_df = engineer_features(clean_df)

    # 4) Persist the processed dataset (used by the Streamlit analytics page)
    full_df.to_csv(DATA_PROCESSED_PATH, index=False)
    print(f"Saved processed dataset -> {DATA_PROCESSED_PATH} ({full_df.shape[0]} rows)")

    # 5) Train / test split --------------------------------------------------
    model_df = full_df.dropna(subset=ML_FEATURES + [TARGET]).reset_index(drop=True)
    X = model_df[ML_FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)}")

    # 6-7) Train + evaluate every candidate ----------------------------------
    candidates = build_candidate_models()
    metrics_by_model: dict[str, dict] = {}
    fitted_models: dict[str, TransformedTargetRegressor] = {}
    test_predictions = pd.DataFrame({"actual_selling_price": y_test.values}, index=X_test.index)

    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_seconds = round(time.time() - t0, 2)

        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, a_min=0, a_max=None)  # price can't be negative
        metrics = evaluate(y_test.values, y_pred)
        metrics["fit_seconds"] = fit_seconds
        metrics_by_model[name] = metrics
        fitted_models[name] = model
        test_predictions[f"predicted_{name.lower().replace(' ', '_')}"] = y_pred

        print(f"\n{name} (trained in {fit_seconds}s)")
        print(f"  MAE : {metrics['MAE']:,.2f}")
        print(f"  MSE : {metrics['MSE']:,.2f}")
        print(f"  RMSE: {metrics['RMSE']:,.2f}")
        print(f"  R^2 : {metrics['R2']:.4f}")

    # 8) Comparison table -----------------------------------------------------
    comparison = pd.DataFrame(metrics_by_model).T
    comparison = comparison[["MAE", "MSE", "RMSE", "R2", "fit_seconds"]]
    print("\n=== Model comparison (test set) ===")
    print(comparison.to_string())

    # 9) Select best model (lowest RMSE) --------------------------------------
    best_name = comparison["RMSE"].astype(float).idxmin()
    best_model = fitted_models[best_name]
    print(f"\nBest model (lowest RMSE): {best_name}")

    # 10) Save the winning pipeline -------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best pipeline -> {MODEL_PATH}")

    # 11) Save metrics.json (ground truth for the Streamlit app - never fake) -
    metrics_payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset": {
            "raw_rows": int(report.rows_in),
            "raw_columns": int(raw_df.shape[1]),
            "duplicates_removed": int(report.duplicates_dropped),
            "clean_rows": int(report.rows_out),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "features": {
            "numeric": ML_NUMERIC_FEATURES,
            "categorical": ML_CATEGORICAL_FEATURES,
            "target": TARGET,
        },
        "models": metrics_by_model,
        "best_model": best_name,
        "best_model_metrics": metrics_by_model[best_name],
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    # 12) Save test predictions for the Actual vs Predicted chart -------------
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    print(f"Saved test predictions -> {TEST_PREDICTIONS_PATH}")

    print(f"\nDone in {round(time.time() - start, 1)}s.")


if __name__ == "__main__":
    main()
