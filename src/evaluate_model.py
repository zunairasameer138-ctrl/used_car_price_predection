"""
evaluate_model.py

Standalone evaluation of the SAVED pipeline (models/car_price_pipeline.pkl)
against a freshly rebuilt test split. This is deliberately independent of
train_model.py's in-memory metrics: it re-loads the pipeline from disk,
so it verifies what is actually saved on disk, not just what was in memory
right after training. Useful for a quick health check any time (e.g. after
pulling someone else's trained model artifact, or before deploying).

Run with:
    python src/evaluate_model.py

It will:
    1. Rebuild the identical clean+engineered dataset and the identical
       train/test split (same random_state, same test_size) used by
       train_model.py, so the test set here is exactly the test set the
       model was originally scored on.
    2. Load models/car_price_pipeline.pkl
    3. Predict on the test split and print MAE / MSE / RMSE / R^2
    4. Save an Actual vs Predicted scatter plot to
       reports/figures/actual_vs_predicted.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_cleaning import clean_data
from data_loader import load_raw_data
from feature_engineering import engineer_features
from preprocessing import (
    FIGURES_DIR,
    ML_FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


class ModelEvaluationError(Exception):
    """Raised when the saved model cannot be evaluated."""


def load_pipeline():
    if not MODEL_PATH.exists():
        raise ModelEvaluationError(
            f"No saved model found at '{MODEL_PATH}'.\n"
            "Run `python src/train_model.py` first to train and save a pipeline."
        )
    return joblib.load(MODEL_PATH)


def rebuild_test_split():
    """Recreate the exact same train/test split train_model.py used, so this
    script scores the saved model on data it has genuinely never seen."""
    raw = load_raw_data()
    clean, _ = clean_data(raw)
    full = engineer_features(clean)
    model_df = full.dropna(subset=ML_FEATURES + [TARGET]).reset_index(drop=True)
    X = model_df[ML_FEATURES]
    y = model_df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def main() -> None:
    pipeline = load_pipeline()
    _, X_test, _, y_test = rebuild_test_split()

    y_pred = pipeline.predict(X_test)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    print("=== Saved pipeline evaluation (models/car_price_pipeline.pkl) ===")
    print(f"Test rows : {len(X_test)}")
    print(f"MAE       : {mae:,.2f}")
    print(f"MSE       : {mse:,.2f}")
    print(f"RMSE      : {rmse:,.2f}")
    print(f"R^2       : {r2:.4f}")

    # Actual vs Predicted scatter plot ---------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color="#2563eb", edgecolor="none")
    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Selling Price (INR)")
    ax.set_ylabel("Predicted Selling Price (INR)")
    ax.set_title("Actual vs Predicted - Saved Pipeline")
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved chart -> {out_path}")


if __name__ == "__main__":
    main()
