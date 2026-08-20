"""
tests/test_model.py

Covers: model/pipeline loading and the shape of what train_model.py saves.
Requires `python src/train_model.py` to have been run at least once (these
tests read the saved artifacts rather than retraining, so they stay fast).

Run with: pytest tests/test_model.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import (
    DATA_PROCESSED_PATH,
    METRICS_PATH,
    ML_FEATURES,
    MODEL_PATH,
    TARGET,
    TEST_PREDICTIONS_PATH,
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/car_price_pipeline.pkl not found - run `python src/train_model.py` first.",
)


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Expected trained pipeline at {MODEL_PATH}"


def test_model_loads_and_has_predict():
    pipeline = joblib.load(MODEL_PATH)
    assert hasattr(pipeline, "predict")
    assert isinstance(pipeline, TransformedTargetRegressor)


def test_model_predicts_on_processed_feature_columns():
    pipeline = joblib.load(MODEL_PATH)
    assert DATA_PROCESSED_PATH.exists(), "Run train_model.py to generate the processed dataset."
    df = pd.read_csv(DATA_PROCESSED_PATH)
    X_sample = df[ML_FEATURES].head(5)
    preds = pipeline.predict(X_sample)
    assert len(preds) == 5
    assert (preds >= 0).all(), "Predicted prices should never be negative"


def test_metrics_file_has_expected_shape():
    assert METRICS_PATH.exists(), "Run train_model.py to generate reports/metrics.json."
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    assert "models" in metrics and len(metrics["models"]) == 3
    assert set(metrics["models"].keys()) == {"Linear Regression", "Random Forest", "XGBoost"}
    assert "best_model" in metrics and metrics["best_model"] in metrics["models"]

    for name, m in metrics["models"].items():
        for key in ("MAE", "MSE", "RMSE", "R2"):
            assert key in m, f"{name} is missing metric {key}"
            assert isinstance(m[key], (int, float))
        assert m["MAE"] > 0
        assert m["RMSE"] >= m["MAE"] * 0  # RMSE should always be non-negative
        assert m["RMSE"] > 0


def test_best_model_has_lowest_rmse():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    rmses = {name: m["RMSE"] for name, m in metrics["models"].items()}
    best_by_rmse = min(rmses, key=rmses.get)
    assert metrics["best_model"] == best_by_rmse


def test_test_predictions_file_matches_test_row_count():
    assert TEST_PREDICTIONS_PATH.exists()
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    preds_df = pd.read_csv(TEST_PREDICTIONS_PATH)
    assert len(preds_df) == metrics["dataset"]["test_rows"]
    assert "actual_selling_price" in preds_df.columns
