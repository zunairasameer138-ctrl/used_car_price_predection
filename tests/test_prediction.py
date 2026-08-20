"""
tests/test_prediction.py

Covers: src/predict.py - input validation, single predictions, error
handling for missing models/unknown categories, and the What-If Simulator's
predict_sensitivity() helper.

Run with: pytest tests/test_prediction.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import MODEL_PATH
from predict import (
    PredictionError,
    build_input_dataframe,
    load_pipeline,
    predict_price,
    predict_sensitivity,
    validate_inputs,
)

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/car_price_pipeline.pkl not found - run `python src/train_model.py` first.",
)

BASE_INPUT = dict(
    brand="Maruti", year=2015, km_driven=60000, fuel="Petrol",
    seller_type="Individual", transmission="Manual", owner="First Owner",
)


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


# --------------------------------------------------------------------------
# validate_inputs
# --------------------------------------------------------------------------

def test_validate_inputs_accepts_reasonable_values():
    assert validate_inputs(year=2015, km_driven=50000) == []


def test_validate_inputs_flags_bad_year():
    problems = validate_inputs(year=1800, km_driven=1000)
    assert problems


def test_validate_inputs_flags_negative_km():
    problems = validate_inputs(year=2015, km_driven=-100)
    assert problems


# --------------------------------------------------------------------------
# build_input_dataframe
# --------------------------------------------------------------------------

def test_build_input_dataframe_has_one_row():
    df = build_input_dataframe(**BASE_INPUT)
    assert len(df) == 1
    assert df.loc[0, "name"] == "Maruti Unknown"


# --------------------------------------------------------------------------
# load_pipeline
# --------------------------------------------------------------------------

def test_load_pipeline_missing_file_raises(tmp_path):
    with pytest.raises(PredictionError):
        load_pipeline(tmp_path / "does_not_exist.pkl")


def test_load_pipeline_succeeds(pipeline):
    assert hasattr(pipeline, "predict")


# --------------------------------------------------------------------------
# predict_price
# --------------------------------------------------------------------------

def test_predict_price_returns_positive_float(pipeline):
    price = predict_price(pipeline, **BASE_INPUT)
    assert isinstance(price, float)
    assert price > 0


def test_predict_price_invalid_input_raises(pipeline):
    with pytest.raises(PredictionError):
        predict_price(pipeline, **{**BASE_INPUT, "year": 1700})


def test_predict_price_handles_two_word_brand(pipeline):
    price = predict_price(pipeline, **{**BASE_INPUT, "brand": "Land Rover"})
    assert price > 0


def test_predict_price_handles_unseen_brand_gracefully(pipeline):
    # OneHotEncoder(handle_unknown="ignore") must prevent a crash here.
    price = predict_price(pipeline, **{**BASE_INPUT, "brand": "TotallyUnseenBrandXYZ"})
    assert price > 0


def test_predict_price_newer_car_is_worth_more(pipeline):
    older = predict_price(pipeline, **{**BASE_INPUT, "year": 2010})
    newer = predict_price(pipeline, **{**BASE_INPUT, "year": 2019})
    assert newer > older


def test_predict_price_higher_mileage_is_worth_less(pipeline):
    low_km = predict_price(pipeline, **{**BASE_INPUT, "km_driven": 10000})
    high_km = predict_price(pipeline, **{**BASE_INPUT, "km_driven": 150000})
    assert high_km < low_km


# --------------------------------------------------------------------------
# predict_sensitivity (powers the What-If Simulator)
# --------------------------------------------------------------------------

def test_predict_sensitivity_returns_one_price_per_value(pipeline):
    values = [0, 50000, 100000, 150000]
    prices = predict_sensitivity(pipeline, BASE_INPUT, "km_driven", values)
    assert len(prices) == len(values)
    assert all(p > 0 for p in prices)


def test_predict_sensitivity_km_driven_is_monotonically_non_increasing(pipeline):
    values = list(range(0, 200_001, 20_000))
    prices = predict_sensitivity(pipeline, BASE_INPUT, "km_driven", values)
    assert all(prices[i] >= prices[i + 1] - 1 for i in range(len(prices) - 1))


def test_predict_sensitivity_year_is_monotonically_non_decreasing(pipeline):
    values = list(range(1992, 2021, 2))
    prices = predict_sensitivity(pipeline, BASE_INPUT, "year", values)
    assert all(prices[i] <= prices[i + 1] + 1 for i in range(len(prices) - 1))


def test_predict_sensitivity_unknown_field_raises(pipeline):
    with pytest.raises(PredictionError):
        predict_sensitivity(pipeline, BASE_INPUT, "not_a_real_field", [1, 2, 3])


def test_predict_sensitivity_matches_predict_price_at_baseline(pipeline):
    """Sweeping a field and landing exactly on the baseline value should
    reproduce the same price predict_price() gives directly."""
    direct = predict_price(pipeline, **BASE_INPUT)
    swept = predict_sensitivity(pipeline, BASE_INPUT, "km_driven", [BASE_INPUT["km_driven"]])
    assert swept[0] == pytest.approx(direct)
