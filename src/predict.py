"""
predict.py

Turns raw user input (the fields collected on the Streamlit "Price
Prediction" page) into a single price estimate, using the exact same
feature engineering used during training.

Workflow
--------
    user input (dict)
        -> build_input_dataframe()   # 1-row raw-schema DataFrame
        -> engineer_features()       # SAME function train_model.py uses
        -> pipeline.predict()        # SAME saved pipeline train_model.py fit
        -> float price (INR)

Why a synthetic `name` is built from `brand`
---------------------------------------------
The dataset's only free-text field is `name` (e.g. "Hyundai Verna 1.6 SX"),
which feature_engineering.engineer_features() parses into `brand` + `model`.
A user filling in a prediction form picks `brand` directly from a dropdown
(there is no reason to make them type a full listing title), so
build_input_dataframe() reconstructs a minimal `name` string
("<brand> Unknown") purely so it can be run through the *identical*
engineer_features() call path used at training time - see the note at the
top of feature_engineering.py for why that matters. `extract_brand_model`
round-trips this correctly for every brand, including the two-word
"Land Rover".
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engineering import engineer_features
from preprocessing import ML_FEATURES, MODEL_PATH


class PredictionError(Exception):
    """Raised when a price prediction cannot be produced."""


# Populated once from the processed dataset by the Streamlit app / callers;
# kept here too so predict.py works standalone without the app.
KNOWN_FUEL_TYPES = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
KNOWN_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
KNOWN_TRANSMISSIONS = ["Manual", "Automatic"]
KNOWN_OWNERS = ["First Owner", "Second Owner", "Third Owner",
                "Fourth & Above Owner", "Test Drive Car"]


def load_pipeline(path: str | Path = MODEL_PATH):
    """Load the saved (preprocessing + model) pipeline from disk.

    Raises
    ------
    PredictionError
        If the model file does not exist (i.e. training has not been run
        yet) or fails to load.
    """
    path = Path(path)
    if not path.exists():
        raise PredictionError(
            f"No trained model found at '{path}'.\n"
            "Run `python src/train_model.py` first to train and save the "
            "price prediction pipeline."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise PredictionError(f"Failed to load model at '{path}': {exc}") from exc


def build_input_dataframe(
    brand: str,
    year: int,
    km_driven: float,
    fuel: str,
    seller_type: str,
    transmission: str,
    owner: str,
) -> pd.DataFrame:
    """Build the single-row raw-schema DataFrame engineer_features() expects."""
    row = {
        "name": f"{brand} Unknown",
        "year": int(year),
        "selling_price": 0.0,  # placeholder - not used, engineer_features doesn't need it
        "km_driven": float(km_driven),
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }
    return pd.DataFrame([row])


def validate_inputs(year: int, km_driven: float) -> list[str]:
    """Return a list of human-readable validation problems (empty = valid)."""
    problems: list[str] = []
    if year < 1900 or year > 2100:
        problems.append(f"Year {year} is not a plausible manufacturing year.")
    if km_driven < 0:
        problems.append("Kilometers driven cannot be negative.")
    if km_driven > 2_000_000:
        problems.append("Kilometers driven looks unrealistically high (>2,000,000 km).")
    return problems


def predict_price(
    pipeline,
    brand: str,
    year: int,
    km_driven: float,
    fuel: str,
    seller_type: str,
    transmission: str,
    owner: str,
) -> float:
    """
    Predict the selling price (INR) for a used car with the given attributes.

    Raises
    ------
    PredictionError
        If inputs fail validation or the pipeline cannot produce a prediction.
    """
    problems = validate_inputs(year, km_driven)
    if problems:
        raise PredictionError(" ".join(problems))

    raw_row = build_input_dataframe(brand, year, km_driven, fuel, seller_type, transmission, owner)

    try:
        engineered = engineer_features(raw_row)
        X = engineered[ML_FEATURES]
        prediction = pipeline.predict(X)[0]
    except KeyError as exc:
        raise PredictionError(f"Missing expected feature column: {exc}") from exc
    except Exception as exc:
        raise PredictionError(f"Prediction failed: {exc}") from exc

    return max(0.0, float(prediction))


def predict_sensitivity(
    pipeline,
    base_inputs: dict,
    vary_field: str,
    values: list,
) -> list[float]:
    """
    Re-predict price while sweeping ONE field across `values`, holding every
    other field at its value in `base_inputs`. Powers the What-If Simulator's
    sensitivity chart (e.g. "how does price change as km_driven goes from
    0 to 200,000, holding brand/year/fuel/... fixed?").

    Parameters
    ----------
    pipeline
        Loaded pipeline (from load_pipeline()).
    base_inputs : dict
        Must contain brand, year, km_driven, fuel, seller_type,
        transmission, owner - the same keyword arguments predict_price()
        takes.
    vary_field : str
        Which key of base_inputs to sweep. Must be one of that dict's keys.
    values : list
        The sequence of values to substitute for `vary_field`.

    Returns
    -------
    list[float]
        One predicted price per entry in `values`, same order.
    """
    if vary_field not in base_inputs:
        raise PredictionError(f"Cannot sweep unknown field '{vary_field}'.")

    prices = []
    for v in values:
        trial_inputs = dict(base_inputs)
        trial_inputs[vary_field] = v
        prices.append(predict_price(pipeline, **trial_inputs))
    return prices


if __name__ == "__main__":
    # Manual smoke test: `python src/predict.py`
    pipe = load_pipeline()
    price = predict_price(
        pipe,
        brand="Maruti",
        year=2017,
        km_driven=45000,
        fuel="Petrol",
        seller_type="Individual",
        transmission="Manual",
        owner="First Owner",
    )
    print(f"Estimated price: Rs. {price:,.0f}")
