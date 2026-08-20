"""
feature_engineering.py

Derives new columns from the cleaned "CAR DETAILS FROM CAR DEKHO" dataset.

IMPORTANT CONSISTENCY RULE
---------------------------
`engineer_features()` is called on the FULL dataset during training
(src/train_model.py) and on a SINGLE-ROW DataFrame built from user input
during prediction (src/predict.py, app/app.py). Both call sites import this
exact same function, so the two can never drift apart.

Because of that, REFERENCE_YEAR below is a fixed constant (the latest model
year actually present in the training data: 2020) rather than something
computed dynamically as `df['year'].max()`. If it were computed dynamically,
a single-row prediction dataframe would always have max(year) == that row's
own year, forcing car_age to 0 for every prediction - a subtle but serious
train/serve skew bug. Fixing it as a constant avoids that entirely.

Engineered features
--------------------
brand              : str   - first (or first two, for known multi-word
                              manufacturers) tokens of `name`.
model               : str   - remainder of `name` after the brand. Kept for
                              display/analytics only (NOT used as an ML
                              feature - see note in `ML_CATEGORICAL_FEATURES`
                              in preprocessing.py for why).
car_age             : int   - REFERENCE_YEAR - year. Formula: how many years
                              old the car was as of the dataset's collection
                              year. More useful to a regression model than
                              raw `year`, which it replaces as the numeric
                              time feature (car_age and year carry identical
                              information, so only one is fed to the model).
mileage_per_year    : float - km_driven / max(car_age, 1). Formula: average
                              annual usage. Purpose: two cars with the same
                              odometer reading tell a different story if one
                              is 2 years old and the other is 15 - this
                              feature captures "used hard" vs "used lightly".
                              Potential leakage: none - both inputs (km_driven,
                              year) are known at listing time, before a sale
                              price exists. It is, however, collinear with
                              km_driven and car_age individually; this is
                              fine for tree models (Random Forest/XGBoost)
                              and acceptable for Linear Regression since we
                              only care about predictive accuracy, not
                              interpreting individual coefficients.
km_driven_category  : str   - Low/Medium/High/Very High usage bucket, for
                              readable analytics charts only (not fed to the
                              ML model - it's a coarser copy of km_driven,
                              which the model already receives at full
                              resolution).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed on purpose - see module docstring above.
REFERENCE_YEAR = 2020

# Manufacturers whose name in the `name` column is two tokens wide.
# Found by inspecting `df['name'].str.split().str[0].value_counts()` on the
# real dataset - every other brand is a single token (hyphens like
# "Mercedes-Benz" already come through str.split() as one token).
KNOWN_TWO_WORD_BRANDS = ["Land Rover"]

KM_DRIVEN_BINS = [-np.inf, 30_000, 70_000, 120_000, np.inf]
KM_DRIVEN_LABELS = ["Low (<30k km)", "Medium (30k-70k km)",
                     "High (70k-120k km)", "Very High (>120k km)"]


def extract_brand_model(name: str) -> tuple[str, str]:
    """Split a `name` string like 'Land Rover Discovery S 2.0 SD4' into
    ('Land Rover', 'Discovery S 2.0 SD4')."""
    name = str(name).strip()
    for brand in KNOWN_TWO_WORD_BRANDS:
        if name.startswith(brand):
            remainder = name[len(brand):].strip()
            return brand, remainder if remainder else "Unknown"
    parts = name.split(maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], "Unknown"
    return "Unknown", "Unknown"


def engineer_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """
    Add engineered columns to a cleaned dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing at least: name, year, km_driven.
    reference_year : int
        Year used to compute car_age. Defaults to the fixed REFERENCE_YEAR
        constant so training and prediction always agree.

    Returns
    -------
    pd.DataFrame
        Copy of `df` with brand, model, car_age, mileage_per_year and
        km_driven_category appended.
    """
    out = df.copy()

    brand_model = out["name"].apply(extract_brand_model)
    out["brand"] = brand_model.apply(lambda t: t[0])
    out["model"] = brand_model.apply(lambda t: t[1])

    out["car_age"] = (reference_year - out["year"]).astype(int)
    # A brand-new (age 0) car still has an annual usage rate - avoid divide
    # by zero by treating age 0 the same as age 1 for this ratio only.
    safe_age = out["car_age"].clip(lower=1)
    out["mileage_per_year"] = (out["km_driven"] / safe_age).round(1)

    out["km_driven_category"] = pd.cut(
        out["km_driven"], bins=KM_DRIVEN_BINS, labels=KM_DRIVEN_LABELS
    ).astype(str)

    return out


if __name__ == "__main__":
    # Manual smoke test: `python src/feature_engineering.py`
    from data_loader import load_raw_data
    from data_cleaning import clean_data

    raw = load_raw_data()
    cleaned, _ = clean_data(raw)
    engineered = engineer_features(cleaned)
    print(engineered[["name", "brand", "model", "year", "car_age",
                       "km_driven", "mileage_per_year", "km_driven_category"]].head(10))
    print("\nBrand counts:\n", engineered["brand"].value_counts())
