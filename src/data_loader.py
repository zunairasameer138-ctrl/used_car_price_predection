"""
data_loader.py

Responsible for loading the raw Used Car Price dataset from disk and
performing lightweight structural validation before any cleaning or
modelling happens.

Dataset: "CAR DETAILS FROM CAR DEKHO.csv"
Source : CarDekho listings, republished on Kaggle by nehalbirla
         (https://www.kaggle.com/datasets/nehalbirla/car-details-from-cardekho)

Expected raw columns (exact, case-sensitive):
    name, year, selling_price, km_driven, fuel, seller_type, transmission, owner
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path is resolved relative to the project root (two levels up from this file:
# src/data_loader.py -> src -> project root), so the loader works no matter
# where the caller's current working directory is.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "CAR DETAILS FROM CAR DEKHO.csv"

# The exact columns that must exist in the raw file for the rest of the
# pipeline (cleaning, feature engineering, training, prediction) to work.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
)

TARGET_COLUMN = "selling_price"


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or fails structural validation."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """
    Load the raw used-car dataset from a CSV or Excel file.

    Parameters
    ----------
    path : str | Path
        Path to the raw data file. Supports .csv, .xlsx and .xls.
        Defaults to data/raw/CAR DETAILS FROM CAR DEKHO.csv

    Returns
    -------
    pd.DataFrame
        The raw dataset, unmodified other than being parsed into a DataFrame.

    Raises
    ------
    DataLoadError
        If the file is missing, empty, unreadable, or missing required columns.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(
            f"Dataset not found at '{path}'.\n"
            f"Download it and place it at: {DEFAULT_RAW_PATH}\n"
            f"See README.md -> 'Dataset' section for the exact download URL."
        )

    if path.stat().st_size == 0:
        raise DataLoadError(f"Dataset file at '{path}' is empty (0 bytes).")

    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise DataLoadError(
                f"Unsupported file extension '{suffix}'. Use .csv, .xlsx or .xls."
            )
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Dataset file at '{path}' has no parsable data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Dataset file at '{path}' could not be parsed: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive catch-all
        raise DataLoadError(f"Failed to read dataset at '{path}': {exc}") from exc

    if df.empty:
        raise DataLoadError(f"Dataset at '{path}' was parsed but contains 0 rows.")

    validate_columns(df, REQUIRED_COLUMNS)

    return df


def validate_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """
    Confirm every required column is present in the DataFrame.

    Raises
    ------
    DataLoadError
        If one or more required columns are missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataLoadError(
            "Dataset is missing required column(s): "
            f"{missing}. Found columns: {list(df.columns)}. "
            "Do not rename/reorder columns in the raw file - keep it exactly "
            "as downloaded from the source."
        )


def load_processed_data(path: str | Path = PROJECT_ROOT / "data" / "processed" / "cleaned_car_data.csv") -> pd.DataFrame:
    """
    Load the cleaned + feature-engineered dataset produced by
    data_cleaning.py + feature_engineering.py (see src/train_model.py).

    Raises
    ------
    DataLoadError
        If the processed file does not exist yet (i.e. the pipeline has not
        been run), or if it cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(
            f"Processed dataset not found at '{path}'.\n"
            "Run `python src/train_model.py` first - it generates the cleaned, "
            "feature-engineered dataset as a side effect of training."
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read processed dataset at '{path}': {exc}") from exc


if __name__ == "__main__":
    # Simple manual smoke test: `python src/data_loader.py`
    data = load_raw_data()
    print(f"Loaded raw dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print(data.head())
