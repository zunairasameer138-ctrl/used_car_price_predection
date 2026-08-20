"""
tests/test_data.py

Covers: data loading, required-column validation, and the cleaning pipeline.
Run with: pytest tests/test_data.py -v  (or just `pytest` from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_cleaning import clean_data
from data_loader import DataLoadError, REQUIRED_COLUMNS, load_raw_data, validate_columns


# --------------------------------------------------------------------------
# load_raw_data
# --------------------------------------------------------------------------

def test_load_raw_data_returns_nonempty_dataframe():
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_raw_data_has_required_columns():
    df = load_raw_data()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required column: {col}"


def test_load_raw_data_missing_file_raises():
    with pytest.raises(DataLoadError):
        load_raw_data("data/raw/this_file_does_not_exist.csv")


def test_load_raw_data_empty_file_raises(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    with pytest.raises(DataLoadError):
        load_raw_data(empty_file)


def test_load_raw_data_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("name,year\nA,2020\n")
    with pytest.raises(DataLoadError):
        load_raw_data(bad_file)


# --------------------------------------------------------------------------
# validate_columns
# --------------------------------------------------------------------------

def test_validate_columns_passes_with_all_required():
    df = pd.DataFrame({col: [] for col in REQUIRED_COLUMNS})
    validate_columns(df)  # should not raise


def test_validate_columns_raises_with_missing_column():
    df = pd.DataFrame({"name": [], "year": []})  # missing most required columns
    with pytest.raises(DataLoadError):
        validate_columns(df)


# --------------------------------------------------------------------------
# clean_data
# --------------------------------------------------------------------------

def test_clean_data_removes_exact_duplicates():
    raw = load_raw_data()
    cleaned, report = clean_data(raw)
    assert cleaned.duplicated().sum() == 0
    assert report.duplicates_dropped > 0  # the real dataset has 763 known duplicates
    assert report.rows_out == len(cleaned)
    assert report.rows_out < report.rows_in


def test_clean_data_fixes_opelcorsa_glitch():
    raw = load_raw_data()
    cleaned, _ = clean_data(raw)
    # every remaining name should have brand and model separated by a space
    glued = cleaned["name"].str.startswith("OpelCorsa")
    assert glued.sum() == 0
    assert (cleaned["name"].str.startswith("Opel Corsa")).sum() >= 1


def test_clean_data_all_prices_positive():
    raw = load_raw_data()
    cleaned, _ = clean_data(raw)
    assert (cleaned["selling_price"] > 0).all()


def test_clean_data_all_km_driven_non_negative():
    raw = load_raw_data()
    cleaned, _ = clean_data(raw)
    assert (cleaned["km_driven"] >= 0).all()


def test_clean_data_handles_synthetic_dirty_input():
    """clean_data() should also behave sensibly on a small synthetic frame
    with an injected duplicate, a bad year, and a non-positive price -
    not just on the real, mostly-clean dataset."""
    dirty = pd.DataFrame({
        "name": ["Maruti Swift", "Maruti Swift", "Hyundai i20", "Old Timer", "Free Car"],
        "year": [2015, 2015, 2018, 1850, 2016],       # row 4 (1850) is invalid
        "selling_price": [300000, 300000, 450000, 100000, 0],  # last row invalid (0)
        "km_driven": [40000, 40000, 25000, 90000, 10000],
        "fuel": ["Petrol", "Petrol", "Petrol", "Diesel", "Petrol"],
        "seller_type": ["Individual"] * 5,
        "transmission": ["Manual"] * 5,
        "owner": ["First Owner"] * 5,
    })
    cleaned, report = clean_data(dirty)
    assert report.duplicates_dropped == 1
    assert report.invalid_year_rows_dropped == 1
    assert report.invalid_price_rows_dropped == 1
    assert len(cleaned) == 2  # only the two genuinely valid, unique rows remain
