"""
data_cleaning.py

Cleans the raw "CAR DETAILS FROM CAR DEKHO" dataset.

This module encodes decisions made after actually inspecting the real data
(see reports/eda_summary.txt / README.md for the inspection output):

    * 4,340 rows, 8 columns, ZERO missing values in the raw file.
    * 763 fully-duplicated rows (same listing scraped/entered more than once).
    * Two rows have a data-entry glitch in `name`: "OpelCorsa" is missing the
      space between brand and model ("OpelCorsa 1.6Gls" / "OpelCorsa 1.4 GL").
    * `year` ranges 1992-2020, `selling_price` ranges 20,000-8,900,000 (INR),
      `km_driven` ranges 1-806,599. All are already plausible / no impossible
      values (e.g. no negative prices, no year 0), but we still defensively
      guard against them so the pipeline stays correct if the raw CSV is
      ever refreshed with messier data.

Nothing is dropped silently: every cleaning step returns counts, and
`clean_data` returns a report dict alongside the cleaned DataFrame so callers
(train_model.py, tests, notebooks) can log/inspect exactly what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as _dt

import pandas as pd

# Minimum plausible manufacturing year for a used car in this dataset.
# (The real data starts at 1992; 1900 is a generous lower safety bound in
# case the raw file is ever refreshed with older vehicles.)
MIN_VALID_YEAR = 1900
MAX_VALID_YEAR = _dt.datetime.now().year + 1  # allow next-year models

# Known raw `name` string glitches -> fix, found by manual inspection.
NAME_FIXES = {
    "OpelCorsa": "Opel Corsa",
}

STRING_COLUMNS = ["name", "fuel", "seller_type", "transmission", "owner"]


@dataclass
class CleaningReport:
    """Transparent record of every row-removing decision made during cleaning."""

    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    missing_value_rows_dropped: int = 0
    invalid_year_rows_dropped: int = 0
    invalid_price_rows_dropped: int = 0
    invalid_km_rows_dropped: int = 0
    name_glitches_fixed: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Run the full cleaning pipeline on the raw dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by data_loader.load_raw_data().

    Returns
    -------
    (clean_df, report) : tuple[pd.DataFrame, CleaningReport]
        The cleaned dataframe (new index, no duplicates, only valid rows)
        and a report describing exactly what was changed/removed and why.
    """
    report = CleaningReport(rows_in=len(df))
    out = df.copy()

    # ------------------------------------------------------------------
    # 1) String cleaning: strip stray whitespace so categories like
    #    "Petrol" and " Petrol " aren't treated as different categories.
    # ------------------------------------------------------------------
    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    # ------------------------------------------------------------------
    # 2) Fix known name-entry glitches (brand/model glued together).
    #    Why: `name` is later split on whitespace to derive `brand` in
    #    feature_engineering.py. Left unfixed, "OpelCorsa 1.6Gls" would be
    #    parsed as brand="OpelCorsa" instead of the real brand "Opel".
    # ------------------------------------------------------------------
    for bad, good in NAME_FIXES.items():
        mask = out["name"].str.startswith(bad)
        n = int(mask.sum())
        if n:
            out.loc[mask, "name"] = out.loc[mask, "name"].str.replace(
                bad, good, n=1, regex=False
            )
            report.name_glitches_fixed += n

    # ------------------------------------------------------------------
    # 3) Drop rows with missing values in required columns.
    #    Why: the raw file currently has zero missing values, but every
    #    downstream step (encoding, scaling, prediction) assumes complete
    #    rows for these columns, so we guard defensively for future data
    #    refreshes rather than assume today's cleanliness always holds.
    # ------------------------------------------------------------------
    before = len(out)
    out = out.dropna(subset=["name", "year", "selling_price", "km_driven",
                              "fuel", "seller_type", "transmission", "owner"])
    report.missing_value_rows_dropped = before - len(out)

    # ------------------------------------------------------------------
    # 4) Remove exact duplicate rows.
    #    Why: 763 of the 4,340 raw rows are byte-for-byte duplicates of
    #    another row (identical name, year, price, km, fuel, seller type,
    #    transmission and owner). These almost certainly represent the same
    #    listing captured more than once and would otherwise let the model
    #    (and the "most common brand" KPI) silently over-weight repeated
    #    listings.
    # ------------------------------------------------------------------
    before = len(out)
    out = out.drop_duplicates(keep="first")
    report.duplicates_dropped = before - len(out)

    # ------------------------------------------------------------------
    # 5) Validate `year`. Why: a used car cannot be manufactured before
    #    cars existed, nor in the future (data entry error).
    # ------------------------------------------------------------------
    before = len(out)
    out = out[(out["year"] >= MIN_VALID_YEAR) & (out["year"] <= MAX_VALID_YEAR)]
    report.invalid_year_rows_dropped = before - len(out)

    # ------------------------------------------------------------------
    # 6) Validate `selling_price`. Why: price must be strictly positive -
    #    zero or negative prices are data errors, not real transactions.
    # ------------------------------------------------------------------
    before = len(out)
    out = out[out["selling_price"] > 0]
    report.invalid_price_rows_dropped = before - len(out)

    # ------------------------------------------------------------------
    # 7) Validate `km_driven`. Why: distance driven cannot be negative.
    #    (No upper cap here - 806,599 km on a ~28 year old commercial-use
    #    vehicle is unusual but physically plausible, so we keep it and let
    #    the tree-based models handle the long tail rather than deleting
    #    real high-usage vehicles.)
    # ------------------------------------------------------------------
    before = len(out)
    out = out[out["km_driven"] >= 0]
    report.invalid_km_rows_dropped = before - len(out)

    # ------------------------------------------------------------------
    # 8) Enforce dtypes and reset index.
    # ------------------------------------------------------------------
    out["year"] = out["year"].astype(int)
    out["selling_price"] = out["selling_price"].astype(float)
    out["km_driven"] = out["km_driven"].astype(float)
    out = out.reset_index(drop=True)

    report.rows_out = len(out)
    report.notes.append(
        f"{report.rows_in} raw rows -> {report.rows_out} clean rows "
        f"({report.duplicates_dropped} exact duplicates removed)."
    )

    return out, report


if __name__ == "__main__":
    # Manual smoke test: `python src/data_cleaning.py`
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned, rep = clean_data(raw)
    print("Cleaning report:")
    for k, v in rep.as_dict().items():
        print(f"  {k}: {v}")
