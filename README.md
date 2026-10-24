# Used Car Price Analytics, Prediction & What-If Simulation System

An end-to-end, portfolio-ready data science project: a real dataset, a fully
documented cleaning + feature engineering pipeline, three benchmarked
regression models, and a 6-page Streamlit dashboard with a market analytics
view, a price predictor, and a **What-If Price Simulator**.

> Every number in this README (row counts, metrics, feature importances) was
> produced by actually running this codebase - nothing is invented. Re-run
> `python src/train_model.py` yourself at any time to reproduce them.

---

## Table of Contents

1. [Features](#features)
2. [Screenshots](#screenshots)
3. [Dataset](#dataset)
4. [Tech Stack](#tech-stack)
5. [Project Architecture & Workflow](#project-architecture--workflow)
6. [Folder Structure](#folder-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Model Information & Results](#model-information--results)
10. [Testing](#testing)
11. [Git / GitHub](#git--github)
12. [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
13. [Troubleshooting](#troubleshooting)
14. [Future Improvements](#future-improvements)
15. [Portfolio / Interview Guidance](#portfolio--interview-guidance)
16. [Author](#author)

---

## Features

**Market Analytics**
- Headline KPIs (car count, average/median price, average mileage, most common brand)
- Interactive filters (brand, fuel, transmission, year range, price range)
- Price distribution, brand comparisons, fuel/transmission breakdowns, correlation heatmap

**Price Prediction**
- Enter Brand, Manufacturing Year, Kilometers Driven, Fuel Type, Transmission,
  Seller Type and Ownership -> get an estimated market price
- Built on a single saved Scikit-learn pipeline (preprocessing + model)

**What-If Price Simulator (USP)**
- Set up a baseline vehicle, then adjust specs on a second panel and
  instantly see the new price next to the original, with a rupee and
  percentage delta
- A sensitivity chart sweeps one factor (kilometers driven or manufacturing
  year) across its full range so you can see exactly how the price curve
  responds, with your current scenario marked on it

**Model Performance**
- Real MAE / MSE / RMSE / R² for all three models, actual-vs-predicted chart,
  and an honest explanation of *why* the winning model won

**Developer / Administrator**
- Retrain from the CLI (`python src/train_model.py`), then use the sidebar's
  "Clear cached data & reload" to pick the new model/metrics up in the app
  without restarting it

No authentication is implemented - it isn't needed for this project's scope.

---

## Screenshots

*(Run the app locally with `streamlit run app/app.py` and drop your own
screenshots in `reports/figures/` - placeholders below.)*

| Dashboard | Market Analytics |
|---|---|
| `reports/figures/screenshot_dashboard.png` | `reports/figures/screenshot_analytics.png` |

| Price Prediction | What-If Simulator |
|---|---|
| `reports/figures/screenshot_prediction.png` | `reports/figures/screenshot_whatif.png` |

---

## Dataset

| | |
|---|---|
| **Name** | CAR DETAILS FROM CAR DEKHO |
| **Original source** | CarDekho listings, published on Kaggle by nehalbirla: [kaggle.com/datasets/nehalbirla/car-details-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/car-details-from-cardekho) |
| **Mirror used by this project** | [raw.githubusercontent.com/chandanverma07/DataSets/master/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv](https://raw.githubusercontent.com/chandanverma07/DataSets/master/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv) (identical content to the Kaggle original, verified against five independent mirrors) |
| **Format** | CSV |
| **Exact filename expected by the code** | `CAR DETAILS FROM CAR DEKHO.csv` |
| **Raw size** | 4,340 rows x 8 columns |
| **Raw columns (exact, case-sensitive)** | `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner` |
| **Target column** | `selling_price` (Indian Rupees) |
| **Listing years covered** | 1992-2020 |

### Download instructions

1. Open the mirror URL above (or the Kaggle page, which requires a free
   account).
2. Download the file. If it downloads with a different name, rename it to
   exactly `CAR DETAILS FROM CAR DEKHO.csv`.
3. Place it in `data/raw/` (that exact path - `data/raw/CAR DETAILS FROM CAR DEKHO.csv`).
4. **Do not modify the file.** All cleaning happens in code
   (`src/data_cleaning.py`) so the transformation is documented and repeatable.

```bash
curl -L -o "data/raw/CAR DETAILS FROM CAR DEKHO.csv" \
  "https://raw.githubusercontent.com/chandanverma07/DataSets/master/CAR%20DETAILS%20FROM%20CAR%20DEKHO.csv"
```

### Known data-quality issues (found by inspection, handled in `data_cleaning.py`)

- **763 exact duplicate rows** (same listing captured more than once) - dropped.
- **Two rows** have `name` glued together (`"OpelCorsa 1.6Gls"` instead of
  `"Opel Corsa 1.6Gls"`) - fixed before brand/model parsing.
- Zero missing values in any column of the raw file (verified, not assumed).

### Adapting the brief to the real dataset

The original brief imagined optional fields like Model, Engine specs,
Location and Condition. This real dataset does not contain engine specs or
a location column, so:

- **Model** is parsed out of `name` and shown in Market Analytics, but is
  **not** used as a prediction input - it's close to a unique identifier
  per row (thousands of distinct strings across ~3,500 rows), so one-hot
  encoding it would mostly memorize individual listings rather than
  generalize. See `src/preprocessing.py` for the full reasoning.
- **Engine specs / Location** simply don't exist in this data and were not
  invented.
- **Condition** is approximated with the real `owner` column (First /
  Second / Third / Fourth & Above Owner / Test Drive Car), labeled
  "Ownership / Condition Proxy" wherever it appears in the app.

---

## Tech Stack

Python 3.12 &middot; Pandas &middot; NumPy &middot; Matplotlib &middot;
Scikit-learn &middot; XGBoost &middot; Plotly &middot; Streamlit &middot;
Joblib &middot; Pytest &middot; Jupyter/nbformat

---

## Project Architecture & Workflow

```
Real Used-Car Dataset (CarDekho, via Kaggle)
        |
        v
   CSV File (data/raw/)
        |
        v
   Data Loading + Validation .......... src/data_loader.py
        |
        v
   Data Cleaning ....................... src/data_cleaning.py
   (dedupe, fix glitches, validate ranges)
        |
        v
   Exploratory Data Analysis ........... notebooks/01, 02, 03
        |
        v
   Feature Engineering .................. src/feature_engineering.py
   (brand/model parse, car_age, mileage_per_year)
        |
        v
   Train / Test Split (80/20, seed 42)
        |
        v
   Preprocessing Pipeline ............... src/preprocessing.py
   (impute -> scale numeric | impute -> one-hot categorical)
        |
        v
   Model Training ........................ src/train_model.py
   (Linear Regression, Random Forest, XGBoost - each on log1p(price))
        |
        v
   Model Evaluation ...................... MAE / MSE / RMSE / R^2
        |
        v
   Model Comparison + Best Model Selection (lowest test RMSE)
        |
        v
   Save Complete ML Pipeline ............. models/car_price_pipeline.pkl (joblib)
        |
        v
   Streamlit Dashboard .................... app/app.py
        |
        v
   User Input -> src/predict.py -> Saved Pipeline -> Price Prediction
        |
        v
   Analytics + Prediction + What-If Simulation Result
```

---

## Folder Structure

```
used-car-price-dashboard/
|
├── data/
│   ├── raw/
│   │   └── CAR DETAILS FROM CAR DEKHO.csv      # original, untouched
│   └── processed/
│       └── cleaned_car_data.csv                 # generated by train_model.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb                # executed, real outputs
│   ├── 02_data_cleaning.ipynb                    # executed, real outputs
│   ├── 03_eda.ipynb                              # executed, real charts
│   └── 04_model_training.ipynb                   # executed, real metrics
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # load + validate raw CSV/Excel
│   ├── data_cleaning.py       # dedupe, fix glitches, validate ranges
│   ├── feature_engineering.py # brand/model, car_age, mileage_per_year
│   ├── preprocessing.py       # shared ColumnTransformer + path constants
│   ├── train_model.py         # trains & saves the winning pipeline
│   ├── evaluate_model.py      # standalone re-evaluation of the saved model
│   └── predict.py             # single predictions + What-If sensitivity sweeps
│
├── models/
│   └── car_price_pipeline.pkl # generated by train_model.py (joblib)
│
├── app/
│   └── app.py                 # 6-page Streamlit dashboard
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_prediction.py
│
├── reports/
│   ├── metrics.json           # generated by train_model.py - real metrics
│   ├── test_predictions.csv   # generated by train_model.py
│   └── figures/                # generated by evaluate_model.py / notebooks
│
├── requirements.txt
├── requirements-dev.txt        # optional, only for re-running notebooks
├── .gitignore
├── pytest.ini
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone / create the project folder and enter it

```bash
cd used-car-price-dashboard
```

### 2. Create a virtual environment

**Windows**
```bat
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -c "import numpy, pandas, sklearn, streamlit, joblib, xgboost, plotly; print('All core packages import OK')"
```

### 5. Download the dataset

See [Dataset](#dataset) above. Place the CSV at exactly:
`data/raw/CAR DETAILS FROM CAR DEKHO.csv`

---

## Usage

### Train the model

```bash
python src/train_model.py
```

This loads the raw CSV, cleans it, engineers features, saves the processed
dataset, trains all three models, evaluates them, saves the winning pipeline
to `models/car_price_pipeline.pkl`, and writes `reports/metrics.json` +
`reports/test_predictions.csv`. Expect ~2-5 seconds on a laptop.

### Re-evaluate the saved model at any time

```bash
python src/evaluate_model.py
```

Reloads `models/car_price_pipeline.pkl` from disk, re-scores it on the same
held-out test split, and saves an actual-vs-predicted chart to
`reports/figures/actual_vs_predicted.png`.

### Run the dashboard

```bash
streamlit run app/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

### Make a single prediction from the command line

```bash
python src/predict.py
```

### Run the test suite

```bash
pytest
```

---

## Model Information & Results

**Features used by the model**

| Type | Features |
|---|---|
| Numeric | `car_age` (derived from year), `km_driven`, `mileage_per_year` (derived) |
| Categorical | `brand` (derived from name), `fuel`, `seller_type`, `transmission`, `owner` |
| Target | `selling_price`, modeled as `log1p(selling_price)` (see below) |

**Why log-transform the target:** `selling_price` ranges from Rs. 20,000 to
Rs. 89,00,000 and is strongly right-skewed. Every model here is wrapped in a
`TransformedTargetRegressor` that fits on `log1p(price)` and automatically
un-transforms predictions back to rupees - this keeps a handful of very
expensive cars from dominating the loss function.

**Results (test set, 716 rows, `random_state=42`) - generated after running `python src/train_model.py`:**

| Model | MAE (Rs.) | RMSE (Rs.) | R&sup2; | Fit time |
|---|---:|---:|---:|---:|
| **Linear Regression** (best) | 143,968 | 304,177 | **0.7128** | 0.04s |
| XGBoost | 143,329 | 335,400 | 0.6508 | 0.12s |
| Random Forest | 149,502 | 370,859 | 0.5730 | 2.19s |

**Why did Linear Regression win?** Random Forest and XGBoost were tuned via
`RandomizedSearchCV` (5-fold CV on log-transformed price) specifically to
control overfitting - an out-of-the-box, unregularized Random Forest reached
**0.94 R&sup2; on the training set but only 0.61 on the test set**, a
textbook overfitting gap given only ~2,861 training rows. Even regularized,
the tree models stayed more sensitive than Linear Regression to the handful
of very expensive outlier cars in the (small) held-out test set - R&sup2; and
RMSE computed on the raw rupee scale are dominated by squared error on those
few outliers. Regressing on `log1p(price)` mostly linearizes the relationship
between price and car_age / km_driven / brand tier, which plays to Linear
Regression's strengths on a comparatively small, low-dimensional feature set.
This is a genuinely useful, explainable result rather than a bug - see
[Portfolio / Interview Guidance](#portfolio--interview-guidance).

**Top Random Forest feature importances** (for reference - not the deployed
model, but informative about what drives price):

1. `car_age` (0.50)
2. `fuel = Diesel` (0.10)
3. `transmission = Automatic` / `Manual` (0.13 combined)
4. `km_driven` (0.06)
5. `mileage_per_year` (0.05)

---

## Testing

```bash
pytest -v
```

| File | Covers |
|---|---|
| `tests/test_data.py` | Raw data loading, required-column validation, cleaning pipeline (dedup, glitch fixes, range validation) on both the real dataset and a synthetic dirty sample |
| `tests/test_model.py` | Saved pipeline loads and predicts, `reports/metrics.json` shape, best-model selection logic, test-predictions file consistency |
| `tests/test_prediction.py` | Input validation, single predictions, two-word brands, unseen brands, error handling, and the What-If `predict_sensitivity()` helper (including monotonicity checks: price falls as mileage rises, rises as year increases) |

35 tests, all passing against the real trained pipeline at time of writing.
`test_model.py` and `test_prediction.py` auto-skip with a clear message if
`models/car_price_pipeline.pkl` doesn't exist yet (i.e. before you've run
`train_model.py`), rather than failing confusingly.

---

## Git / GitHub

```bash
git init
git add .
git commit -m "Initial project setup"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

`.gitignore` excludes the virtual environment, caches, and every
**generated** artifact (processed CSV, trained model, metrics, figures) -
these regenerate from `python src/train_model.py` and don't need to be
versioned. The raw dataset is also excluded (see the note inside
`.gitignore`) since it's third-party data with its own license on Kaggle;
push the code and let collaborators re-download it via the link above.

---

## Deployment (Streamlit Community Cloud)

```
Local Project
      |
      v
   GitHub (push the repo - see above)
      |
      v
Streamlit Community Cloud (share.streamlit.io)
      |
      v
   New app -> pick your repo/branch -> main file path: app/app.py -> Deploy
```

**Before deploying:** since `data/raw/`, `data/processed/`, and `models/`
are git-ignored, either (a) temporarily commit the trained model +
processed dataset so the deployed app has something to load, or (b) add a
`packages.txt`/startup step that runs `python src/train_model.py` once on
first boot. Option (a) is simpler for a portfolio demo.

**Common deployment issues**

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on deploy | A package used locally isn't in `requirements.txt` | Double-check `pip freeze` against `requirements.txt` |
| App loads but every page shows "no processed dataset" | `data/processed/cleaned_car_data.csv` wasn't included/generated on the deployed instance | Commit the generated artifacts, or add a startup training step |
| App loads but Price Prediction fails | `models/car_price_pipeline.pkl` missing on the deployed instance | Same as above |
| Charts don't render | `plotly` missing from `requirements.txt` | Add it - it's already listed here |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `DataLoadError: Dataset not found` | Download the CSV and place it at exactly `data/raw/CAR DETAILS FROM CAR DEKHO.csv` (see [Dataset](#dataset)) |
| `PredictionError: No trained model found` | Run `python src/train_model.py` first |
| Streamlit page shows "no processed dataset" | Same - run `python src/train_model.py` |
| `pip install` fails on a specific package | Confirm you activated the virtual environment; try upgrading pip: `pip install --upgrade pip` |
| Predictions seem off for a very new/old car | The Price Prediction and What-If pages intentionally cap Manufacturing Year to 1992-2020, matching the training data's actual range, to avoid extrapolating |
| Tests are skipped | `test_model.py`/`test_prediction.py` skip automatically until `models/car_price_pipeline.pkl` exists - run training first |

---

## Future Improvements

- Add a location/region feature if a richer dataset becomes available (this
  dataset has none).
- Try a stacked/blended model combining Linear Regression's stability with
  tree-based nonlinearity.
- Add SHAP-based explanations to the Price Prediction page (per-prediction
  "why this number").
- Persist What-If scenarios so a user can compare more than two versions of
  a car side by side.
- Swap `RandomizedSearchCV` for a slightly larger, time-boxed search once
  more training data is available (the current dataset's small size is the
  main limiter on tree-model performance - see Model Performance page).

---

## Portfolio / Interview Guidance

Talking points this project is built to support:

- **"Walk me through your pipeline."** Raw CSV -> validated load -> documented
  cleaning (name a real bug you found and fixed: the "OpelCorsa" glued
  string) -> engineered features with an explicit train/serve-consistency
  guarantee (`feature_engineering.engineer_features()` is the *only* place
  either training or prediction computes derived columns) -> a single saved
  `Pipeline` object (preprocessing + model) so there's no way for
  training-time and serving-time preprocessing to drift apart.
- **"Why didn't the fancier model win?"** See the Model Performance
  explanation above - use it to demonstrate you understand overfitting,
  target skew, and outlier sensitivity rather than just reporting whichever
  number is highest.
- **"How did you avoid overfitting?"** `RandomizedSearchCV` with 5-fold CV,
  and a concrete before/after number (0.94 train R&sup2; -> 0.61 test R&sup2;
  for an unregularized Random Forest) to show you actually caught it rather
  than asserting it.
- **"How do you handle a category the model never saw during training?"**
  `OneHotEncoder(handle_unknown="ignore")`, exercised in
  `tests/test_prediction.py::test_predict_price_handles_unseen_brand_gracefully`.
- **"What's the most interesting feature engineering decision you made?"**
  Deliberately *excluding* `model` (the free-text remainder of the car name)
  from the ML features despite it being available, because of its
  near-unique cardinality - a concrete example of judgment over
  "throw everything at the model."

---

## Author

**Your Name Here**
Portfolio project - Used Car Price Analytics, Prediction & What-If Simulation System
Built with Python, Scikit-learn and Streamlit.

## License

See `LICENSE`. The code in this repository is provided under the MIT
License (see file for full text); the dataset itself is third-party data
governed by its own license on Kaggle.
## Local AI Buying Advisor

This project includes a local AI-powered Used Car Buying Advisor using **Ollama** and the open-weight **Qwen 2.5 3B** model.

### Features
- Explains the machine-learning used-car price estimate in plain language.
- Provides cautious buying considerations based on the vehicle details.
- Suggests practical inspection questions for used-car buyers.
- Runs locally through Ollama instead of requiring a paid cloud AI API.

### Setup
1. Install Ollama from https://ollama.com/
2. Download the model:

   ```bash
   ollama pull qwen2.5:3b
   
   3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
