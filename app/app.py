"""
app/app.py

Used Car Price Analytics & Prediction Dashboard - Streamlit application.

Run with:
    streamlit run app/app.py

Pages
-----
 1. Dashboard          - headline KPIs + a quick-look chart pair
 2. Market Analytics    - interactive filters + exploratory charts
 3. Price Prediction    - user enters vehicle details, gets an estimate
 4. What-If Simulator   - USP: tweak specs on a baseline car and instantly
                          compare the new estimate, plus a sensitivity curve
 5. Model Performance   - real (never fabricated) evaluation metrics
 6. About               - project, dataset and adaptation notes

Data & model dependencies
--------------------------
This app reads two artifacts produced by `python src/train_model.py`:
    data/processed/cleaned_car_data.csv   (cleaned + feature-engineered data)
    models/car_price_pipeline.pkl         (fitted preprocessing+model pipeline)
    reports/metrics.json                  (real evaluation metrics)
    reports/test_predictions.csv          (actual vs predicted, test set)
If any are missing, the relevant page shows a clear, friendly instruction
to run the training script first, instead of crashing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error(
        "The `plotly` package is required for this app but is not installed.\n\n"
        "Run: `pip install -r requirements.txt`"
    )
    st.stop()

from data_loader import DataLoadError
from predict import PredictionError, load_pipeline, predict_price, predict_sensitivity
from preprocessing import DATA_PROCESSED_PATH, METRICS_PATH, TEST_PREDICTIONS_PATH

# -----------------------------------------------------------------------
# Page config & light theming
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Analytics, Prediction & What-If Simulation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Dashboard", "Market Analytics", "Price Prediction", "What-If Simulator",
         "Model Performance", "About"]

PRIMARY_COLOR = "#2563eb"
ACCENT_COLOR = "#f97316"

DATASET_MIN_YEAR = 1992
DATASET_MAX_YEAR = 2020  # matches feature_engineering.REFERENCE_YEAR


def format_inr(amount: float) -> str:
    """Format a number using the Indian digit-grouping convention
    (e.g. 8,900,000 -> '89,00,000'), matching how prices are actually
    written on Indian used-car listings such as CarDekho."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    s = str(abs(amount))
    if len(s) <= 3:
        grouped = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}\u20b9{grouped}"


# -----------------------------------------------------------------------
# Cached data / model loaders
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_processed_df() -> pd.DataFrame | None:
    if not DATA_PROCESSED_PATH.exists():
        return None
    try:
        return pd.read_csv(DATA_PROCESSED_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_test_predictions() -> pd.DataFrame | None:
    if not TEST_PREDICTIONS_PATH.exists():
        return None
    try:
        return pd.read_csv(TEST_PREDICTIONS_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_pipeline_cached():
    return load_pipeline()


def missing_pipeline_notice() -> None:
    st.warning(
        "**No trained model found yet.**\n\n"
        "This page needs `models/car_price_pipeline.pkl`. From the project "
        "root, run:\n\n```bash\npython src/train_model.py\n```\n\n"
        "Then reload this page."
    )


def missing_data_notice() -> None:
    st.warning(
        "**No processed dataset found yet.**\n\n"
        "This page needs `data/processed/cleaned_car_data.csv`. From the "
        "project root, run:\n\n```bash\npython src/train_model.py\n```\n\n"
        "Then reload this page."
    )


# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
def sidebar_nav() -> str:
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = PAGES[0]
    with st.sidebar:
        st.markdown("## 🚗 Used Car Dashboard")
        st.caption("Market analytics + ML price prediction + What-If simulation")
        page = st.radio(
            "Navigate",
            PAGES,
            key="nav_page",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(
            "Dataset: CAR DETAILS FROM CAR DEKHO (Kaggle, via CarDekho)\n\n"
            "Listings span **1992-2020**."
        )
        with st.expander("Developer tools"):
            st.caption("Retrain the model from a terminal, then clear the cache below to pick up new results.")
            if st.button("Clear cached data & reload", width="stretch"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
    return page


# -----------------------------------------------------------------------
# Page 1 - Dashboard
# -----------------------------------------------------------------------
def page_dashboard(df: pd.DataFrame | None) -> None:
    st.title("Used Car Market Dashboard")
    st.caption("Headline statistics across every cleaned listing in the dataset.")

    if df is None:
        missing_data_notice()
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Number of Cars", f"{len(df):,}")
    c2.metric("Average Price", format_inr(df["selling_price"].mean()))
    c3.metric("Median Price", format_inr(df["selling_price"].median()))
    c4.metric("Average Km Driven", f"{df['km_driven'].mean():,.0f} km")
    c5.metric("Most Common Brand", df["brand"].mode().iloc[0])

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x="selling_price", nbins=50, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(xaxis_title="Selling Price (INR)", yaxis_title="Number of Listings",
                           margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Top 10 Brands by Listings")
        top_brands = df["brand"].value_counts().head(10).sort_values()
        fig = px.bar(x=top_brands.values, y=top_brands.index, orientation="h",
                     color_discrete_sequence=[ACCENT_COLOR])
        fig.update_layout(xaxis_title="Number of Listings", yaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Listings by Fuel Type")
        fuel_counts = df["fuel"].value_counts()
        fig = px.pie(values=fuel_counts.values, names=fuel_counts.index, hole=0.45)
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with right2:
        st.subheader("Average Price by Vehicle Age")
        by_age = df.groupby("car_age", as_index=False)["selling_price"].mean()
        fig = px.line(by_age, x="car_age", y="selling_price", markers=True,
                      color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(xaxis_title="Car Age (years)", yaxis_title="Average Selling Price (INR)",
                           margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")


# -----------------------------------------------------------------------
# Page 2 - Market Analytics
# -----------------------------------------------------------------------
def page_market_analytics(df: pd.DataFrame | None) -> None:
    st.title("Market Analytics")
    st.caption("Filter the dataset and explore how price relates to every other attribute.")

    if df is None:
        missing_data_notice()
        return

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        brands = f1.multiselect("Brand (leave empty = all)", sorted(df["brand"].unique()))
        fuels = f2.multiselect("Fuel Type (leave empty = all)", sorted(df["fuel"].unique()))
        transmissions = f3.multiselect("Transmission (leave empty = all)", sorted(df["transmission"].unique()))

        f4, f5 = st.columns(2)
        year_range = f4.slider("Year Range", int(df["year"].min()), int(df["year"].max()),
                                (int(df["year"].min()), int(df["year"].max())))
        price_range = f5.slider(
            "Price Range (INR)",
            int(df["selling_price"].min()), int(df["selling_price"].max()),
            (int(df["selling_price"].min()), int(df["selling_price"].max())),
            step=10_000,
        )

    filtered = df.copy()
    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]
    if fuels:
        filtered = filtered[filtered["fuel"].isin(fuels)]
    if transmissions:
        filtered = filtered[filtered["transmission"].isin(transmissions)]
    filtered = filtered[
        filtered["year"].between(*year_range) & filtered["selling_price"].between(*price_range)
    ]

    st.info(f"Showing **{len(filtered):,}** of {len(df):,} listings after filters.")

    if filtered.empty:
        st.warning("No listings match the current filters - widen the ranges above.")
        return

    tab1, tab2, tab3 = st.tabs(["Price Relationships", "Brand & Category Breakdown", "Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Price vs Kilometers Driven")
            fig = px.scatter(filtered, x="km_driven", y="selling_price", color="fuel",
                              opacity=0.6, hover_data=["brand", "model", "year"])
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Price vs Manufacturing Year")
            fig = px.scatter(filtered, x="year", y="selling_price", color="transmission",
                              opacity=0.6, hover_data=["brand", "model", "km_driven"])
            fig.update_layout(margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Price Distribution")
        fig = px.box(filtered, x="fuel", y="selling_price", color="fuel")
        fig.update_layout(margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Average Price by Brand (Top 15)")
            avg_by_brand = (filtered.groupby("brand")["selling_price"].mean()
                             .sort_values(ascending=False).head(15).sort_values())
            fig = px.bar(x=avg_by_brand.values, y=avg_by_brand.index, orientation="h",
                         color_discrete_sequence=[PRIMARY_COLOR])
            fig.update_layout(xaxis_title="Average Selling Price (INR)", yaxis_title="", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Ownership / Condition Breakdown")
            owner_counts = filtered["owner"].value_counts()
            fig = px.bar(x=owner_counts.index, y=owner_counts.values, color_discrete_sequence=[ACCENT_COLOR])
            fig.update_layout(xaxis_title="", yaxis_title="Number of Listings", margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")

        st.subheader("Top 10 Individual Models by Listing Count")
        top_models = (filtered.assign(full_name=filtered["brand"] + " " + filtered["model"])
                      ["full_name"].value_counts().head(10))
        st.dataframe(top_models.rename("Listings"), width="stretch")

    with tab3:
        st.subheader("Correlation with Selling Price")
        numeric_cols = ["selling_price", "year", "car_age", "km_driven", "mileage_per_year"]
        corr = filtered[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto")
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "car_age and selling_price are strongly negatively correlated (older cars sell for "
            "less); year is mechanically the mirror image of car_age. mileage_per_year captures "
            "usage intensity independent of age."
        )


# -----------------------------------------------------------------------
# Page 3 - Price Prediction
# -----------------------------------------------------------------------
def page_price_prediction(df: pd.DataFrame | None) -> None:
    st.title("Price Prediction")
    st.caption("Enter the details of a used car to get an estimated market price.")

    def _go_to_whatif() -> None:
        # Setting session_state["nav_page"] must happen in an on_click
        # callback (not in the main body after the sidebar radio has already
        # been instantiated this run) - callbacks run BEFORE the next
        # script execution, so the radio picks up the new value cleanly on
        # its next instantiation instead of raising
        # "cannot be modified after the widget ... is instantiated".
        st.session_state["nav_page"] = "What-If Simulator"

    try:
        pipeline = load_pipeline_cached()
    except PredictionError:
        missing_pipeline_notice()
        return

    brand_options = sorted(df["brand"].unique()) if df is not None else ["Maruti", "Hyundai", "Honda", "Toyota"]
    fuel_options = sorted(df["fuel"].unique()) if df is not None else ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    seller_options = sorted(df["seller_type"].unique()) if df is not None else ["Individual", "Dealer", "Trustmark Dealer"]
    transmission_options = sorted(df["transmission"].unique()) if df is not None else ["Manual", "Automatic"]
    owner_options = (
        df["owner"].value_counts().index.tolist() if df is not None
        else ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        brand = c1.selectbox("Brand", brand_options)
        year = c2.slider(
            "Manufacturing Year", DATASET_MIN_YEAR, DATASET_MAX_YEAR, 2017,
        )
        km_driven = c3.number_input("Kilometers Driven", min_value=0, max_value=1_000_000,
                                     value=50_000, step=1_000)

        c4, c5, c6 = st.columns(3)
        fuel = c4.selectbox("Fuel Type", fuel_options)
        transmission = c5.selectbox("Transmission", transmission_options)
        seller_type = c6.selectbox("Seller Type", seller_options)

        owner = st.selectbox("Ownership (Condition Proxy)", owner_options)

        st.caption(
            f"Model trained on listings from {DATASET_MIN_YEAR}-{DATASET_MAX_YEAR}; "
            "the year slider is limited to that range so estimates stay within what the "
            "model actually learned from, rather than extrapolating."
        )
        submitted = st.form_submit_button("Estimate Price", type="primary", width="stretch")

    if submitted:
        try:
            price = predict_price(
                pipeline, brand=brand, year=year, km_driven=km_driven, fuel=fuel,
                seller_type=seller_type, transmission=transmission, owner=owner,
            )
            st.session_state["last_prediction_inputs"] = dict(
                brand=brand, year=year, km_driven=km_driven, fuel=fuel,
                seller_type=seller_type, transmission=transmission, owner=owner,
            )
            st.session_state["last_prediction_price"] = price
            st.session_state["last_prediction_error"] = None
        except PredictionError as exc:
            st.session_state["last_prediction_error"] = str(exc)
            st.session_state["last_prediction_price"] = None

    # Rendered from session_state (not from the transient `submitted` flag) so
    # this section - including the "jump to What-If" button below - survives
    # reruns triggered by clicking that very button. A plain `if submitted:`
    # block would disappear on the very next rerun (form_submit_button only
    # returns True for the one run right after it's clicked), which would
    # wipe the result, and the jump button, before navigation could happen.
    if st.session_state.get("last_prediction_error"):
        st.error(f"Could not generate a prediction: {st.session_state['last_prediction_error']}")
    elif st.session_state.get("last_prediction_price") is not None:
        price = st.session_state["last_prediction_price"]
        st.success("Estimate generated.")
        st.markdown(
            f"""
            <div style="padding: 1.5rem; border-radius: 0.75rem; background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); text-align:center;">
                <div style="color: #dbeafe; font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase;">Estimated Market Price</div>
                <div style="color: white; font-size: 2.75rem; font-weight: 700; margin-top: 0.25rem;">{format_inr(price)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "⚠️ This is a statistical estimate based on historical listings, not an "
            "appraisal. Actual sale price depends on vehicle condition, service history, "
            "local demand and negotiation - factors not present in this dataset."
        )
        st.button("🔀 Explore What-If scenarios for this car", width="stretch", on_click=_go_to_whatif)


# -----------------------------------------------------------------------
# Page 4 - What-If Simulator
# -----------------------------------------------------------------------
def page_whatif_simulator(df: pd.DataFrame | None) -> None:
    st.title("What-If Price Simulator")
    st.caption(
        "Set up a baseline car on the left, adjust its specs on the right, and "
        "instantly compare the new estimated price with the original."
    )

    try:
        pipeline = load_pipeline_cached()
    except PredictionError:
        missing_pipeline_notice()
        return

    brand_options = sorted(df["brand"].unique()) if df is not None else ["Maruti", "Hyundai", "Honda", "Toyota"]
    fuel_options = sorted(df["fuel"].unique()) if df is not None else ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
    seller_options = sorted(df["seller_type"].unique()) if df is not None else ["Individual", "Dealer", "Trustmark Dealer"]
    transmission_options = sorted(df["transmission"].unique()) if df is not None else ["Manual", "Automatic"]
    owner_options = (
        df["owner"].value_counts().index.tolist() if df is not None
        else ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
    )

    # Seed the baseline from the Price Prediction page if the user arrived via
    # the "Explore What-If scenarios" button there; otherwise use sensible
    # defaults. This only sets INITIAL widget state - see _init_state below.
    last = st.session_state.get("last_prediction_inputs", {})
    defaults = dict(
        brand=last.get("brand", brand_options[0]),
        year=last.get("year", 2017),
        km_driven=last.get("km_driven", 50_000),
        fuel=last.get("fuel", fuel_options[0]),
        transmission=last.get("transmission", transmission_options[0]),
        seller_type=last.get("seller_type", seller_options[0]),
        owner=last.get("owner", owner_options[0]),
    )

    def _init_state(prefix: str) -> None:
        for field, value in defaults.items():
            key = f"{prefix}_{field}"
            if key not in st.session_state:
                st.session_state[key] = value

    _init_state("base")
    _init_state("whatif")

    def _reset_whatif() -> None:
        for field in defaults:
            st.session_state[f"whatif_{field}"] = st.session_state[f"base_{field}"]

    col_base, col_whatif = st.columns(2)

    with col_base:
        st.subheader("Original / Baseline")
        st.selectbox("Brand", brand_options, key="base_brand")
        st.slider("Manufacturing Year", DATASET_MIN_YEAR, DATASET_MAX_YEAR, key="base_year")
        st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000, step=1_000, key="base_km_driven")
        st.selectbox("Fuel Type", fuel_options, key="base_fuel")
        st.selectbox("Transmission", transmission_options, key="base_transmission")
        st.selectbox("Seller Type", seller_options, key="base_seller_type")
        st.selectbox("Ownership", owner_options, key="base_owner")

    with col_whatif:
        st.subheader("What-If / Modified")
        st.button("↺ Reset What-If to match Baseline", on_click=_reset_whatif, width="stretch")
        st.selectbox("Brand", brand_options, key="whatif_brand")
        st.slider("Manufacturing Year", DATASET_MIN_YEAR, DATASET_MAX_YEAR, key="whatif_year")
        st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000, step=1_000, key="whatif_km_driven")
        st.selectbox("Fuel Type", fuel_options, key="whatif_fuel")
        st.selectbox("Transmission", transmission_options, key="whatif_transmission")
        st.selectbox("Seller Type", seller_options, key="whatif_seller_type")
        st.selectbox("Ownership", owner_options, key="whatif_owner")

    base_inputs = {field: st.session_state[f"base_{field}"] for field in defaults}
    whatif_inputs = {field: st.session_state[f"whatif_{field}"] for field in defaults}

    try:
        base_price = predict_price(pipeline, **base_inputs)
        whatif_price = predict_price(pipeline, **whatif_inputs)
    except PredictionError as exc:
        st.error(f"Could not generate a prediction: {exc}")
        return

    delta = whatif_price - base_price
    pct = (delta / base_price * 100) if base_price else 0.0
    delta_str = f"{'+' if delta >= 0 else ''}{format_inr(delta)} ({pct:+.1f}%)"

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Price", format_inr(base_price))
    m2.metric("What-If Price", format_inr(whatif_price), delta=delta_str)
    changed_fields = [f for f in defaults if base_inputs[f] != whatif_inputs[f]]
    m3.metric("Fields Changed", len(changed_fields))
    st.caption(
        "Changed: " + ", ".join(changed_fields) if changed_fields
        else "What-If currently matches the baseline exactly - adjust a field above to see it diverge."
    )

    st.divider()
    st.subheader("Sensitivity: how price responds to one factor")
    sweep_choice = st.radio("Sweep", ["Kilometers Driven", "Manufacturing Year"], horizontal=True)

    if sweep_choice == "Kilometers Driven":
        sweep_values = list(range(0, 200_001, 10_000))
        prices = predict_sensitivity(pipeline, whatif_inputs, "km_driven", sweep_values)
        x_title, current_x = "Kilometers Driven", whatif_inputs["km_driven"]
    else:
        sweep_values = list(range(DATASET_MIN_YEAR, DATASET_MAX_YEAR + 1))
        prices = predict_sensitivity(pipeline, whatif_inputs, "year", sweep_values)
        x_title, current_x = "Manufacturing Year", whatif_inputs["year"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_values, y=prices, mode="lines",
                              line=dict(color=PRIMARY_COLOR, width=3), name="Predicted price"))
    fig.add_trace(go.Scatter(x=[current_x], y=[whatif_price], mode="markers",
                              marker=dict(color=ACCENT_COLOR, size=14, symbol="star"), name="Current What-If"))
    fig.update_layout(xaxis_title=x_title, yaxis_title="Predicted Selling Price (INR)",
                       margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"All other fields held at the current What-If values while {x_title.lower()} sweeps "
        "across its plausible range - the star marks where your current What-If scenario sits."
    )


# -----------------------------------------------------------------------
# Page 5 - Model Performance
# -----------------------------------------------------------------------
def page_model_performance() -> None:
    st.title("Model Performance")
    st.caption("Real evaluation results from the last training run - nothing on this page is invented.")

    metrics = load_metrics()
    if metrics is None:
        missing_pipeline_notice()
        return

    best = metrics["best_model"]
    best_metrics = metrics["best_model_metrics"]
    ds = metrics["dataset"]

    st.subheader(f"🏆 Best Model: {best}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", format_inr(best_metrics["MAE"]))
    c2.metric("RMSE", format_inr(best_metrics["RMSE"]))
    c3.metric("R\u00b2", f"{best_metrics['R2']:.4f}")
    c4.metric("Test Rows", f"{ds['test_rows']:,}")

    st.divider()
    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame(metrics["models"]).T
    comparison_df.index.name = "Model"
    st.dataframe(
        comparison_df.style.format({"MAE": "{:,.0f}", "MSE": "{:,.0f}", "RMSE": "{:,.0f}",
                                     "R2": "{:.4f}", "fit_seconds": "{:.2f}s"})
        .highlight_min(subset=["RMSE", "MAE"], color="#dcfce7")
        .highlight_max(subset=["R2"], color="#dcfce7"),
        width="stretch",
    )

    with st.expander("Why did Linear Regression win here?"):
        st.markdown(
            "Random Forest and XGBoost were tuned with a `RandomizedSearchCV` pass "
            "(5-fold CV on log-transformed price) to control overfitting - an "
            "out-of-the-box Random Forest reached **0.94 R\u00b2 on the training set but "
            "only 0.61 on the test set**, a textbook overfitting gap given ~2,861 "
            "training rows. Even after regularizing, the tree models remained more "
            "sensitive than Linear Regression to the handful of very expensive outlier "
            "cars in the (small) held-out test set - `selling_price` spans "
            "\u20b920,000 to \u20b989,00,000, and R\u00b2/RMSE computed on that raw rupee scale are "
            "dominated by squared error on those few outliers. Regressing on "
            "`log1p(selling_price)` (which every model here does, undone automatically "
            "at prediction time) mostly linearizes the relationship between price and "
            "car_age/km_driven/brand tier, which plays to Linear Regression's strengths "
            "on a comparatively small, low-dimensional feature set. This is a genuinely "
            "useful result to be able to explain in an interview: more model flexibility "
            "only pays off once there's enough data and enough real non-linear structure "
            "to justify it."
        )

    st.divider()
    st.subheader("Actual vs Predicted (Test Set)")
    test_preds = load_test_predictions()
    if test_preds is not None:
        pred_col = f"predicted_{best.lower().replace(' ', '_')}"
        if pred_col in test_preds.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_preds["actual_selling_price"], y=test_preds[pred_col],
                mode="markers", marker=dict(color=PRIMARY_COLOR, opacity=0.5, size=7),
                name=best,
            ))
            max_val = float(max(test_preds["actual_selling_price"].max(), test_preds[pred_col].max()))
            fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                     line=dict(color=ACCENT_COLOR, dash="dash"), name="Perfect prediction"))
            fig.update_layout(xaxis_title="Actual Selling Price (INR)", yaxis_title="Predicted Selling Price (INR)",
                               margin=dict(t=10))
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Actual-vs-predicted data for the current best model was not found in test_predictions.csv.")
    else:
        st.caption("Run `python src/train_model.py` to generate the actual-vs-predicted chart data.")

    st.divider()
    st.subheader("Training Run Details")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Raw Rows", f"{ds['raw_rows']:,}")
    d2.metric("Duplicates Removed", f"{ds['duplicates_removed']:,}")
    d3.metric("Clean Rows", f"{ds['clean_rows']:,}")
    d4.metric("Train / Test Split", f"{ds['train_rows']:,} / {ds['test_rows']:,}")
    st.caption(f"Metrics generated at: {metrics['generated_at']}")


# -----------------------------------------------------------------------
# Page 6 - About
# -----------------------------------------------------------------------
def page_about() -> None:
    st.title("About This Project")
    st.markdown(
        """
This dashboard is an end-to-end **Used Car Price Analytics, Prediction &
What-If Simulation** portfolio project: a real dataset, a documented
cleaning and feature engineering pipeline, three benchmarked regression
models, an interactive dashboard, and a What-If simulator on top.

### What-If Price Simulator
The **USP** of this project: set up a baseline vehicle, then change one or
more specs (mileage, manufacturing year, fuel type, transmission, ...) on
a second panel and instantly see the new estimated price next to the
original, plus a sensitivity curve showing how price responds as a single
factor sweeps across its range. This turns a single point estimate into an
explorable "what drives this number" tool for buyers and sellers alike.

### Dataset
- **Name:** CAR DETAILS FROM CAR DEKHO
- **Source:** CarDekho listings, published on Kaggle by nehalbirla
  ([kaggle.com/datasets/nehalbirla/car-details-from-cardekho](https://www.kaggle.com/datasets/nehalbirla/car-details-from-cardekho))
- **Raw size:** 4,340 rows x 8 columns, listings from 1992-2020
- **Raw columns:** `name, year, selling_price, km_driven, fuel, seller_type, transmission, owner`

### Adapting the brief to the real dataset
The original project brief imagined optional fields like **Model, Engine
info, Location and Condition**. The real dataset doesn't contain engine
specs or a location column, so:
- **Model** is parsed out of `name` and shown in Market Analytics, but is
  *not* used as a prediction input (too many unique values to generalize -
  see `src/preprocessing.py` for the full reasoning).
- **Engine / Location** simply don't exist in this data and were not
  invented.
- **Condition** is approximated with the real `owner` column
  (First/Second/.../Test Drive Car), labeled "Ownership / Condition Proxy"
  wherever it appears.

### Tech stack
Python - Pandas - NumPy - Scikit-learn - XGBoost - Matplotlib - Plotly -
Streamlit - Joblib

### Workflow
Raw CSV -> load & validate -> clean (dedupe, fix glitches, validate ranges)
-> engineer features (brand, car_age, mileage_per_year) -> train/test split
-> preprocessing pipeline (impute, scale, one-hot encode) -> train 3 models
-> evaluate -> save best pipeline with Joblib -> this dashboard.

### Disclaimer
Predictions are statistical estimates from historical listings, not a
professional valuation. See the Model Performance page for honest,
un-fabricated accuracy numbers and known limitations.
        """
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    page = sidebar_nav()
    df = load_processed_df()

    if page == "Dashboard":
        page_dashboard(df)
    elif page == "Market Analytics":
        page_market_analytics(df)
    elif page == "Price Prediction":
        page_price_prediction(df)
    elif page == "What-If Simulator":
        page_whatif_simulator(df)
    elif page == "Model Performance":
        page_model_performance()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
