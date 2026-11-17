"""Streamlit web demo for the Used Car Resale Value Estimation project.

Run:
    streamlit run app.py

If the models have not been trained yet, the app offers to train them from
``used_cars.csv`` (or from a CSV you upload).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from car_features import FEATURES

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
DATA_PATH = ROOT / "used_cars.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"

ACCENT = "#2E86DE"
MUTED = "#A5B1C2"

st.set_page_config(
    page_title="Used Car Resale Value Estimator",
    page_icon="🚗",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    return f"${x:,.0f}"


def chart(fig, **kwargs):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), **kwargs)
    st.plotly_chart(fig, width="stretch")


def train_with_status(csv_path: Path) -> None:
    import train  # imported lazily so the app starts fast

    with st.status("Training models - this takes under a minute ...", expanded=True) as status:
        try:
            train.run(csv_path, MODELS_DIR, PLOTS_DIR, log=status.write)
        except Exception as exc:  # surface any data problem to the user
            status.update(label="Training failed", state="error")
            st.exception(exc)
            st.stop()
        status.update(label="Training complete", state="complete")
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


@st.cache_data(show_spinner=False)
def load_metrics(_mtime: float) -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading models ...")
def load_models(_mtime: float, files: tuple) -> dict:
    return {name: joblib.load(MODELS_DIR / fname) for name, fname in files}


@st.cache_data(show_spinner=False)
def load_table(name: str, _mtime: float) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / name)


# --------------------------------------------------------------------------- #
# First-run setup screen
# --------------------------------------------------------------------------- #
if not METRICS_PATH.exists():
    st.title("🚗 Used Car Resale Value Estimator")
    st.info("The models haven't been trained yet. Train them once and the app will load instantly afterwards.")
    if DATA_PATH.exists():
        st.write(f"Found dataset: `{DATA_PATH.name}`")
        if st.button("Train models now", type="primary"):
            train_with_status(DATA_PATH)
    else:
        st.write(
            "Put `used_cars.csv` next to `app.py` and refresh, or upload it here. "
            "Expected columns: `brand, model, model_year, milage, fuel_type, engine, "
            "transmission, ext_col, int_col, accident, clean_title, price`."
        )
        uploaded = st.file_uploader("Upload used_cars.csv", type="csv")
        if uploaded is not None:
            DATA_PATH.write_bytes(uploaded.getvalue())
            train_with_status(DATA_PATH)
    st.stop()


mtime = METRICS_PATH.stat().st_mtime
metrics = load_metrics(mtime)
meta = metrics["meta"]
models = load_models(mtime, tuple(metrics["model_files"].items()))
comparison = pd.DataFrame(metrics["comparison"])
test_preds = load_table("test_predictions.csv", mtime)
listings = load_table("listings.csv", mtime)
best_model = metrics["best_model"]
model_order = comparison.sort_values("Validation R2 (5-fold CV)", ascending=False)["Algorithm"].tolist()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Model")
    chosen = st.selectbox(
        "Prediction model",
        model_order,
        index=model_order.index(best_model),
        format_func=lambda n: f"{n}  ★ best" if n == best_model else n,
        help="The best model is the one with the highest validation (cross-validated) R².",
    )
    row = comparison.set_index("Algorithm").loc[chosen]
    c1, c2 = st.columns(2)
    c1.metric("Test R²", f"{row['Test R2']:.3f}")
    c2.metric("Median error", f"{row['Test median % error']:.1f}%")
    c1.metric("Test MAE", money(row["Test MAE"]))
    c2.metric("Interpretability", row["Interpretability"])

    st.divider()
    r = metrics["rows"]
    st.caption(
        f"Trained on **{r['train']:,}** cars, tested on **{r['test']:,}** "
        f"({r['clean']:,} clean rows of {r['raw']:,}).  \n"
        f"Last trained: {metrics['trained_at'][:16].replace('T', ' ')} UTC"
    )
    if DATA_PATH.exists():
        with st.expander("Retrain models"):
            st.caption("Re-runs `train.py` on `used_cars.csv`.")
            if st.button("Retrain"):
                train_with_status(DATA_PATH)


st.title("🚗 Used Car Resale Value Estimator")
st.caption(
    "Estimate a used car's market value with five regression models - Linear Regression, "
    "CART, Pruned CART, Random Forest and Gradient Boosting. Prices are in US dollars."
)

tab_predict, tab_compare, tab_data, tab_about = st.tabs(
    ["Estimate a price", "Model comparison", "Explore the data", "About"]
)


# --------------------------------------------------------------------------- #
# Tab 1 - Price estimator
# --------------------------------------------------------------------------- #
def make_row(**values) -> pd.DataFrame:
    return pd.DataFrame([values])[FEATURES]


with tab_predict:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("Car details")
        brands = meta["brands"]
        default_brand = "BMW" if "BMW" in brands else brands[0]
        brand = st.selectbox("Brand", brands, index=brands.index(default_brand))
        model_names = meta["models_by_brand"][brand]
        model_name = st.selectbox("Model", model_names)

        d = meta["model_defaults"].get(f"{brand}|||{model_name}", {})
        rng = meta["ranges"]
        k = f"{brand}|{model_name}"  # widget keys tied to the car, so defaults refresh on change

        def default(field):
            v = d.get(field)
            return rng[field]["median"] if v is None or (isinstance(v, float) and np.isnan(v)) else v

        a, b = st.columns(2)
        year = a.number_input(
            "Model year",
            min_value=int(rng["model_year"]["min"]),
            max_value=int(rng["model_year"]["max"]),
            value=int(d.get("model_year", rng["model_year"]["median"])),
            step=1,
            key=f"year_{k}",
        )
        mileage = b.number_input(
            "Mileage (miles)",
            min_value=0,
            max_value=int(max(rng["mileage"]["max"], 300_000)),
            value=int(round(rng["mileage"]["median"], -3)),
            step=1_000,
            key=f"mileage_{k}",
        )

        fuel_types = meta["fuel_types"]
        fuel = a.selectbox(
            "Fuel type",
            fuel_types,
            index=fuel_types.index(d["fuel_type"]) if d.get("fuel_type") in fuel_types else 0,
            key=f"fuel_{k}",
        )
        transmissions = meta["transmissions"]
        transmission = b.selectbox(
            "Transmission",
            transmissions,
            index=transmissions.index(d["transmission"]) if d.get("transmission") in transmissions else 0,
            key=f"trans_{k}",
        )

        a, b, c = st.columns(3)
        horsepower = a.number_input(
            "HP", min_value=50, max_value=1200, value=int(default("horsepower")), step=1, key=f"hp_{k}"
        )
        liters = b.number_input(
            "Engine (L)", min_value=0.0, max_value=8.5, value=float(default("engine_liters")), step=0.1,
            format="%.1f", key=f"l_{k}",
        )
        cylinders = c.number_input(
            "Cylinders", min_value=0, max_value=16, value=int(default("cylinders")), step=1, key=f"cyl_{k}"
        )

        a, b = st.columns(2)
        ext_col = a.selectbox("Exterior colour", meta["ext_cols"], key=f"ext_{k}")
        int_col = b.selectbox("Interior colour", meta["int_cols"], key=f"int_{k}")
        accident = st.radio("Accident history", meta["accident"], horizontal=True, key=f"acc_{k}")
        st.caption("Engine specs and fuel/transmission are pre-filled with the typical values for this model.")

    car = dict(
        model_year=year,
        mileage=float(mileage),
        horsepower=float(horsepower),
        engine_liters=float(liters),
        cylinders=float(cylinders),
        brand=brand,
        model=model_name,
        fuel_type=fuel,
        transmission=transmission,
        ext_col=ext_col,
        int_col=int_col,
        accident=accident,
    )
    X_car = make_row(**car)
    estimates = {name: float(m.predict(X_car)[0]) for name, m in models.items()}
    price = estimates[chosen]
    medape = comparison.set_index("Algorithm").loc[chosen, "Test median % error"] / 100

    with right:
        st.subheader("Estimated resale value")
        st.metric(f"{chosen}", money(price))
        st.caption(
            f"Half of test-set predictions from this model were within ±{medape:.0%} of the real price, "
            f"so a reasonable range is **{money(price * (1 - medape))} - {money(price * (1 + medape))}**."
        )

        est_df = pd.DataFrame({"Model": list(estimates), "Estimate": list(estimates.values())})
        est_df = est_df.set_index("Model").loc[model_order].reset_index()
        fig = go.Figure(
            go.Bar(
                x=est_df["Estimate"],
                y=est_df["Model"],
                orientation="h",
                marker_color=[ACCENT if m == chosen else MUTED for m in est_df["Model"]],
                text=[money(v) for v in est_df["Estimate"]],
                textposition="auto",
                hovertemplate="%{y}: %{text}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
        chart(fig, title="What every model thinks", height=260, showlegend=False)

        # Sensitivity curves: how value changes with mileage and age
        mi_grid = np.linspace(0, max(rng["mileage"]["p99"], mileage * 1.2), 30)
        mi_rows = pd.concat([make_row(**{**car, "mileage": float(m)}) for m in mi_grid], ignore_index=True)
        yr_grid = np.arange(max(int(rng["model_year"]["min"]), year - 15), int(rng["model_year"]["max"]) + 1)
        yr_rows = pd.concat([make_row(**{**car, "model_year": int(y)}) for y in yr_grid], ignore_index=True)
        mi_pred = models[chosen].predict(mi_rows)
        yr_pred = models[chosen].predict(yr_rows)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(x=mi_grid, y=mi_pred, labels={"x": "Mileage", "y": "Estimated value"})
            fig.add_scatter(x=[mileage], y=[price], mode="markers", marker=dict(size=11, color=ACCENT),
                            name="This car", showlegend=False)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            fig.update_xaxes(tickformat=",.0f")
            chart(fig, title="Value vs mileage", height=280)
        with c2:
            fig = px.line(x=yr_grid, y=yr_pred, labels={"x": "Model year", "y": "Estimated value"})
            fig.add_scatter(x=[year], y=[price], mode="markers", marker=dict(size=11, color=ACCENT),
                            name="This car", showlegend=False)
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            chart(fig, title="Value vs model year", height=280)

    st.subheader("Comparable cars in the dataset")
    comps = listings[(listings["brand"] == brand) & (listings["model"] == model_name)]
    label = f"Same model ({brand} {model_name})"
    if len(comps) < 3:
        comps = listings[(listings["brand"] == brand) & ((listings["model_year"] - year).abs() <= 2)]
        label = f"{brand} cars from {year - 2}-{year + 2}"
    if comps.empty:
        st.write("No similar listings in the dataset.")
    else:
        comps = comps.assign(_d=(comps["model_year"] - year).abs() + (comps["mileage"] - mileage).abs() / 50_000)
        comps = comps.sort_values("_d").head(10)
        st.caption(f"{label} - {len(comps)} closest listings")
        st.dataframe(
            comps[["brand", "model", "model_year", "mileage", "horsepower", "transmission", "accident", "price"]],
            hide_index=True,
            width="stretch",
            column_config={
                "model_year": st.column_config.NumberColumn("Year", format="%d"),
                "mileage": st.column_config.NumberColumn("Mileage", format="%,d mi"),
                "horsepower": st.column_config.NumberColumn("HP", format="%d"),
                "price": st.column_config.NumberColumn("Listed price", format="$%,d"),
            },
        )


# --------------------------------------------------------------------------- #
# Tab 2 - Model comparison
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("Comparison table")
    st.caption(
        "Validation score = mean R² from 5-fold cross-validation on the training set. "
        "Test score = R² on the held-out 20% test set (same split as the notebook, random_state=42). "
        "The best model is picked on validation score, never on test score."
    )
    table = comparison.set_index("Algorithm").loc[model_order].reset_index()
    display = table.copy()
    display["Algorithm"] = [f"★ {n}" if n == best_model else n for n in display["Algorithm"]]
    st.dataframe(
        display[
            [
                "Algorithm",
                "Validation R2 (5-fold CV)",
                "Test R2",
                "Test MAE",
                "Test RMSE",
                "Training time (s)",
                "Inference time (ms / 1k cars)",
                "Interpretability",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "Validation R2 (5-fold CV)": st.column_config.NumberColumn("Validation score (CV R²)", format="%.4f"),
            "Test R2": st.column_config.NumberColumn("Test score (R²)", format="%.4f"),
            "Test MAE": st.column_config.NumberColumn("Test MAE", format="$%,.0f"),
            "Test RMSE": st.column_config.NumberColumn("Test RMSE", format="$%,.0f"),
            "Training time (s)": st.column_config.NumberColumn("Training time (s)", format="%.3f"),
            "Inference time (ms / 1k cars)": st.column_config.NumberColumn(
                "Inference time (ms / 1k cars)", format="%.2f"
            ),
        },
    )
    with st.expander("Interpretability notes"):
        for _, r in comparison.iterrows():
            st.markdown(f"**{r['Algorithm']}** - {r['Interpretability']}: {r['Interpretability notes']}")
    st.download_button(
        "Download comparison table (CSV)",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        long = table.melt(
            id_vars="Algorithm",
            value_vars=["Validation R2 (5-fold CV)", "Test R2"],
            var_name="Score",
            value_name="R²",
        )
        long["Score"] = long["Score"].map({"Validation R2 (5-fold CV)": "Validation (CV)", "Test R2": "Test"})
        fig = px.bar(long, x="Algorithm", y="R²", color="Score", barmode="group",
                     color_discrete_sequence=[MUTED, ACCENT])
        chart(fig, title="R² - higher is better", height=340, legend_title_text="")
    with c2:
        long = table.melt(id_vars="Algorithm", value_vars=["Test MAE", "Test RMSE"], var_name="Metric",
                          value_name="Error")
        fig = px.bar(long, x="Algorithm", y="Error", color="Metric", barmode="group",
                     color_discrete_sequence=[ACCENT, MUTED])
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        chart(fig, title="Test error - lower is better", height=340, legend_title_text="")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(table, x="Algorithm", y="Training time (s)", color_discrete_sequence=[ACCENT])
        chart(fig, title="Training time (seconds)", height=300)
    with c2:
        fig = px.bar(table, x="Algorithm", y="Inference time (ms / 1k cars)", color_discrete_sequence=[ACCENT])
        chart(fig, title="Inference time (ms per 1,000 cars)", height=300)

    st.divider()
    st.subheader("Notebook pipeline vs improved pipeline")
    st.caption(
        "Both use the same cleaned rows and the same train/test split. The notebook pipeline one-hot encodes "
        "mileage and the engine description as text; the improved pipeline turns mileage into a number, "
        "extracts horsepower, engine size and cylinders, groups rare categories and learns log(price)."
    )
    base = metrics["notebook_baseline"]
    vs_rows = []
    for name in model_order:
        if name in base:
            imp = comparison.set_index("Algorithm").loc[name]
            vs_rows.append(
                {
                    "Algorithm": name,
                    "Notebook R²": base[name]["R2"],
                    "Improved R²": imp["Test R2"],
                    "Notebook MAE": base[name]["MAE"],
                    "Improved MAE": imp["Test MAE"],
                    "Notebook RMSE": base[name]["RMSE"],
                    "Improved RMSE": imp["Test RMSE"],
                }
            )
    vs = pd.DataFrame(vs_rows)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(
            vs,
            hide_index=True,
            width="stretch",
            column_config={
                "Notebook R²": st.column_config.NumberColumn(format="%.4f"),
                "Improved R²": st.column_config.NumberColumn(format="%.4f"),
                "Notebook MAE": st.column_config.NumberColumn(format="$%,.0f"),
                "Improved MAE": st.column_config.NumberColumn(format="$%,.0f"),
                "Notebook RMSE": st.column_config.NumberColumn(format="$%,.0f"),
                "Improved RMSE": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )
    with c2:
        long = vs.melt(id_vars="Algorithm", value_vars=["Notebook R²", "Improved R²"], var_name="Pipeline",
                       value_name="Test R²")
        fig = px.bar(long, x="Algorithm", y="Test R²", color="Pipeline", barmode="group",
                     color_discrete_sequence=[MUTED, ACCENT])
        chart(fig, height=300, legend_title_text="")

    st.divider()
    st.subheader("Actual vs predicted (test set)")
    scatter_model = st.selectbox("Show model", model_order, index=model_order.index(chosen), key="scatter_model")
    fig = px.scatter(
        test_preds,
        x="actual",
        y=scatter_model,
        hover_data={"brand": True, "model": True, "model_year": True, "mileage": ":,.0f"},
        labels={"actual": "Actual price", scatter_model: "Predicted price"},
        log_x=True,
        log_y=True,
        opacity=0.55,
        color_discrete_sequence=[ACCENT],
    )
    lo, hi = float(test_preds["actual"].min()), float(test_preds["actual"].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"),
                             name="Perfect prediction"))
    fig.update_xaxes(tickprefix="$")
    fig.update_yaxes(tickprefix="$")
    chart(fig, height=460, title="Points on the dashed line are perfect predictions (log scale)")

    st.subheader("What drives the price?")
    exp = metrics["explanations"].get(scatter_model, {})
    imp = exp.get("values") if exp.get("kind") == "importance" else exp.get("importance")
    if imp:
        imp_s = pd.Series(imp).sort_values()
        fig = px.bar(x=imp_s.values, y=imp_s.index, orientation="h", color_discrete_sequence=[ACCENT],
                     labels={"x": "Share of importance", "y": ""})
        fig.update_xaxes(tickformat=".0%")
        title = "Feature importance" if exp.get("kind") == "importance" else "Relative size of coefficients"
        chart(fig, title=f"{scatter_model}: {title}", height=380)
    if exp.get("kind") == "coefficients":
        coefs = pd.Series(exp["values"])
        st.caption("Largest Linear Regression coefficients, shown as the % change in price they cause "
                   "(numeric features: per 1 standard deviation; categories: vs. the baseline category).")
        st.dataframe(
            pd.DataFrame({"Feature": coefs.index, "Effect on price": (np.exp(coefs.values) - 1) * 100}),
            hide_index=True,
            width="stretch",
            column_config={"Effect on price": st.column_config.NumberColumn(format="%+.1f%%")},
        )
    if scatter_model in ("CART Regression", "Pruned CART"):
        rules_path = MODELS_DIR / "cart_rules.txt"
        if rules_path.exists():
            with st.expander("CART Regression decision rules (first 4 levels)"):
                st.code(rules_path.read_text(encoding="utf-8"), language="text")
    if scatter_model == "Pruned CART" and (MODELS_DIR / "pruning_path.csv").exists():
        pp = load_table("pruning_path.csv", mtime)
        fig = px.line(pp, x="alpha", y="val_rmse_log", markers=True, log_x=True,
                      labels={"alpha": "ccp_alpha", "val_rmse_log": "Validation RMSE (log price)"},
                      hover_data=["leaves", "depth"])
        fig.add_vline(x=metrics["best_ccp_alpha"], line_dash="dash", line_color=ACCENT)
        chart(fig, title="Cost-complexity pruning: choosing alpha on the validation set", height=320)


# --------------------------------------------------------------------------- #
# Tab 3 - Data explorer
# --------------------------------------------------------------------------- #
with tab_data:
    brands_pick = st.multiselect("Filter by brand", meta["brands"], placeholder="All brands")
    df = listings[listings["brand"].isin(brands_pick)] if brands_pick else listings

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings", f"{len(df):,}")
    c2.metric("Brands", df["brand"].nunique())
    c3.metric("Median price", money(df["price"].median()) if len(df) else "-")
    c4.metric("Median mileage", f"{df['mileage'].median():,.0f} mi" if len(df) else "-")

    if len(df):
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="price", nbins=60, color_discrete_sequence=[ACCENT],
                               range_x=[0, df["price"].quantile(0.99)])
            fig.update_xaxes(tickprefix="$", tickformat=",.0f", title="Price (top 1% cut off)")
            chart(fig, title="Price distribution", height=340)
        with c2:
            top = df.groupby("brand")["price"].agg(["median", "count"]).query("count >= 5")
            top = top.sort_values("median", ascending=False).head(20).reset_index()
            fig = px.bar(top, x="median", y="brand", orientation="h", hover_data=["count"],
                         color_discrete_sequence=[ACCENT], labels={"median": "Median price", "brand": ""})
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            chart(fig, title="Median price by brand (brands with 5+ listings)", height=340)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df, x="mileage", y="price", color="accident", opacity=0.5, log_y=True,
                             hover_data=["brand", "model", "model_year"])
            fig.update_yaxes(tickprefix="$")
            fig.update_xaxes(tickformat=",.0f")
            chart(fig, title="Price vs mileage", height=360, legend=dict(orientation="h", y=-0.25))
        with c2:
            fig = px.box(df, x="model_year", y="price", log_y=True, color_discrete_sequence=[ACCENT])
            fig.update_yaxes(tickprefix="$")
            chart(fig, title="Price by model year", height=360)

        with st.expander("Show data"):
            st.dataframe(df, hide_index=True, width="stretch")


# --------------------------------------------------------------------------- #
# Tab 4 - About
# --------------------------------------------------------------------------- #
with tab_about:
    r = metrics["rows"]
    st.markdown(
        f"""
### Used Car Resale Value Estimation

**Goal:** predict the resale price of a used car from its brand, model, age, mileage, engine,
transmission, colours and accident history.

**Data:** `{metrics['data_file']}` - {r['raw']:,} listings, {r['clean']:,} after cleaning
(duplicates and rows with missing values removed, prices parsed from text).
Split 80/20 into {r['train']:,} training and {r['test']:,} test cars (`random_state=42`).

**Models:** Linear Regression, CART regression tree (depth 8), cost-complexity-pruned CART,
Random Forest (100 trees) and Gradient Boosting (100 trees) - same hyper-parameters as the notebook.

**How this app differs from the notebook (`Unit_05.ipynb`)**

| Step | Notebook | This app |
|---|---|---|
| Mileage | one-hot encoded as text (`"51,000 mi."`) | numeric miles |
| Engine | one-hot encoded full description | horsepower, litres, cylinders extracted |
| Rare categories | one column each | grouped when seen < 5 times |
| Target | price in $ | log(price), converted back to $ |
| Model selection | best test R² | best 5-fold cross-validated R² |

The notebook's original numbers are reproduced in the *Model comparison* tab for reference.

**Limitations:** the dataset is small (a few thousand US listings), so rare or exotic models are
estimated from brand-level patterns. A handful of very expensive cars strongly affect R² and RMSE,
which is why the median % error is also shown.

**Reproduce:** `pip install -r requirements.txt` → `python train.py` → `streamlit run app.py`
"""
    )
    st.caption(f"scikit-learn {metrics['sklearn_version']} · best model: {best_model}")
