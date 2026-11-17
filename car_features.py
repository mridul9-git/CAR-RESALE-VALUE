"""Shared data-cleaning and feature-engineering code.

Used by both ``train.py`` (to build the training data) and ``app.py`` (to turn the
web form into a model-ready row), so the two can never drift apart.

Two pipelines are defined:

* **Notebook baseline** - an exact copy of ``Unit_05.ipynb``: every non-year column
  (including mileage and the engine description) is one-hot encoded as text.
* **Improved (Option B)** - mileage becomes a number, horsepower / engine size /
  cylinders are extracted from the engine description, rare categories are grouped
  and the models learn log(price) so a handful of supercars do not dominate.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ["selling_price", "Selling_Price", "price", "Price"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns the improved model uses
NUMERIC_FEATURES = ["model_year", "mileage", "horsepower", "engine_liters", "cylinders"]
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
    "ext_col",
    "int_col",
    "accident",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_HP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*HP", re.IGNORECASE)
_LITRE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:L|Liter|Litre)\b", re.IGNORECASE)
_CYL_WORD_RE = re.compile(r"(\d{1,2})\s*Cylinder", re.IGNORECASE)
_CYL_LAYOUT_RE = re.compile(r"\b[VIHW](\d{1,2})\b")


# --------------------------------------------------------------------------- #
# Loading / cleaning (mirrors notebook cells 4, 6 and the price fix in cell 9)
# --------------------------------------------------------------------------- #
def load_and_clean(csv_path) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load the CSV and apply exactly the notebook's cleaning steps.

    Returns raw feature frame ``X`` (original columns), numeric price ``y`` and the
    detected target column name. Row order is preserved so that
    ``train_test_split(random_state=42)`` reproduces the notebook's split.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna()
    df.columns = df.columns.str.strip()

    target_col = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"No price column found. Columns: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = parse_price(df[target_col])

    valid = y.notna() & (y > 0)
    return X.loc[valid].copy(), y.loc[valid].copy(), target_col


def parse_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------- #
# Feature engineering for the improved pipeline
# --------------------------------------------------------------------------- #
def parse_mileage(value) -> float:
    """'51,000 mi.' -> 51000.0"""
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else np.nan


def parse_engine(text) -> tuple[float, float, float]:
    """Extract (horsepower, litres, cylinders) from a free-text engine description."""
    if pd.isna(text):
        return np.nan, np.nan, np.nan
    s = str(text)
    hp = _HP_RE.search(s)
    litres = _LITRE_RE.search(s)
    cyl = _CYL_WORD_RE.search(s) or _CYL_LAYOUT_RE.search(s)
    return (
        float(hp.group(1)) if hp else np.nan,
        float(litres.group(1)) if litres else np.nan,
        float(cyl.group(1)) if cyl else np.nan,
    )


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw notebook columns -> improved feature frame (columns = FEATURES)."""
    out = pd.DataFrame(index=X.index)
    out["model_year"] = pd.to_numeric(X["model_year"], errors="coerce")
    out["mileage"] = X["milage"].map(parse_mileage)
    engine = X["engine"].map(parse_engine)
    out["horsepower"] = engine.map(lambda t: t[0])
    out["engine_liters"] = engine.map(lambda t: t[1])
    out["cylinders"] = engine.map(lambda t: t[2])
    for col in CATEGORICAL_FEATURES:
        out[col] = X[col].astype(str).str.strip()
    return out[FEATURES]


# --------------------------------------------------------------------------- #
# Preprocessors
# --------------------------------------------------------------------------- #
def notebook_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Exact replica of notebook cell 8 (dtype-based column split)."""
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def improved_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=5,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def log_target(estimator) -> TransformedTargetRegressor:
    """Train on log(price), predict back on the dollar scale."""
    return TransformedTargetRegressor(
        regressor=estimator, func=np.log1p, inverse_func=np.expm1
    )
