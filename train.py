"""Train every model for the web demo and save the artefacts the app needs.

Usage:
    python train.py                       # uses ./used_cars.csv
    python train.py --data path/to/file.csv

Outputs
    models/*.joblib              trained pipelines (improved / Option B)
    models/metrics.json          comparison table, notebook baseline, metadata
    models/comparison_table.csv  the "suggested comparison table" from the brief
    models/test_predictions.csv  actual vs predicted on the held-out test set
    models/listings.csv          cleaned dataset (for the explorer tab)
    models/cart_rules.txt        readable if/else rules of the CART tree
    plots/*.png                  static plots for the report / presentation
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from car_features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    engineer_features,
    improved_preprocessor,
    load_and_clean,
    log_target,
    notebook_preprocessor,
)

ROOT = Path(__file__).resolve().parent

MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "CART Regression": "cart.joblib",
    "Pruned CART": "pruned_cart.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

INTERPRETABILITY = {
    "Linear Regression": ("High", "One coefficient per feature; effect of each input is directly readable."),
    "CART Regression": ("High", "Single tree of depth 8 - every prediction is a short chain of if/else rules."),
    "Pruned CART": ("Medium", "Single tree, but cost-complexity pruning may keep many leaves."),
    "Random Forest": ("Low", "Average of 100 trees; only global feature importances are available."),
    "Gradient Boosting": ("Low", "100 sequential trees; explained through feature importances."),
}


# --------------------------------------------------------------------------- #
# Model definitions (hyper-parameters copied from the notebook)
# --------------------------------------------------------------------------- #
def base_estimators() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "CART Regression": DecisionTreeRegressor(
            max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def improved_pipeline(estimator):
    return log_target(Pipeline([("preprocessor", improved_preprocessor()), ("model", estimator)]))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ape = np.abs(y_pred - y_true) / y_true
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MedAPE": float(np.median(ape) * 100),
    }


# --------------------------------------------------------------------------- #
# Notebook baseline - exact replica of Unit_05.ipynb (for side-by-side reporting)
# --------------------------------------------------------------------------- #
def notebook_baseline(X_raw, y, log) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    results = {}
    for name, est in base_estimators().items():
        pipe = Pipeline([("preprocessor", notebook_preprocessor(X_raw)), ("model", est)])
        pipe.fit(X_train, y_train)
        results[name] = regression_metrics(y_test, pipe.predict(X_test))
        log(f"  [notebook] {name:<18} R2={results[name]['R2']:.4f}  MAE={results[name]['MAE']:,.0f}")
    return results


# --------------------------------------------------------------------------- #
# Pruned CART: choose ccp_alpha on a validation split carved out of the train set
# --------------------------------------------------------------------------- #
def choose_ccp_alpha(X_train, y_train, log) -> tuple[float, pd.DataFrame]:
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    pre = improved_preprocessor()
    Xf = pre.fit_transform(X_fit)
    Xv = pre.transform(X_val)
    path = DecisionTreeRegressor(random_state=RANDOM_STATE).cost_complexity_pruning_path(
        Xf, np.log1p(y_fit)
    )
    alphas = np.unique(path.ccp_alphas[:-1])
    if len(alphas) > 40:
        alphas = alphas[np.linspace(0, len(alphas) - 1, 40).astype(int)]

    rows = []
    for a in alphas:
        tree = DecisionTreeRegressor(ccp_alpha=a, random_state=RANDOM_STATE).fit(Xf, np.log1p(y_fit))
        pred_log = tree.predict(Xv)
        rows.append(
            {
                "alpha": float(a),
                "val_rmse_log": float(np.sqrt(mean_squared_error(np.log1p(y_val), pred_log))),
                "val_R2": float(r2_score(y_val, np.expm1(pred_log))),
                "leaves": int(tree.get_n_leaves()),
                "depth": int(tree.get_depth()),
            }
        )
    df = pd.DataFrame(rows)
    best = df.loc[df["val_rmse_log"].idxmin()]
    log(f"  best ccp_alpha={best['alpha']:.5f} (leaves={int(best['leaves'])}, depth={int(best['depth'])})")
    return float(best["alpha"]), df


# --------------------------------------------------------------------------- #
# Interpretability helpers
# --------------------------------------------------------------------------- #
def aggregate_importances(feature_names, values) -> dict:
    """Sum one-hot columns back to their original feature."""
    totals = {f: 0.0 for f in FEATURES}
    for name, v in zip(feature_names, values):
        if name.startswith("num__"):
            totals[name[5:]] += float(v)
            continue
        for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if name.startswith(f"cat__{col}_"):
                totals[col] += float(v)
                break
    s = sum(totals.values()) or 1.0
    return dict(sorted(((k, v / s) for k, v in totals.items()), key=lambda kv: -kv[1]))


def explain_model(fitted) -> dict:
    inner = fitted.regressor_
    pre = inner.named_steps["preprocessor"]
    est = inner.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        return {"kind": "importance", "values": aggregate_importances(names, est.feature_importances_)}
    if hasattr(est, "coef_"):
        # Coefficients are on the log-price scale: exp(coef)-1 = % change in price.
        coefs = pd.Series(est.coef_, index=names)
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(20)
        return {
            "kind": "coefficients",
            "values": {k: float(v) for k, v in top.items()},
            "importance": aggregate_importances(names, np.abs(est.coef_)),
        }
    return {}


# --------------------------------------------------------------------------- #
# Metadata for the web form
# --------------------------------------------------------------------------- #
def _num_or_none(s: pd.Series):
    s = s.dropna()
    return None if s.empty else float(s.median())


def build_meta(F: pd.DataFrame, y: pd.Series) -> dict:
    def ordered(col, top=None):
        vc = F[col].value_counts()
        return vc.index.tolist()[:top] if top else vc.index.tolist()

    defaults = {}
    for (brand, model), g in F.groupby(["brand", "model"]):
        defaults[f"{brand}|||{model}"] = {
            "horsepower": _num_or_none(g["horsepower"]),
            "engine_liters": _num_or_none(g["engine_liters"]),
            "cylinders": _num_or_none(g["cylinders"]),
            "fuel_type": g["fuel_type"].mode().iat[0],
            "transmission": g["transmission"].mode().iat[0],
            "model_year": int(g["model_year"].median()),
        }

    def rng(col):
        s = F[col].dropna()
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "p99": float(s.quantile(0.99)),
        }

    return {
        "brands": sorted(F["brand"].unique().tolist()),
        "models_by_brand": {b: sorted(g["model"].unique().tolist()) for b, g in F.groupby("brand")},
        "model_defaults": defaults,
        "fuel_types": ordered("fuel_type"),
        "transmissions": ordered("transmission"),
        "ext_cols": ordered("ext_col", 40),
        "int_cols": ordered("int_col", 30),
        "accident": ordered("accident"),
        "ranges": {c: rng(c) for c in ["model_year", "mileage", "horsepower", "engine_liters", "cylinders"]},
        "price": {
            "min": float(y.min()),
            "max": float(y.max()),
            "median": float(y.median()),
            "mean": float(y.mean()),
        },
    }


# --------------------------------------------------------------------------- #
# Static plots for the report
# --------------------------------------------------------------------------- #
def save_plots(plots_dir: Path, table: pd.DataFrame, baseline: dict, preds: pd.DataFrame,
               best: str, explanations: dict, log) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not installed - skipping static plots")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    names = table["Algorithm"].tolist()

    # 1. Validation vs test R2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, table["Validation R2 (5-fold CV)"], 0.4, label="Validation R² (5-fold CV)", color="#4C78A8")
    ax.bar(x + 0.2, table["Test R2"], 0.4, label="Test R²", color="#F58518")
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylabel("R²")
    ax.set_title("Model comparison - R²")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "model_comparison_r2.png", dpi=150)
    plt.close(fig)

    # 2. Notebook baseline vs improved
    common = [n for n in names if n in baseline]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(common))
    ax.bar(x - 0.2, [baseline[n]["R2"] for n in common], 0.4, label="Notebook pipeline", color="#BAB0AC")
    ax.bar(x + 0.2, table.set_index("Algorithm").loc[common, "Test R2"], 0.4,
           label="Improved pipeline", color="#54A24B")
    ax.set_xticks(x, common, rotation=15)
    ax.set_ylabel("Test R²")
    ax.set_title("Notebook pipeline vs improved pipeline (same test set)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_vs_improved.png", dpi=150)
    plt.close(fig)

    # 3. Actual vs predicted for the best model
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preds["actual"], preds[best], alpha=0.4, s=12)
    lo, hi = preds["actual"].min(), preds["actual"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual price ($)")
    ax.set_ylabel("Predicted price ($)")
    ax.set_title(f"{best}: actual vs predicted (test set)")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted_best.png", dpi=150)
    plt.close(fig)

    # 4. Feature importance of the tree ensembles
    for model_name in ("Gradient Boosting", "Random Forest"):
        exp = explanations.get(model_name, {})
        if exp.get("kind") == "importance":
            imp = pd.Series(exp["values"]).sort_values()
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(imp.index, imp.values, color="#4C78A8")
            ax.set_xlabel("Share of importance")
            ax.set_title(f"{model_name}: feature importance")
            fig.tight_layout()
            fig.savefig(plots_dir / f"feature_importance_{model_name.lower().replace(' ', '_')}.png", dpi=150)
            plt.close(fig)
    log(f"  plots saved to {plots_dir}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(data_path=ROOT / "used_cars.csv", out_dir=ROOT / "models", plots_dir=ROOT / "plots", log=print) -> dict:
    data_path, out_dir, plots_dir = Path(data_path), Path(out_dir), Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading {data_path} ...")
    raw_rows = len(pd.read_csv(data_path))
    X_raw, y, target_col = load_and_clean(data_path)
    log(f"  {raw_rows} raw rows -> {len(X_raw)} after notebook cleaning (target column: '{target_col}')")

    log("Reproducing the notebook pipeline (baseline) ...")
    baseline = notebook_baseline(X_raw, y, log)

    log("Engineering features (numeric mileage, horsepower, engine size, cylinders) ...")
    F = engineer_features(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        F, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log(f"  train={len(X_train)}  test={len(X_test)}")

    log("Selecting pruning strength for Pruned CART ...")
    best_alpha, pruning_df = choose_ccp_alpha(X_train, y_train, log)

    base = base_estimators()
    estimators = {
        "Linear Regression": base["Linear Regression"],
        "CART Regression": base["CART Regression"],
        "Pruned CART": DecisionTreeRegressor(ccp_alpha=best_alpha, random_state=RANDOM_STATE),
        "Random Forest": base["Random Forest"],
        "Gradient Boosting": base["Gradient Boosting"],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, explanations = [], {}
    preds = pd.DataFrame(
        {
            "actual": y_test.values,
            "brand": X_test["brand"].values,
            "model": X_test["model"].values,
            "model_year": X_test["model_year"].values,
            "mileage": X_test["mileage"].values,
        }
    )

    log("Training improved models ...")
    for name, est in estimators.items():
        cv_scores = cross_val_score(improved_pipeline(est), X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        model = improved_pipeline(est)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        timings = []
        for _ in range(5):
            t0 = time.perf_counter()
            y_pred = model.predict(X_test)
            timings.append(time.perf_counter() - t0)
        infer_ms_per_1k = float(np.median(timings) / len(X_test) * 1000 * 1000)

        m = regression_metrics(y_test, y_pred)
        preds[name] = y_pred
        explanations[name] = explain_model(model)
        joblib.dump(model, out_dir / MODEL_FILES[name], compress=3)

        level, why = INTERPRETABILITY[name]
        tree = model.regressor_.named_steps["model"]
        if isinstance(tree, DecisionTreeRegressor):
            why = f"{why} ({tree.get_n_leaves()} leaves, depth {tree.get_depth()})"
            if tree.get_n_leaves() > 200:
                level = "Low-Medium"

        rows.append(
            {
                "Algorithm": name,
                "Validation R2 (5-fold CV)": float(cv_scores.mean()),
                "Validation R2 std": float(cv_scores.std()),
                "Test R2": m["R2"],
                "Test MAE": m["MAE"],
                "Test RMSE": m["RMSE"],
                "Test median % error": m["MedAPE"],
                "Training time (s)": train_time,
                "Inference time (ms / 1k cars)": infer_ms_per_1k,
                "Interpretability": level,
                "Interpretability notes": why,
            }
        )
        log(
            f"  {name:<18} CV R2={cv_scores.mean():.4f}  test R2={m['R2']:.4f}  "
            f"MAE=${m['MAE']:,.0f}  train={train_time:.2f}s"
        )

    table = pd.DataFrame(rows)
    best_model = table.sort_values("Validation R2 (5-fold CV)", ascending=False).iloc[0]["Algorithm"]
    log(f"Best model by validation score: {best_model}")

    # CART rules (depth-limited for readability)
    cart = joblib.load(out_dir / MODEL_FILES["CART Regression"]).regressor_
    rules = export_text(
        cart.named_steps["model"],
        feature_names=list(cart.named_steps["preprocessor"].get_feature_names_out()),
        max_depth=4,
        decimals=2,
    )
    (out_dir / "cart_rules.txt").write_text(
        "Leaf values are log(1 + price). exp(value) - 1 gives the price in $.\n"
        "Numeric features are standardised (z-scores).\n\n" + rules,
        encoding="utf-8",
    )

    table.to_csv(out_dir / "comparison_table.csv", index=False)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    listings = F.copy()
    listings["price"] = y
    listings.to_csv(out_dir / "listings.csv", index=False)
    pruning_df.to_csv(out_dir / "pruning_path.csv", index=False)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_file": data_path.name,
        "rows": {"raw": raw_rows, "clean": len(X_raw), "train": len(X_train), "test": len(X_test)},
        "best_model": best_model,
        "best_ccp_alpha": best_alpha,
        "model_files": MODEL_FILES,
        "comparison": rows,
        "notebook_baseline": baseline,
        "explanations": explanations,
        "meta": build_meta(F, y),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Saved models and metrics to {out_dir}")

    save_plots(plots_dir, table, baseline, preds, best_model, explanations, log)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train used-car resale price models.")
    parser.add_argument("--data", default=str(ROOT / "used_cars.csv"), help="path to used_cars.csv")
    parser.add_argument("--out", default=str(ROOT / "models"), help="output folder for models")
    parser.add_argument("--plots", default=str(ROOT / "plots"), help="output folder for plots")
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"Dataset not found: {args.data}\nPut used_cars.csv next to train.py or pass --data.")
    run(args.data, args.out, args.plots)


if __name__ == "__main__":
    main()
