# Used Car Resale Value Estimation

Predicts the resale price of a used car using five regression models: Linear Regression,
CART, Pruned CART, Random Forest and Gradient Boosting. It includes an interactive
Streamlit web demo.

| File | What it is |
|---|---|
| `Unit_05.ipynb` | Original analysis notebook (EDA, CART, model comparison, pruning) |
| `car_features.py` | Shared cleaning and feature engineering used by training and the app |
| `train.py` | Trains every model and writes `models/` and `plots/` |
| `app.py` | Streamlit web demo |
| `used_cars.csv` | Dataset (4,009 US used-car listings) |

## Quick start

```bash
pip install -r requirements.txt
python train.py          # about 30 s; creates models/ and plots/
streamlit run app.py     # opens http://localhost:8501
```

If you skip `python train.py`, the app offers a **Train models now** button. If `used_cars.csv`
is missing, it lets you upload the file instead.

## Run with Docker (no Python needed)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it. Then:

```bash
git clone <this-repo-url>
cd <repo-folder>
docker build -t car-resale .
docker run -p 8501:8501 car-resale
```

Open http://localhost:8501. The image trains the models while it builds, which takes about a
minute the first time.

## What the web demo shows

1. **Estimate a price**: enter a car (brand, model, year, mileage, engine, transmission, colours,
   accident history). You get the estimate from the chosen model, every other model's estimate,
   a likely price range, value-vs-mileage and value-vs-year curves, and similar real listings.
2. **Model comparison**: the comparison table (Algorithm | Validation score | Test score |
   Training time | Inference time | Interpretability), R²/MAE/RMSE charts, actual vs predicted
   plots, feature importance, CART rules and the pruning curve. It also puts the original
   notebook pipeline next to the improved one.
3. **Explore the data**: price distribution, price by brand, price vs mileage and price by year,
   all filterable by brand.

## Method

- **Cleaning** (same as the notebook): drop duplicates and rows with missing values, then parse
  `price` from text. 80/20 train/test split with `random_state=42`.
- **Features** (improved pipeline):
  - mileage parsed to a number
  - horsepower, engine size (L) and cylinders extracted from the engine description
  - one-hot encoding for brand, model, fuel type, transmission, colours and accident history,
    with categories seen fewer than 5 times grouped together
- **Target**: models learn `log(1 + price)`, and predictions are converted back to dollars.
- **Validation score**: mean R² over 5-fold cross-validation on the training set.
- **Test score**: R², MAE and RMSE on the held-out test set.
- **Model selection**: the best model is picked by validation score.
- **Pruned CART**: `ccp_alpha` is chosen on a validation split taken from the training set.

`train.py` also reruns the notebook's exact pipeline, so the app can show both sets of results
side by side.

## Outputs of `train.py`

```
models/
  *.joblib                 trained pipelines
  metrics.json             all metrics and metadata used by the app
  comparison_table.csv     model comparison table for the report
  test_predictions.csv     actual vs predicted (test set)
  listings.csv             cleaned dataset
  cart_rules.txt           CART decision rules
  pruning_path.csv         pruning alpha vs validation error
plots/
  model_comparison_r2.png
  baseline_vs_improved.png
  actual_vs_predicted_best.png
  feature_importance_*.png
```

## Deploying (optional)

Push this folder to GitHub, including `used_cars.csv`. Then go to
[share.streamlit.io](https://share.streamlit.io), choose the repository and set `app.py` as the
entry point. The app trains itself on first launch if `models/` is not committed.
