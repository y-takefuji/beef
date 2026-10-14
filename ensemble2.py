"""
Tree-ensemble Regression (RandomForest, XGBoost, CatBoost) for 20-day-ahead beef price
- Time-series validation (forward chaining with gap, no shuffling)
- Target: 20-day price change (y - current price), converted back to price level
- Removes Year and other non-extrapolating / leak-prone features
- Compares against a persistence baseline
- Summary table: method, directional accuracy, 95% CI (Wilson), mean absolute error (3 significant digits)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ----------------------------------------------------
# 0. Settings
# ----------------------------------------------------
TRAIN_PATH = "set10_rev.csv"
TEST_PATH = "test_rev.csv"
TARGET_COL = "y"
PRICE_COL = "116A_3"      # current price (assumed)
DATE_COL = None           # e.g. "date" if you have one; otherwise rows are assumed time-ordered
HORIZON = 20              # forecast horizon in days
GAP = 40                  # gap between train and validation (>= horizon; longer because of 12M/52w features)
N_SPLITS = 5
SEED = 42

# Features that cannot extrapolate or may leak regime/date information
DROP_FEATURES = ["Year"]  # add "Event", "D", "M", "Y_cow_cycle" here if they are date/cycle labels

# ----------------------------------------------------
# 1. Helper metrics
# ----------------------------------------------------
def directional_accuracy(true_change, pred_change):
    """Share of samples where predicted and actual 20d change have the same sign."""
    return float(np.mean(np.sign(pred_change) == np.sign(true_change)))


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def fmt3(v):
    """Format a number with 3 significant digits (e.g. 0.8123 -> '0.812', 1234.5 -> '1230')."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    s = f"{float(v):.3g}"
    # avoid scientific notation for ordinary magnitudes
    if "e" in s:
        s = f"{float(s):.0f}" if abs(float(v)) >= 1 else s
    return s

# ----------------------------------------------------
# 2. Load data
# ----------------------------------------------------
df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

if DATE_COL:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df_test[DATE_COL] = pd.to_datetime(df_test[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df_test = df_test.sort_values(DATE_COL).reset_index(drop=True)

print("Train shape:", df.shape, "| Test shape:", df_test.shape)

# ----------------------------------------------------
# 3. Features / target
# ----------------------------------------------------
y_level = df[TARGET_COL].values
y_test_level = df_test[TARGET_COL].values

X = df.drop(columns=[TARGET_COL])
X_test = df_test.drop(columns=[TARGET_COL])

# Drop non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Non-numeric columns dropped:", non_numeric)
    X = X.drop(columns=non_numeric)

# Drop features that harm out-of-sample behaviour
drop_now = [c for c in DROP_FEATURES if c in X.columns]
X = X.drop(columns=drop_now)
print("Dropped features:", drop_now)

# Align test columns
missing = [c for c in X.columns if c not in X_test.columns]
if missing:
    raise ValueError(f"Columns missing in test.csv: {missing}")
X_test = X_test[X.columns]

# Causal imputation: forward-fill only uses past values (median fill below handles leading NaNs)
X = X.ffill()
X_test = pd.concat([X.tail(60), X_test]).ffill().iloc[60:]   # carry over the last training values

# Current price (used for target transformation and baseline)
price_now = X[PRICE_COL].values
price_now_test = X_test[PRICE_COL].values

# Stationary target: change over the horizon
y_change = y_level - price_now
y_test_change = y_test_level - price_now_test

print(f"Train y range: {y_level.min():.3g} - {y_level.max():.3g}")
print(f"Test  y range: {y_test_level.min():.3g} - {y_test_level.max():.3g}")
if y_test_level.max() > y_level.max():
    print("NOTE: test prices exceed the training range (change target helps with this).")

# ----------------------------------------------------
# 4. Models (trees need no scaling; only imputation)
# ----------------------------------------------------
def make_pipe(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])

models = {
    "RandomForest": make_pipe(RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        max_features=0.5, n_jobs=-1, random_state=SEED)),
    "XGBoost": make_pipe(XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=5.0, n_jobs=-1, random_state=SEED, verbosity=0)),
    "CatBoost": make_pipe(CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=5,
        l2_leaf_reg=5.0, random_seed=SEED, verbose=0)),
}

# ----------------------------------------------------
# 5. Time-series cross-validation (forward chaining with gap)
# ----------------------------------------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)

def eval_level(y_true_level, pred_change, base_price):
    """Convert predicted change to price level and score on the level."""
    pred_level = base_price + pred_change
    return r2_score(y_true_level, pred_level), pred_level

print("\n===== Time-Series CV (gap = %d) =====" % GAP)
cv_rows = []
for name, model in models.items():
    r2_level, r2_change, r2_base = [], [], []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        model.fit(X.iloc[tr], y_change[tr])
        pred_change = model.predict(X.iloc[va])
        r2_l, _ = eval_level(y_level[va], pred_change, price_now[va])
        r2_c = r2_score(y_change[va], pred_change)
        r2_b = r2_score(y_level[va], price_now[va])   # persistence baseline
        r2_level.append(r2_l); r2_change.append(r2_c); r2_base.append(r2_b)
        print(f"{name} Fold {fold}: R2(level)={r2_l:.3g} | R2(change)={r2_c:.3g} | "
              f"Baseline R2(level)={r2_b:.3g}")
    print(f"{name} MEAN: R2(level)={np.mean(r2_level):.3g}, R2(change)={np.mean(r2_change):.3g}, "
          f"Baseline={np.mean(r2_base):.3g}\n")
    cv_rows.append({
        "model": name,
        "CV R2 level": np.mean(r2_level),
        "CV R2 change": np.mean(r2_change),
        "CV R2 change std": np.std(r2_change),
        "Baseline R2 level": np.mean(r2_base),
    })

pd.DataFrame(cv_rows).to_csv("cv_summary.csv", index=False)

# ----------------------------------------------------
# 6. Final fit on all training data and independent test evaluation
# ----------------------------------------------------
print("===== Independent Test Evaluation (test_rev.csv) =====")
baseline_r2 = r2_score(y_test_level, price_now_test)
baseline_mae = mean_absolute_error(y_test_level, price_now_test)
print(f"Persistence baseline: R2={baseline_r2:.3g}, MAE={baseline_mae:.3g}")

results = {}
test_rows = []
summary_rows = []
for name, model in models.items():
    model.fit(X, y_change)
    pred_change = model.predict(X_test)
    pred_level = price_now_test + pred_change
    results[name] = pred_level

    r2_l = r2_score(y_test_level, pred_level)
    r2_c = r2_score(y_test_change, pred_change)
    rmse = np.sqrt(mean_squared_error(y_test_level, pred_level))
    mae = mean_absolute_error(y_test_level, pred_level)
    bias = np.mean(y_test_level - pred_level)

    # Directional accuracy with 95% Wilson CI
    n = len(y_test_change)
    k = int(np.sum(np.sign(pred_change) == np.sign(y_test_change)))
    direction = k / n
    ci_lo, ci_hi = wilson_ci(k, n)

    test_rows.append({
        "model": name, "R2 level": r2_l, "R2 change": r2_c,
        "RMSE": rmse, "MAE": mae, "Bias": bias, "Directional acc": direction,
    })
    summary_rows.append({
        "Method": name,
        "Directional accuracy": fmt3(direction),
        "95% CI": f"[{fmt3(ci_lo)}, {fmt3(ci_hi)}]",
        "Absolute error (MAE)": fmt3(mae),
    })

    print(f"\n{name}")
    print(f"  R2 (level):          {r2_l:.3g}")
    print(f"  R2 (20d change):     {r2_c:.3g}")
    print(f"  RMSE / MAE (level):  {rmse:.3g} / {mae:.3g}")
    print(f"  Bias (mean resid):   {bias:.3g}")
    print(f"  Directional accuracy:{direction:.3g}  95% CI [{ci_lo:.3g}, {ci_hi:.3g}]")

pd.DataFrame(test_rows).to_csv("test_summary.csv", index=False)

# ----------------------------------------------------
# 7. Summary table (method, directional accuracy, 95% CI, absolute error) - 3 significant digits
# ----------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 70)
print("SUMMARY TABLE (independent test, 3 significant digits)")
print("=" * 70)
print(summary_df.to_string(index=False))
summary_df.to_csv("summary_table.csv", index=False)

# ----------------------------------------------------
# 8. Feature importances (normalized to sum to 1 for comparability)
# ----------------------------------------------------
imp_df = pd.DataFrame({"feature": X.columns})
for name, model in models.items():
    imp = np.asarray(model.named_steps["model"].feature_importances_, dtype=float)
    imp_df[name] = imp / imp.sum() if imp.sum() > 0 else imp
imp_df["mean_importance"] = imp_df.drop(columns="feature").mean(axis=1)
imp_df = imp_df.sort_values("mean_importance", ascending=False)
print("\n===== Top 15 Feature Importances =====")
print(imp_df.head(15).drop(columns="mean_importance").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
imp_df.to_csv("feature_importances.csv", index=False)

# ----------------------------------------------------
# 9. Save predictions and plot
# ----------------------------------------------------
out = pd.DataFrame({"y_true": y_test_level, "price_now": price_now_test})
for name, p in results.items():
    out[f"pred_{name}"] = p
out.to_csv("test_predictions.csv", index=False)
print("\nSaved test_predictions.csv")

plt.figure(figsize=(12, 5))
plt.plot(y_test_level, label="Actual (t+20)", color="black", linewidth=2)
plt.plot(price_now_test, label="Persistence baseline", color="gray", linestyle="--")
for name, p in results.items():
    plt.plot(p, label=f"{name} (change target)", alpha=0.85)
plt.legend(); plt.title("2026 test: actual vs predicted (tree ensembles)"); plt.xlabel("Test sample index")
plt.tight_layout(); plt.savefig("test_plot.png", dpi=120)
print("Saved test_plot.png")