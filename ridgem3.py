"""
Ridge-family Regression (Ridge, RidgeCV, BayesianRidge, ARDRegression) for 20-day-ahead beef price
- Each model is run in two variants: [scaled] (StandardScaler) and [raw] (no scaling)
- Time-series validation (forward chaining with gap, no shuffling)
- Target: 20-day price change (y - current price), converted back to price level
- Removes Year and other non-extrapolating / leak-prone features
- Compares against a persistence baseline
- Summary tables (one per variant): Method, Directional accuracy, 95% CI (Wilson), Absolute error (MAE)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, BayesianRidge, ARDRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
RIDGE_ALPHA = 100.0       # fixed penalty strength for plain Ridge (not tuned)

# Features that cannot extrapolate or may leak regime/date information
DROP_FEATURES = ["Year"]  # add "Event", "D", "M", "Y_cow_cycle" here if they are date/cycle labels

# ----------------------------------------------------
# 1. Helper metrics and formatting
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
    """3 significant digits, trailing zeros removed (e.g. 0.76, 41.7, 77)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    s = f"{float(v):.3g}"
    if "e" in s:                       # avoid scientific notation
        s = f"{float(s):f}".rstrip("0").rstrip(".")
    return s


def fmt_acc(v):
    """Directional accuracy with 3 decimals, trailing zero kept (e.g. 0.680)."""
    return f"{float(v):.3f}"

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

# Causal imputation: forward-fill only uses past values (median fill in the pipeline handles leading NaNs)
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
# 4. Models: every model in two variants ([scaled] and [raw])
# ----------------------------------------------------
def make_pipe(estimator, scale):
    """Pipeline: median imputer -> (optional StandardScaler) -> estimator."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

SCALE_TAGS = {True: "scaled", False: "raw"}

def build_models():
    m = {}
    for scale in (True, False):
        tag = SCALE_TAGS[scale]
        m[f"Ridge [{tag}]"] = make_pipe(Ridge(alpha=RIDGE_ALPHA), scale)
        m[f"RidgeCV [{tag}]"] = make_pipe(RidgeCV(alphas=np.logspace(-1, 4, 30)), scale)
        m[f"BayesianRidge [{tag}]"] = make_pipe(BayesianRidge(), scale)
        m[f"ARDRegression [{tag}]"] = make_pipe(ARDRegression(), scale)
    return m

models = build_models()
SCALE_TAG_OF = {n: n.split("[")[1].rstrip("]") for n in models}

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
        extra = ""
        if name.startswith("RidgeCV"):
            extra = f" | alpha={model.named_steps['model'].alpha_:.3g}"
        print(f"{name} Fold {fold}: R2(level)={r2_l:.3g} | R2(change)={r2_c:.3g} | "
              f"Baseline R2(level)={r2_b:.3g}{extra}")
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
results_change = {}
test_rows = []
summary_rows = []

for name, model in models.items():
    model.fit(X, y_change)
    pred_change = model.predict(X_test)
    pred_level = price_now_test + pred_change
    results[name] = pred_level
    results_change[name] = pred_change

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
        "Method": name.split(" [")[0],
        "Variant": SCALE_TAG_OF[name],
        "Directional accuracy": fmt_acc(direction),
        "95% CI": f"[{fmt3(ci_lo)}, {fmt3(ci_hi)}]",
        "Absolute error (MAE)": fmt3(mae),
    })

    print(f"\n{name}")
    if name.startswith("Ridge ["):
        print(f"  Fixed alpha:         {RIDGE_ALPHA:g}")
    if name.startswith("RidgeCV"):
        print(f"  Selected alpha:      {model.named_steps['model'].alpha_:.3g}")
    print(f"  R2 (level):          {r2_l:.3g}")
    print(f"  R2 (20d change):     {r2_c:.3g}")
    print(f"  RMSE / MAE (level):  {rmse:.3g} / {mae:.3g}")
    print(f"  Bias (mean resid):   {bias:.3g}")
    print(f"  Directional accuracy:{direction:.3g}  95% CI [{ci_lo:.3g}, {ci_hi:.3g}]")

pd.DataFrame(test_rows).to_csv("test_summary.csv", index=False)

# ----------------------------------------------------
# 7. Summary tables (one per variant):
#    Method | Directional accuracy | 95% CI | Absolute error (MAE)
# ----------------------------------------------------
SUMMARY_COLS = ["Method", "Directional accuracy", "95% CI", "Absolute error (MAE)"]
summary_all = pd.DataFrame(summary_rows)

for tag in ("scaled", "raw"):
    tab = summary_all[summary_all["Variant"] == tag][SUMMARY_COLS]
    print("\n" + "=" * 78)
    print(f"SUMMARY TABLE [{tag}] (independent test)")
    print("=" * 78)
    print(tab.to_csv(sep="\t", index=False))          # tab-separated, paste-ready
    tab.to_csv(f"summary_table_{tag}.csv", index=False)

summary_all.to_csv("summary_table_all.csv", index=False)

# ----------------------------------------------------
# 8. BayesianRidge details (predictive uncertainty)
# ----------------------------------------------------
br_levels, br_stds = {}, {}
for tag in ("scaled", "raw"):
    br = models[f"BayesianRidge [{tag}]"]
    Xt = br[:-1].transform(X_test)                      # imputer (+ scaler if scaled)
    br_pred, br_std = br.named_steps["model"].predict(Xt, return_std=True)
    br_level = price_now_test + br_pred
    br_levels[tag], br_stds[tag] = br_level, br_std
    coverage = np.mean(np.abs(y_test_level - br_level) <= 1.96 * br_std)
    print(f"\n===== BayesianRidge [{tag}] =====")
    print(f"Noise precision (alpha_):   {br.named_steps['model'].alpha_:.3g}")
    print(f"Weight precision (lambda_): {br.named_steps['model'].lambda_:.3g}")
    print(f"Mean predictive std:        {br_std.mean():.3g}")
    print(f"Test points inside 95% interval: {coverage:.3g}")

# ----------------------------------------------------
# 9. ARDRegression details (feature relevance)
# ----------------------------------------------------
for tag in ("scaled", "raw"):
    ard = models[f"ARDRegression [{tag}]"].named_steps["model"]
    n_active = int(np.sum(np.abs(ard.coef_) > 1e-6))
    print(f"\n===== ARDRegression [{tag}] =====")
    print(f"Active (non-zero) features: {n_active} / {len(ard.coef_)}")

# ----------------------------------------------------
# 10. Coefficients (compare within a variant only: scaled and raw are in different units;
#     scaled coefficients are comparable across features, raw coefficients are not)
# ----------------------------------------------------
for tag in ("scaled", "raw"):
    names = [n for n in models if n.endswith(f"[{tag}]")]
    coef_df = pd.DataFrame({"feature": X.columns})
    for name in names:
        coef_df[name] = models[name].named_steps["model"].coef_
    coef_df["abs_mean"] = coef_df.drop(columns="feature").abs().mean(axis=1)
    coef_df = coef_df.sort_values("abs_mean", ascending=False)
    print(f"\n===== Top 15 Coefficients [{tag}] =====")
    print(coef_df.head(15).drop(columns="abs_mean").to_string(index=False,
          float_format=lambda v: f"{v:.3g}"))
    for name in names:
        print(f"Intercept {name}: {models[name].named_steps['model'].intercept_:.3g}")
    coef_df.to_csv(f"ridge_family_coefficients_{tag}.csv", index=False)

# ----------------------------------------------------
# 11. Save predictions and plots
# ----------------------------------------------------
out = pd.DataFrame({"y_true": y_test_level, "price_now": price_now_test})
for name, p in results.items():
    out[f"pred_{name}"] = p
for tag in ("scaled", "raw"):
    out[f"BayesianRidge_std_{tag}"] = br_stds[tag]
out.to_csv("test_predictions.csv", index=False)
print("\nSaved test_predictions.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, tag in zip(axes, ("scaled", "raw")):
    ax.plot(y_test_level, label="Actual (t+20)", color="black", linewidth=2)
    ax.plot(price_now_test, label="Persistence baseline", color="gray", linestyle="--")
    for name, p in results.items():
        if name.endswith(f"[{tag}]"):
            ax.plot(p, label=name, alpha=0.85)
    ax.fill_between(range(len(br_levels[tag])),
                    br_levels[tag] - 1.96 * br_stds[tag],
                    br_levels[tag] + 1.96 * br_stds[tag],
                    color="tab:orange", alpha=0.15, label="BayesianRidge 95% interval")
    ax.set_title(f"2026 test: actual vs predicted [{tag}]")
    ax.set_xlabel("Test sample index")
    ax.legend()
plt.tight_layout(); plt.savefig("test_plot.png", dpi=120)
print("Saved test_plot.png")