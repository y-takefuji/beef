"""
HBR (High-Breakdown Rank) Regression for 20-day-ahead beef price
- Two variants: HBR_scaled (StandardScaler) and HBR_raw (no scaling)
- Time-series validation (forward chaining with gap, no shuffling)
- Target: 20-day price change (y - current price), converted back to price level
- Removes Year and other non-extrapolating / leak-prone features
- Compares against a persistence baseline
- Summary table: method, directional accuracy, 95% CI (Wilson), mean absolute error (3 significant digits)

Model
  HBR : High-Breakdown Rank estimator (weighted pairwise Wilcoxon dispersion,
        leverage weights from robust Mahalanobis distance via MCD)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
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

HBR_MAX_PAIRS = 20000     # max number of (i, j) pairs used by HBR (random subsample for speed)
HBR_PCA_VAR = 0.95        # variance kept in PCA before robust distance (stabilises MCD)
RANDOM_STATE = 42

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
    """Format a number with 3 significant digits."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    s = f"{float(v):.3g}"
    if "e" in s:
        s = f"{float(s):.0f}" if abs(float(v)) >= 1 else s
    return s

# ----------------------------------------------------
# 2. HBR estimator
# ----------------------------------------------------
class HBRRegressor(BaseEstimator, RegressorMixin):
    """
    High-Breakdown Rank (HBR) estimator.
        D(b) = sum_{i<j} w_ij * | e_i - e_j | ,   w_ij = 1 / (w_i * w_j) style leverage weights
    Leverage weights w_i = min(1, c / d_i) with d_i the robust (MCD) Mahalanobis distance.
    Large-leverage x-points are down-weighted, giving bounded influence in x.
    A random subsample of pairs is used for speed.
    """
    def __init__(self, max_pairs=HBR_MAX_PAIRS, pca_var=HBR_PCA_VAR,
                 random_state=RANDOM_STATE, max_iter=500):
        self.max_pairs = max_pairs
        self.pca_var = pca_var
        self.random_state = random_state
        self.max_iter = max_iter

    def _leverage_weights(self, X):
        n, p = X.shape
        Z = X
        if p > 1:
            pca = PCA(n_components=self.pca_var, svd_solver="full").fit(X)
            Z = pca.transform(X)
            # MCD needs n > ~ 2 * dim; cap components if needed
            max_dim = max(1, n // 3)
            Z = Z[:, :max_dim]
        try:
            mcd = MinCovDet(random_state=self.random_state).fit(Z)
            d = np.sqrt(mcd.mahalanobis(Z))
        except Exception:
            d = np.sqrt(np.sum((Z - np.median(Z, axis=0)) ** 2, axis=1))
        cutoff = np.sqrt(np.percentile(d ** 2, 90)) if np.all(np.isfinite(d)) else 1.0
        cutoff = max(cutoff, 1e-8)
        w = np.minimum(1.0, cutoff / np.maximum(d, 1e-8))
        return w

    def _dispersion(self, b, X, y, I, J, W):
        e = y - X @ b
        diff = e[I] - e[J]
        s = np.sign(diff)
        D = np.sum(W * np.abs(diff))
        # sub-gradient: d/db |e_i - e_j| = -sign(diff) * (x_i - x_j)
        coeff = -W * s
        g = (X[I] - X[J]).T @ coeff
        return D, g

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        w = self._leverage_weights(X)

        # Build (i, j) pairs (all pairs if affordable, otherwise a random subsample)
        total = n * (n - 1) // 2
        if total <= self.max_pairs:
            I, J = np.triu_indices(n, k=1)
        else:
            I = rng.randint(0, n, size=self.max_pairs)
            J = rng.randint(0, n, size=self.max_pairs)
            keep = I != J
            I, J = I[keep], J[keep]
        W = 1.0 / (w[I] * w[J])
        W = W / W.mean()

        b0 = LinearRegression().fit(X, y).coef_
        res = minimize(self._dispersion, b0, args=(X, y, I, J, W), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.intercept_ = float(np.median(y - X @ self.coef_))
        self.leverage_weights_ = w
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

# ----------------------------------------------------
# 3. Load data
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
# 4. Features / target
# ----------------------------------------------------
y_level = df[TARGET_COL].values
y_test_level = df_test[TARGET_COL].values

X = df.drop(columns=[TARGET_COL])
X_test = df_test.drop(columns=[TARGET_COL])

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print("Non-numeric columns dropped:", non_numeric)
    X = X.drop(columns=non_numeric)

drop_now = [c for c in DROP_FEATURES if c in X.columns]
X = X.drop(columns=drop_now)
print("Dropped features:", drop_now)

missing = [c for c in X.columns if c not in X_test.columns]
if missing:
    raise ValueError(f"Columns missing in test.csv: {missing}")
X_test = X_test[X.columns]

# Causal imputation: forward-fill only uses past values (median fill in the pipeline handles leading NaNs)
X = X.ffill()
X_test = pd.concat([X.tail(60), X_test]).ffill().iloc[60:]

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
# 5. Models: HBR with scaling vs. raw (no scaling)
# ----------------------------------------------------
def make_pipe(estimator, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)

models = {
    "HBR_scaled": make_pipe(HBRRegressor(), scale=True),
    "HBR_raw":    make_pipe(HBRRegressor(), scale=False),
}

# ----------------------------------------------------
# 6. Time-series cross-validation (forward chaining with gap)
# ----------------------------------------------------
tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)

def eval_level(y_true_level, pred_change, base_price):
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

pd.DataFrame(cv_rows).to_csv("cv_summary_hbr.csv", index=False)

# ----------------------------------------------------
# 7. Final fit on all training data and independent test evaluation
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

pd.DataFrame(test_rows).to_csv("test_summary_hbr.csv", index=False)

# ----------------------------------------------------
# 8. Summary table (method, directional accuracy, 95% CI, absolute error) - 3 significant digits
# ----------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 70)
print("SUMMARY TABLE (independent test, 3 significant digits)")
print("=" * 70)
print(summary_df.to_string(index=False))
summary_df.to_csv("summary_table_hbr.csv", index=False)

# ----------------------------------------------------
# 9. HBR details (leverage weights) for both variants
# ----------------------------------------------------
for name in models:
    hbr = models[name].named_steps["model"]
    w = hbr.leverage_weights_
    print(f"\n===== {name} =====")
    print(f"Down-weighted training points (w < 1): {int(np.sum(w < 1))} / {len(w)}")
    print(f"Min / mean leverage weight:            {w.min():.3g} / {w.mean():.3g}")

# ----------------------------------------------------
# 10. Coefficients
#     - scaled: magnitudes comparable across features
#     - raw: coefficients are in original feature units (not directly comparable)
# ----------------------------------------------------
coef_df = pd.DataFrame({"feature": X.columns})
for name in models:
    coef_df[name] = models[name].named_steps["model"].coef_
coef_df["abs_mean"] = coef_df.drop(columns="feature").abs().mean(axis=1)
coef_df = coef_df.sort_values("abs_mean", ascending=False)
print("\n===== Top 15 Coefficients =====")
print(coef_df.head(15).drop(columns="abs_mean").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
for name in models:
    print(f"Intercept {name}: {models[name].named_steps['model'].intercept_:.3g}")
coef_df.to_csv("hbr_coefficients.csv", index=False)

# ----------------------------------------------------
# 11. Save predictions and plot
# ----------------------------------------------------
out = pd.DataFrame({"y_true": y_test_level, "price_now": price_now_test})
for name, p in results.items():
    out[f"pred_{name}"] = p
out.to_csv("test_predictions_hbr.csv", index=False)
print("\nSaved test_predictions_hbr.csv")

plt.figure(figsize=(12, 5))
plt.plot(y_test_level, label="Actual (t+20)", color="black", linewidth=2)
plt.plot(price_now_test, label="Persistence baseline", color="gray", linestyle="--")
for name, p in results.items():
    plt.plot(p, label=f"{name} (change target)", alpha=0.85)
plt.legend(); plt.title("2026 test: actual vs predicted (HBR scaled vs raw)")
plt.xlabel("Test sample index")
plt.tight_layout(); plt.savefig("test_plot_hbr.png", dpi=120)
print("Saved test_plot_hbr.png")