"""
Rank-based Linear Regression (Jaeckel, Wilcoxon, HBR, Rank-Transform) for 20-day-ahead beef price
- Each model is run in two variants: [scaled] (StandardScaler) and [raw] (no scaling)
- Time-series validation (forward chaining with gap, no shuffling)
- Target: 20-day price change (y - current price), converted back to price level
- Removes Year and other non-extrapolating / leak-prone features
- Compares against a persistence baseline
- Summary tables (one per variant): Method, Directional accuracy, 95% CI (Wilson), Absolute error (MAE)

Models
  Jaeckel       : R-estimator, Jaeckel dispersion with Wilcoxon scores (no penalty)
  Wilcoxon      : Wilcoxon-score rank regression with a small L2 shrinkage
  HBR           : High-Breakdown Rank estimator (weighted pairwise Wilcoxon dispersion,
                  leverage weights from robust Mahalanobis distance via MCD on
                  highly variable features)
  RankTransform : OLS on rank-transformed features and target
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.covariance import MinCovDet
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

WIL_L2 = 1.0              # L2 shrinkage for the Wilcoxon model (0 = none)
RT_L2 = 0.0               # L2 shrinkage for the rank-transform model (0 = plain OLS on ranks)
HBR_MAX_PAIRS = 20000     # max number of (i, j) pairs used by HBR (random subsample for speed)
HBR_N_HVF = 20            # number of highly variable features kept for the HBR leverage weights
HBR_N_BINS = 10           # mean-bins for normalized dispersion (HVG-style selection)
RANDOM_STATE = 0

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
# 2. Highly variable feature selection (variance / dispersion based)
# ----------------------------------------------------
def select_highly_variable(X, n_top, n_bins=HBR_N_BINS):
    """
    HVG-style selection on raw (unscaled) features.
      1. mean and variance per feature
      2. dispersion = variance / |mean|  (floored)
      3. bin features by mean; z-score dispersion within each bin (normalized dispersion)
      4. keep the top n_top features by normalized dispersion
    Falls back to plain top-variance if there are too few features for binning.
    Returns a sorted array of selected column indices.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    n_top = int(min(max(n_top, 1), p))

    mean = X.mean(axis=0)
    var = X.var(axis=0)

    if p < 2 * n_bins:
        return np.sort(np.argsort(var)[::-1][:n_top])

    disp = var / np.maximum(np.abs(mean), 1e-8)
    disp = np.log(np.maximum(disp, 1e-12))
    mean_log = np.log1p(np.abs(mean))

    edges = np.unique(np.quantile(mean_log, np.linspace(0, 1, n_bins + 1)))
    bin_id = np.clip(np.digitize(mean_log, edges[1:-1]), 0, len(edges) - 2)
    norm_disp = np.zeros(p)
    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        if len(idx) < 2:
            norm_disp[idx] = 0.0
            continue
        sd = disp[idx].std()
        norm_disp[idx] = (disp[idx] - disp[idx].mean()) / (sd if sd > 0 else 1.0)

    norm_disp[var <= 0] = -np.inf
    return np.sort(np.argsort(norm_disp)[::-1][:n_top])

# ----------------------------------------------------
# 3. Rank-based regression estimators
# ----------------------------------------------------
def wilcoxon_scores(n):
    """Wilcoxon score function a(i) = sqrt(12) * (i/(n+1) - 0.5), i = 1..n."""
    return np.sqrt(12.0) * (np.arange(1, n + 1) / (n + 1.0) - 0.5)


class JaeckelRegressor(BaseEstimator, RegressorMixin):
    """
    Jaeckel's R-estimator with Wilcoxon scores.
        D(b) = sum_i a(R(e_i)) * e_i ,   e = y - X b
    Minimised with L-BFGS-B from an OLS start. Intercept = median of final residuals.
    `l2` adds an optional ridge-like shrinkage (l2 * ||b||^2) for stability.
    """
    def __init__(self, l2=0.0, max_iter=500):
        self.l2 = l2
        self.max_iter = max_iter

    def _dispersion(self, b, X, y, scores):
        e = y - X @ b
        order = np.argsort(e, kind="mergesort")
        grad_e = np.empty_like(e)
        grad_e[order] = scores
        D = np.sum(scores * e[order])
        g = -(X.T @ grad_e)
        if self.l2 > 0:
            D += self.l2 * np.sum(b ** 2)
            g = g + 2 * self.l2 * b
        return D, g

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        scores = wilcoxon_scores(n)
        b0 = LinearRegression().fit(X, y).coef_
        res = minimize(self._dispersion, b0, args=(X, y, scores), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.intercept_ = float(np.median(y - X @ self.coef_))
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_


class WilcoxonRegressor(JaeckelRegressor):
    """Wilcoxon-score rank regression with L2 shrinkage."""
    def __init__(self, l2=WIL_L2, max_iter=500):
        super().__init__(l2=l2, max_iter=max_iter)


class HBRRegressor(BaseEstimator, RegressorMixin):
    """
    High-Breakdown Rank (HBR) estimator.
        D(b) = sum_{i<j} w_ij * | e_i - e_j | ,   w_ij = 1 / (w_i * w_j)
    Leverage weights w_i = min(1, c / d_i), with d_i the robust (MCD) Mahalanobis distance
    computed on the highly variable features (selected on the RAW features; no PCA).

    scale=True : features are standardized internally (after imputation) and the
                 regression / robust distance use the standardized features.
    scale=False: the raw imputed features are used for both.
    Feature selection always uses the raw features (variance-based).
    """
    def __init__(self, scale=True, max_pairs=HBR_MAX_PAIRS, n_hvf=HBR_N_HVF,
                 n_bins=HBR_N_BINS, random_state=RANDOM_STATE, max_iter=500):
        self.scale = scale
        self.max_pairs = max_pairs
        self.n_hvf = n_hvf
        self.n_bins = n_bins
        self.random_state = random_state
        self.max_iter = max_iter

    def _leverage_weights(self, X_raw, X_used):
        n, p = X_used.shape
        k = int(min(self.n_hvf, max(1, n // 3), p))
        self.hvf_idx_ = select_highly_variable(X_raw, k, self.n_bins)
        Z = X_used[:, self.hvf_idx_]
        try:
            mcd = MinCovDet(random_state=self.random_state).fit(Z)
            d = np.sqrt(mcd.mahalanobis(Z))
        except Exception:
            d = np.sqrt(np.sum((Z - np.median(Z, axis=0)) ** 2, axis=1))
        cutoff = np.sqrt(np.percentile(d ** 2, 90)) if np.all(np.isfinite(d)) else 1.0
        cutoff = max(cutoff, 1e-8)
        return np.minimum(1.0, cutoff / np.maximum(d, 1e-8))

    def _dispersion(self, b, X, y, I, J, W):
        e = y - X @ b
        diff = e[I] - e[J]
        s = np.sign(diff)
        D = np.sum(W * np.abs(diff))
        coeff = -W * s
        g = (X[I] - X[J]).T @ coeff
        return D, g

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        return self.scaler_.transform(X) if self.scale else X

    def fit(self, X, y):
        X_raw = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X_raw.shape
        rng = np.random.RandomState(self.random_state)

        if self.scale:
            self.scaler_ = StandardScaler().fit(X_raw)
        Xu = self._transform(X_raw)

        w = self._leverage_weights(X_raw, Xu)

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

        b0 = LinearRegression().fit(Xu, y).coef_
        res = minimize(self._dispersion, b0, args=(Xu, y, I, J, W), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = res.x
        self.intercept_ = float(np.median(y - Xu @ self.coef_))
        self.leverage_weights_ = w
        return self

    def predict(self, X):
        return self.intercept_ + self._transform(X) @ self.coef_


class RankTransformRegressor(BaseEstimator, RegressorMixin):
    """
    Rank-transform regression: OLS (optionally ridge) on ranks of X and y.
    New data are mapped through the TRAINING empirical CDF; the predicted target rank is
    converted back through the training target quantile function (bounded output).
    Ranks are invariant to monotone rescaling, so scaling does not change the result.
    """
    def __init__(self, l2=RT_L2):
        self.l2 = l2

    @staticmethod
    def _ecdf_map(sorted_ref, values):
        n = len(sorted_ref)
        left = np.searchsorted(sorted_ref, values, side="left")
        right = np.searchsorted(sorted_ref, values, side="right")
        return ((left + right) / 2.0 + 0.5) / (n + 1.0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.sorted_X_ = [np.sort(X[:, j]) for j in range(X.shape[1])]
        self.sorted_y_ = np.sort(y)
        Xr = np.column_stack([self._ecdf_map(self.sorted_X_[j], X[:, j])
                              for j in range(X.shape[1])])
        yr = self._ecdf_map(self.sorted_y_, y)
        self.lin_ = Ridge(alpha=self.l2).fit(Xr, yr) if self.l2 > 0 else LinearRegression().fit(Xr, yr)
        self.coef_ = self.lin_.coef_
        self.intercept_ = float(self.lin_.intercept_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xr = np.column_stack([self._ecdf_map(self.sorted_X_[j], X[:, j])
                              for j in range(X.shape[1])])
        yr_pred = np.clip(self.lin_.predict(Xr), 0.0, 1.0)
        return np.quantile(self.sorted_y_, yr_pred)

# ----------------------------------------------------
# 4. Load data
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
# 5. Features / target
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

y_change = y_level - price_now
y_test_change = y_test_level - price_now_test

print(f"Train y range: {y_level.min():.3g} - {y_level.max():.3g}")
print(f"Test  y range: {y_test_level.min():.3g} - {y_test_level.max():.3g}")
if y_test_level.max() > y_level.max():
    print("NOTE: test prices exceed the training range (change target helps with this).")

# ----------------------------------------------------
# 6. Models: every model in two variants ([scaled] and [raw])
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
        m[f"Jaeckel [{tag}]"] = make_pipe(JaeckelRegressor(l2=0.0), scale=scale)
        m[f"Wilcoxon [{tag}]"] = make_pipe(WilcoxonRegressor(l2=WIL_L2), scale=scale)
        # HBR scales internally (after variance-based feature selection), so no pipeline scaler
        m[f"HBR [{tag}]"] = make_pipe(HBRRegressor(scale=scale), scale=False)
        # Ranks are scale-invariant: pipeline scaler is applied for the [scaled] variant only
        m[f"RankTransform [{tag}]"] = make_pipe(RankTransformRegressor(l2=RT_L2), scale=scale)
    return m

models = build_models()
SCALE_TAG_OF = {n: n.split("[")[1].rstrip("]") for n in models}

# ----------------------------------------------------
# 7. Time-series cross-validation (forward chaining with gap)
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

pd.DataFrame(cv_rows).to_csv("cv_summary_rank.csv", index=False)

# ----------------------------------------------------
# 8. Final fit on all training data and independent test evaluation
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
        "Method": name.split(" [")[0],
        "Variant": SCALE_TAG_OF[name],
        "Directional accuracy": fmt_acc(direction),
        "95% CI": f"[{fmt3(ci_lo)}, {fmt3(ci_hi)}]",
        "Absolute error (MAE)": fmt3(mae),
    })

    print(f"\n{name}")
    print(f"  R2 (level):          {r2_l:.3g}")
    print(f"  R2 (20d change):     {r2_c:.3g}")
    print(f"  RMSE / MAE (level):  {rmse:.3g} / {mae:.3g}")
    print(f"  Bias (mean resid):   {bias:.3g}")
    print(f"  Directional accuracy:{direction:.3g}  95% CI [{ci_lo:.3g}, {ci_hi:.3g}]")

pd.DataFrame(test_rows).to_csv("test_summary_rank.csv", index=False)

# ----------------------------------------------------
# 9. Summary tables (one per variant):
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
    tab.to_csv(f"summary_table_rank_{tag}.csv", index=False)

summary_all.to_csv("summary_table_rank_all.csv", index=False)

# ----------------------------------------------------
# 10. HBR details (leverage weights and selected highly variable features)
# ----------------------------------------------------
for tag in ("scaled", "raw"):
    hbr = models[f"HBR [{tag}]"].named_steps["model"]
    w = hbr.leverage_weights_
    print(f"\n===== HBR [{tag}] =====")
    print(f"Down-weighted training points (w < 1): {int(np.sum(w < 1))} / {len(w)}")
    print(f"Min / mean leverage weight:            {w.min():.3g} / {w.mean():.3g}")
    print(f"Highly variable features used for leverage ({len(hbr.hvf_idx_)}):")
    print("  " + ", ".join(X.columns[hbr.hvf_idx_]))

# ----------------------------------------------------
# 11. Rank-transform details
# ----------------------------------------------------
rt = models["RankTransform [raw]"].named_steps["model"]
print("\n===== RankTransform =====")
print(f"Training target change range (output limit): "
      f"{rt.sorted_y_.min():.3g} to {rt.sorted_y_.max():.3g}")
print("NOTE: output is bounded by the training range of the 20d change; "
      "it cannot extrapolate beyond it.")
diff_rt = np.max(np.abs(results["RankTransform [scaled]"] - results["RankTransform [raw]"]))
print(f"Max |scaled - raw| prediction difference (should be ~0 for ranks): {diff_rt:.3g}")

# ----------------------------------------------------
# 12. Coefficients (compared within a variant only: scaled and raw are in different units)
#     RankTransform coefficients are on rank-scaled features in (0, 1)
# ----------------------------------------------------
for tag in ("scaled", "raw"):
    names = [n for n in models if n.endswith(f"[{tag}]")]
    coef_df = pd.DataFrame({"feature": X.columns})
    for name in names:
        coef_df[name] = models[name].named_steps["model"].coef_
    norm = coef_df.drop(columns="feature").abs().max().replace(0, 1)
    coef_df["abs_mean"] = (coef_df.drop(columns="feature").abs() / norm).mean(axis=1)
    coef_df = coef_df.sort_values("abs_mean", ascending=False)
    print(f"\n===== Top 15 Coefficients [{tag}] =====")
    print(coef_df.head(15).drop(columns="abs_mean").to_string(index=False,
          float_format=lambda v: f"{v:.3g}"))
    for name in names:
        print(f"Intercept {name}: {models[name].named_steps['model'].intercept_:.3g}")
    coef_df.to_csv(f"rank_regression_coefficients_{tag}.csv", index=False)

# ----------------------------------------------------
# 13. Save predictions and plots
# ----------------------------------------------------
out = pd.DataFrame({"y_true": y_test_level, "price_now": price_now_test})
for name, p in results.items():
    out[f"pred_{name}"] = p
out.to_csv("test_predictions_rank.csv", index=False)
print("\nSaved test_predictions_rank.csv")

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, tag in zip(axes, ("scaled", "raw")):
    ax.plot(y_test_level, label="Actual (t+20)", color="black", linewidth=2)
    ax.plot(price_now_test, label="Persistence baseline", color="gray", linestyle="--")
    for name, p in results.items():
        if name.endswith(f"[{tag}]"):
            ax.plot(p, label=name, alpha=0.85)
    ax.set_title(f"2026 test: actual vs predicted [{tag}]")
    ax.set_xlabel("Test sample index")
    ax.legend()
plt.tight_layout(); plt.savefig("test_plot_rank.png", dpi=120)
print("Saved test_plot_rank.png")