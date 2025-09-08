"""
Lidar Cloud Pixel-Level Feature Exploration
Author: Dr. Zou (pipeline scaffold prepared by GPT-5 Thinking)

Goal
----
• Use existing mask .tif as the driver (because masks are fewer) and match H5 files.
• Extract per-pixel features from provided channels + key derived features (gradients, filters, ratios).
• Build a sampled feature table with labels {0: non-cloud, 1: cloud, 2: uncertain} for EDA/class-wise stats.
• Save outputs:
  - output/feature_samples.parquet (all sampled pixels with features)
  - output/feature_stats_by_class.csv (summary statistics per class)
  - logs/skip.log (mismatches or errors)

Assumptions
-----------
• H5 datasets are 2D arrays (time × height) for each key in channel_keys.
• Mask .tif aligns with the H5 grids. If shapes differ, the mask is resized by nearest-neighbor.
• Directory layout:
    h5s_path/YYYY/MMDD/*.h5
    masks_path/YYYY/MMDD/*.tif
• Station filtering by 51628, 53845, 58847 if station IDs appear in file names. If not present, files are still matched by date folder.
• Default axes: time axis = 0, height axis = 1. Adjust TIME_AXIS/HEIGHT_AXIS if needed.

Speed/Memory Notes
------------------
• Pure NumPy + SciPy ndimage (vectorized). No Python loops over pixels.
• Optional random subsampling per class per (mask,H5) pair to keep tables manageable.
• Multiprocessing ready for per-file parallelism if needed (Windows-safe spawn).

Usage
-----
1) Adjust CONFIG below if necessary (paths, sampling, stations).
2) Run:
   python lidar_cloud_pixel_features.py
3) Check `output/feature_stats_by_class.csv` and the Parquet for EDA.
"""
from __future__ import annotations
import os
import re
import math
import json
import glob
import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from scipy.ndimage import gaussian_filter, sobel, uniform_filter, laplace
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# ======================
# ====== CONFIG ========
# ======================
# --- Paths from user ---
h5s_path = r"F:\Workspace\Projects\LidarCloud\CMA\2025"
channel_keys = ['CH355', 'CH532', 'CH1064', 'PDR355', 'PDR532']
masks_path = h5s_path.replace('CMA', 'mask')  # expects same YYYY/MMDD tree

# --- Stations of interest (optional filter in filenames) ---
STATIONS = {"51628", "53845", "58847"}

# --- Axes orientation (before transform) ---
TIME_AXIS_RAW = 0
HEIGHT_AXIS_RAW = 1
# Apply transpose+vertical flip to align with mask (Z,T)
APPLY_TRANSPOSE_FLIP = True
TIME_AXIS = 1 if APPLY_TRANSPOSE_FLIP else 0
HEIGHT_AXIS = 0 if APPLY_TRANSPOSE_FLIP else 1

# --- Range correction & normalization for CH channels ---
NORMALIZE_RANGE = {
    "51628": {"CH355": 3.948e5, "CH532": 5.169e5, "CH1064": 2.283e5},
    "53845": {"CH355": 4.158e5, "CH532": 4.750e5, "CH1064": 2.294e5},
    "58847": {"CH355": 4.547e5, "CH532": 3.911e5, "CH1064": 2.321e5},
}

# --- Sampling control ---
MAX_SAMPLES_PER_CLASS = 80000      # per (mask,h5) pair
RANDOM_SEED = 1337

# --- Output ---
OUT_DIR = "output"
LOG_DIR = "logs"
TABLE_PATH = os.path.join(OUT_DIR, "feature_samples.parquet")
STATS_PATH = os.path.join(OUT_DIR, "feature_stats_by_class.csv")
KS_PATH = os.path.join(OUT_DIR, "ks_scores.csv")
HISTS_PATH = os.path.join(OUT_DIR, "hists_top8_label01.png")
SKIP_LOG = os.path.join(LOG_DIR, "skip.log")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# =========================
# ====== UTILITIES ========
# =========================
_station_re = re.compile(r"(\d{5})")  # capture 5-digit station ID anywhere in name
_ts_re = re.compile(r"(\d{8})(?:[_-]?(\d{4,6}))?")  # date YYYYMMDD optionally time


def _read_mask(mask_path: str) -> np.ndarray:
    """Read mask tif as uint8/uint16 -> int array in {0,1,2}.
    Returns shape (T, Z) to match H5 orientation assumption.
    """
    img = Image.open(mask_path)
    arr = np.array(img)
    if arr.ndim == 3:
        # use first channel if RGB accidentally
        arr = arr[..., 0]
    # ensure integer
    arr = arr.astype(np.int16)
    return arr


def _get_station_from_name(path: str) -> Optional[str]:
    base = os.path.basename(path)
    m = _station_re.search(base)
    if m:
        return m.group(1)
    return None


def _get_date_from_parents(path: str) -> Optional[Tuple[str, str]]:
    """Infer (YYYY, MMDD) from the parent folders.
    Expected: .../YYYY/MMDD/filename
    """
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        return None
    yyyy, mmdd = parts[-3], parts[-2]
    if len(yyyy) == 4 and len(mmdd) == 4 and yyyy.isdigit() and mmdd.isdigit():
        return yyyy, mmdd
    return None


def _list_mask_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "*", "*", "*.tif")))


def _list_h5_candidates(h5_root: str, yyyy: str, mmdd: str) -> List[str]:
    return sorted(glob.glob(os.path.join(h5_root, yyyy, mmdd, "*.h5")))


def _choose_h5_for_mask(mask_path: str, h5_candidates: List[str]) -> Optional[str]:
    """Pick H5 file matching by station ID if possible; otherwise first candidate.
    Heuristic: match station; if multiple, try closest timestamp parsed from name.
    """
    if not h5_candidates:
        return None
    mask_station = _get_station_from_name(mask_path)

    def _parse_ts(p: str) -> Optional[datetime]:
        base = os.path.basename(p)
        m = _ts_re.search(base)
        if not m:
            return None
        ymd, hm = m.group(1), m.group(2)
        try:
            if hm:
                # allow HHMM or HHMMSS
                if len(hm) == 4:
                    return datetime.strptime(ymd + hm, "%Y%m%d%H%M")
                elif len(hm) == 6:
                    return datetime.strptime(ymd + hm, "%Y%m%d%H%M%S")
            return datetime.strptime(ymd, "%Y%m%d")
        except Exception:
            return None

    # filter by station first
    cand = h5_candidates
    if mask_station and any(mask_station in os.path.basename(x) for x in cand):
        cand = [x for x in cand if mask_station in os.path.basename(x)]
        if not cand:
            cand = h5_candidates

    # if multiple, choose by closest timestamp to mask name's timestamp (if any)
    mask_ts = _parse_ts(mask_path)
    if mask_ts:
        cand_with_dt = [(p, _parse_ts(p)) for p in cand]
        cand_with_dt = [(p, dt) for (p, dt) in cand_with_dt if dt is not None]
        if cand_with_dt:
            p_best = min(cand_with_dt, key=lambda kv: abs((kv[1] - mask_ts).total_seconds()))[0]
            return p_best

    return cand[0]


# ==========================
# ====== FEATURES ==========
# ==========================

@dataclass
class FeatureConfig:
    keys: List[str]
    time_axis: int = TIME_AXIS
    height_axis: int = HEIGHT_AXIS
    # window for local stats
    local_win: int = 5
    # gaussian sigmas
    sigma1: float = 1.0
    sigma2: float = 2.0


def safe_log1p(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, a_min=0.0, a_max=None)  # lidar backscatter is non-negative in general
    return np.log1p(x)


def robust_scale(x: np.ndarray) -> np.ndarray:
    """Median / MAD scaling (robust z-score)."""
    x = x.astype(np.float32)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad <= 1e-8:
        return (x - med)
    return (x - med) / (1.4826 * mad)


def local_mean_var(x: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    mu = uniform_filter(x, size=size)
    mu2 = uniform_filter(x * x, size=size)
    var = np.maximum(mu2 - mu * mu, 0.0)
    return mu, var


def derive_single_channel_features(name: str, arr: np.ndarray, cfg: FeatureConfig) -> Dict[str, np.ndarray]:
    """Compute key 2D features for one channel.
    Returns a dict of feature_name -> 2D array (same shape as arr).
    """
    feat = {}
    X = safe_log1p(arr)
    Xr = robust_scale(X)

    # Gaussian blur and Difference of Gaussians
    g1 = gaussian_filter(Xr, sigma=cfg.sigma1)
    g2 = gaussian_filter(Xr, sigma=cfg.sigma2)
    dog = g1 - g2

    # Sobel along time & height (axis order matters)
    # sobel operates per-axis when axis is specified
    sob_t = sobel(Xr, axis=cfg.time_axis)
    sob_z = sobel(Xr, axis=cfg.height_axis)
    grad_mag = np.hypot(sob_t, sob_z)

    # Laplacian (second-order)
    lap = laplace(Xr)

    # Local statistics
    mu, var = local_mean_var(Xr, size=cfg.local_win)
    std = np.sqrt(var + 1e-8)
    local_contrast = (Xr - mu) / (std + 1e-6)

    # Pack
    #feat[f"{name}_log1p"] = X
    feat[f"{name}_arr"] = arr
    feat[f"{name}_robust"] = Xr
    feat[f"{name}_gauss_s{cfg.sigma1}"] = g1
    feat[f"{name}_dog_{cfg.sigma1}_{cfg.sigma2}"] = dog
    feat[f"{name}_sob_t"] = sob_t
    feat[f"{name}_sob_z"] = sob_z
    feat[f"{name}_gradmag"] = grad_mag
    feat[f"{name}_laplace"] = lap
    feat[f"{name}_lmean_w{cfg.local_win}"] = mu
    feat[f"{name}_lstd_w{cfg.local_win}"] = std
    feat[f"{name}_lcontrast_w{cfg.local_win}"] = local_contrast
    return feat


def derive_cross_channel_features(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Cross-channel color ratios and depol combinations (safe divisions)."""
    out = {}

    def sdiv(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=(b != 0))

    ch355 = data.get('CH355')
    ch532 = data.get('CH532')
    ch1064 = data.get('CH1064')
    pdr355 = data.get('PDR355')
    pdr532 = data.get('PDR532')

    if ch355 is not None and ch532 is not None:
        out['ratio_532_355'] = sdiv(ch532, ch355)
        out['logratio_532_355'] = safe_log1p(out['ratio_532_355'])
    if ch532 is not None and ch1064 is not None:
        out['ratio_1064_532'] = sdiv(ch1064, ch532)
        out['logratio_1064_532'] = safe_log1p(out['ratio_1064_532'])
    if ch355 is not None and ch1064 is not None:
        out['ratio_1064_355'] = sdiv(ch1064, ch355)
        out['logratio_1064_355'] = safe_log1p(out['ratio_1064_355'])

    # Depol × intensity (can emphasize ice/edges)
    if pdr532 is not None and ch532 is not None:
        out['pdr532_x_ch532'] = pdr532 * safe_log1p(ch532)
    if pdr355 is not None and ch355 is not None:
        out['pdr355_x_ch355'] = pdr355 * safe_log1p(ch355)

    return out


# ==========================
# ====== PIPELINE ==========
# ==========================

# ---------- Height helpers ----------
_DEF_HEIGHT_KEYS = [
    'height', 'range', 'altitude', 'HEIGHT', 'RANGE', 'ALTITUDE', 'z', 'Z',
]

def _read_height_vector(f: h5py.File, height_len: int) -> np.ndarray:
    for k in _DEF_HEIGHT_KEYS:
        if k in f:
            h = np.array(f[k][...]).squeeze()
            if h.ndim == 1 and h.size == height_len:
                h = h.astype(np.float32)
                h = np.maximum(h, 0.0)
                return h
    return np.arange(height_len, dtype=np.float32)


def _read_h5_channels_with_corrections(h5_path: str, keys: List[str], station: Optional[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Read channels, apply CH range^2 correction + station normalization, then transpose+flip to (Z,T)."""
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, 'r') as f:
        avail = [k for k in keys if k in f]
        if not avail:
            return {}, np.array([])
        probe = f[avail[0]][...]
        if probe.ndim != 2:
            raise ValueError(f"Dataset {avail[0]} in {h5_path} is not 2D.")
        T = probe.shape[TIME_AXIS_RAW]
        Z = probe.shape[HEIGHT_AXIS_RAW]
        height = _read_height_vector(f, Z).astype(np.float32)
        height_sq = np.square(height)
        for k in avail:
            arr = np.array(f[k][...], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Dataset {k} in {h5_path} is not 2D.")
            if k.startswith('CH'):
                if HEIGHT_AXIS_RAW == 1:
                    arr = arr * height_sq[None, :]
                else:
                    arr = arr * height_sq[:, None]
                max_map = NORMALIZE_RANGE.get(station or "", {})
                max_val = max_map.get(k)
                if max_val and max_val > 0:
                    arr = arr / float(max_val)
            if arr.ndim == 2:
                arr = np.transpose(arr, (1, 0))[::-1, :]
            out[k] = arr
        h_after = height[::-1] if APPLY_TRANSPOSE_FLIP else height
        return out, h_after

def _compute_stats_and_ks(table_path: str,
                          top_k: int = 20,
                          top_m_cv: int = 40,
                          min_samples_per_class: int = 100,
                          cv_folds: int = 3,
                          random_state: int = 1337):
    """Compute per-class stats; rank features by separability using KS, symmetric KL, and JSD.
    Additionally evaluate 3-feature combinations using LDA/Mahalanobis criterion and (optionally) CV-AUC.
    Also plot overlaid *normalized* histograms (probability densities) for the top-8 by JSD,
    and a 3D scatter for the best triplet.

    Parameters
    ----------
    table_path : str
        Input parquet path.
    top_k : int
        Number of univariate top features (by JSD) to consider when enumerating triplets.
    top_m_cv : int
        Number of best triplets (by LDA score) to further evaluate with CV-AUC.
    min_samples_per_class : int
        Minimum samples required in each class for a feature or triplet to be considered.
    cv_folds : int
        Folds for StratifiedKFold used in CV-AUC on triplets.
    random_state : int
        RNG seed for reproducibility.
    """
    import os, math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    from itertools import combinations

    # Optional but robust imports for multivariate evaluation
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # ---------- Load & basic filtering ----------
    full = pd.read_parquet(table_path)
    full = full[full['label'].isin([0, 1])]

    feature_cols = [c for c in full.columns if c not in {'label', 'station', 'yyyy', 'mmdd', 'mask_file', 'h5_file'}]
    feature_cols = [c for c in feature_cols if np.issubdtype(full[c].dtype, np.number)]

    # ---------- Class-wise summary stats ----------
    def summarize(group: pd.DataFrame) -> pd.DataFrame:
        desc = group[feature_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
        desc = desc.rename(columns={'50%': 'p50', '25%': 'p25', '75%': 'p75', '5%': 'p05', '95%': 'p95'})
        keep = ['count', 'mean', 'std', 'min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max']
        return desc[keep]

    stats = full.groupby('label', as_index=True).apply(summarize)
    stats.index = [f"label{lbl}:{name}" for lbl, name in stats.index]
    stats.to_csv(STATS_PATH)

    x0 = full[full['label'] == 0]
    x1 = full[full['label'] == 1]

    # ---------- Divergence helpers (histogram-based) ----------
    def _pmf(a: np.ndarray, bins: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        cnt, _ = np.histogram(a, bins=bins, density=False)
        p = cnt.astype(np.float64) + eps
        s = p.sum()
        return p / s if s > 0 else np.full_like(p, 1.0 / len(p), dtype=np.float64)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    # ---------- Univariate separability ----------
    rows_uni = []
    for col in feature_cols:
        a = x0[col].values
        b = x1[col].values
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < min_samples_per_class or len(b) < min_samples_per_class:
            continue
        # KS on raw samples
        ks = ks_2samp(a, b, alternative='two-sided', mode='auto')

        # Common bins based on robust percentiles
        try:
            pmin = np.nanmin([np.percentile(a, 1), np.percentile(b, 1)])
            pmax = np.nanmax([np.percentile(a, 99), np.percentile(b, 99)])
        except Exception:
            pmin, pmax = np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b]))
        if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin >= pmax:
            continue
        bins = np.linspace(pmin, pmax, 64)

        # PMFs + divergences
        p = _pmf(a, bins)
        q = _pmf(b, bins)
        kl_pq = _kl(p, q)
        kl_qp = _kl(q, p)
        sym_kl = 0.5 * (kl_pq + kl_qp)
        jsd = _jsd(p, q)
        # entropies (nats)
        H0 = float(-np.sum(p * np.log(p)))
        H1 = float(-np.sum(q * np.log(q)))

        rows_uni.append((col, float(ks.statistic), float(ks.pvalue), sym_kl, kl_pq, kl_qp, jsd, H0, H1))

    score_cols = ['feature', 'ks_stat', 'ks_pvalue', 'sym_kl', 'kl_pq', 'kl_qp', 'jsd', 'H0', 'H1']
    score_df = pd.DataFrame(rows_uni, columns=score_cols)
    if not score_df.empty:
        score_df = score_df.sort_values(['jsd', 'ks_stat'], ascending=[False, False])
    score_df.to_csv(KS_PATH, index=False)

    # ---------- Plot: overlaid histograms for top-8 by JSD ----------
    top = score_df.head(8)
    n = len(top)
    if n > 0:
        cols = 2
        rows_n = math.ceil(n / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(10, 3.2 * rows_n), constrained_layout=True)
        if rows_n == 1:
            axes = np.array(axes).reshape(1, -1)
        for i, row in enumerate(top.itertuples(index=False)):
            fname, ks_val, pv, symkl, klpq, klqp, jsd_val, H0, H1 = row
            r, c = divmod(i, cols)
            ax = axes[r, c]
            a = x0[fname].values
            b = x1[fname].values
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            try:
                pmin = np.nanmin([np.percentile(a, 1), np.percentile(b, 1)])
                pmax = np.nanmax([np.percentile(a, 99), np.percentile(b, 99)])
            except Exception:
                pmin, pmax = np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b]))
            if not np.isfinite(pmin) or not np.isfinite(pmax) or pmin >= pmax:
                pmin, pmax = np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b]))
            bins = np.linspace(pmin, pmax, 40)
            ax.hist(a, bins=bins, density=True, alpha=0.5, label='label=0')
            ax.hist(b, bins=bins, density=True, alpha=0.5, label='label=1')
            ax.set_title(f"{fname} | KS={ks_val:.3f} | SymKL={symkl:.3f} | JSD={jsd_val:.3f}")
            ax.grid(True, ls=':', alpha=0.4)
            if i % cols == 0:
                ax.set_ylabel('prob. density')
            ax.legend()
        for j in range(n, rows_n * cols):
            r, c = divmod(j, cols)
            fig.delaxes(axes[r, c])
        fig.suptitle('Top-8 Features by JSD (label 0 vs 1)', y=1.02)
        fig.savefig(HISTS_PATH, dpi=180)
        plt.close(fig)

    # ---------- Multivariate separability: 3-feature combos ----------
    # Candidate pool: top_k by JSD
    if score_df.empty or len(score_df) < 3:
        return  # not enough features for triplets

    cand_feats = score_df['feature'].head(max(3, top_k)).tolist()

    # Pre-split classes for speed
    def _dropna_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        return df[cols + ['label']].replace([np.inf, -np.inf], np.nan).dropna()

    # LDA/Mahalanobis score on a triplet
    def lda_score_triplet(df01: pd.DataFrame, f1: str, f2: str, f3: str) -> tuple:
        cols = [f1, f2, f3]
        sub = _dropna_cols(df01, cols)
        sub0 = sub[sub['label'] == 0]
        sub1 = sub[sub['label'] == 1]
        if len(sub0) < min_samples_per_class or len(sub1) < min_samples_per_class:
            return None  # insufficient
        X0 = sub0[cols].values
        X1 = sub1[cols].values
        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)
        # Pooled within-class covariance with shrinkage
        # (fit on centered data per class to get Sw)
        X0c = X0 - mu0
        X1c = X1 - mu1
        Xc = np.vstack([X0c, X1c])
        cov = LedoitWolf().fit(Xc).covariance_
        # Stabilize if needed
        try:
            # Solve Sw^{-1} (mu0 - mu1) via linear solve
            dmu = (mu0 - mu1)
            w = np.linalg.solve(cov, dmu)
            lda_j = float(dmu @ w)
        except np.linalg.LinAlgError:
            # Add tiny ridge
            cov = cov + 1e-6 * np.eye(cov.shape[0])
            w = np.linalg.solve(cov, dmu)
            lda_j = float(dmu @ w)
        n_use = len(sub)  # samples used
        return lda_j, n_use

    triplet_rows = []
    df01 = full[['label'] + cand_feats].copy()
    for f1, f2, f3 in combinations(cand_feats, 3):
        res = lda_score_triplet(df01, f1, f2, f3)
        if res is None:
            continue
        lda_j, n_use = res
        triplet_rows.append((f1, f2, f3, lda_j, n_use))

    if not triplet_rows:
        # No valid triplets (e.g., too many NaNs or too few samples)
        return

    trip_cols = ['f1', 'f2', 'f3', 'lda_score', 'n_samples']
    trip_df = pd.DataFrame(triplet_rows, columns=trip_cols)
    trip_df = trip_df.sort_values('lda_score', ascending=False).reset_index(drop=True)

    # ---------- Optional CV-AUC on top-M by LDA ----------
    top_cv = trip_df.head(max(1, min(top_m_cv, len(trip_df)))).copy()
    auc_list = []
    for row in top_cv.itertuples(index=False):
        f1, f2, f3, lda_j, n_use = row
        cols = [f1, f2, f3]
        sub = df01[cols + ['label']].replace([np.inf, -np.inf], np.nan).dropna()
        y = sub['label'].values.astype(int)
        X = sub[cols].values
        if len(np.unique(y)) < 2 or len(y) < (cv_folds * 2):
            auc_list.append(np.nan)
            continue
        # Pipeline: Standardize + Logistic Regression
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=random_state)
        )
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        aucs = []
        for tr, te in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
        auc_list.append(float(np.mean(aucs)))
    top_cv['cv_auc'] = auc_list

    # Merge cv_auc back (only for those evaluated)
    trip_df = trip_df.merge(
        top_cv[['f1', 'f2', 'f3', 'cv_auc']],
        on=['f1', 'f2', 'f3'],
        how='left'
    )

    TRIPLETS_PATH = os.path.join(OUT_DIR, 'triplets_rank.csv')
    trip_df.to_csv(TRIPLETS_PATH, index=False)

    # ---------- 3D scatter for best triplet ----------
    # Prefer best by cv_auc, fallback to lda_score
    best = trip_df.sort_values(
        by=['cv_auc', 'lda_score'],
        ascending=[False, False],
        na_position='last'
    ).iloc[0]
    f1, f2, f3 = best['f1'], best['f2'], best['f3']
    sub = full[['label', f1, f2, f3]].replace([np.inf, -np.inf], np.nan).dropna()
    # Clip extremes for visualization
    for f in (f1, f2, f3):
        lo, hi = np.percentile(sub[f], 1), np.percentile(sub[f], 99)
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            sub[f] = np.clip(sub[f], lo, hi)

    max_points = 4000
    per_cls = max_points // 2
    sub0 = sub[sub['label'] == 0]
    sub1 = sub[sub['label'] == 1]
    if len(sub0) > per_cls:
        sub0 = sub0.sample(per_cls, random_state=random_state)
    if len(sub1) > per_cls:
        sub1 = sub1.sample(per_cls, random_state=random_state)

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sub0[f1], sub0[f2], sub0[f3], s=2, alpha=0.35, label='label=0')
    ax.scatter(sub1[f1], sub1[f2], sub1[f3], s=2, alpha=0.35, label='label=1')
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)
    ttl_auc = best['cv_auc'] if pd.notnull(best['cv_auc']) else float('nan')
    ax.set_title(f'Best Triplet: {f1}, {f2}, {f3} | LDA={best["lda_score"]:.3f} | AUC={ttl_auc:.3f}')
    ax.legend(loc='best')
    scatter_path = os.path.join(OUT_DIR, 'scatter3d_best_triplet.png')
    fig.savefig(scatter_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    # ---------- (Optional) Keep your entropy-based 3D scatter ----------
    if not score_df.empty:
        score_df['Havg'] = 0.5 * (score_df['H0'] + score_df['H1'])
        top3 = score_df.sort_values('Havg', ascending=False).head(3)
        if len(top3) == 3:
            f1e, f2e, f3e = top3['feature'].tolist()
            cols_needed = ['label', f1e, f2e, f3e]
            sub = full[cols_needed].replace([np.inf, -np.inf], np.nan).dropna()
            for f in (f1e, f2e, f3e):
                lo, hi = np.percentile(sub[f], 1), np.percentile(sub[f], 99)
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    sub[f] = np.clip(sub[f], lo, hi)
            max_points = 4000
            per_cls = max_points // 2
            sub0 = sub[sub['label'] == 0]
            sub1 = sub[sub['label'] == 1]
            if len(sub0) > per_cls:
                sub0 = sub0.sample(per_cls, random_state=random_state)
            if len(sub1) > per_cls:
                sub1 = sub1.sample(per_cls, random_state=random_state)
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(sub0[f1e], sub0[f2e], sub0[f3e], s=2, alpha=0.35, label='label=0')
            ax.scatter(sub1[f1e], sub1[f2e], sub1[f3e], s=2, alpha=0.35, label='label=1')
            ax.set_xlabel(f1e)
            ax.set_ylabel(f2e)
            ax.set_zlabel(f3e)
            ax.set_title(f'3D Scatter by Top-3 Entropy Features: {f1e}, {f2e}, {f3e}')
            ax.legend(loc='best')
            scatter_path2 = os.path.join(OUT_DIR, 'scatter3d_top3_entropy.png')
            fig.savefig(scatter_path2, dpi=180, bbox_inches='tight')
            plt.close(fig)




def _resize_mask_if_needed(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    # nearest-neighbor resize via PIL; ensure (width,height) order
    # current arrays are (T, Z); we map to (width,height)=(Z,T)
    img = Image.fromarray(mask.astype(np.int16))
    img = img.resize((target_shape[1], target_shape[0]), resample=Image.NEAREST)
    arr = np.array(img).astype(np.int16)
    return arr


def _read_h5_channels(h5_path: str, keys: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, 'r') as f:
        for k in keys:
            if k in f:
                arr = f[k][...]
                if arr.ndim != 2:
                    raise ValueError(f"Dataset {k} in {h5_path} is not 2D.")
                out[k] = arr.astype(np.float32)
            else:
                # silently skip missing keys
                pass
    return out


def extract_features_for_pair(mask_path: str, h5_path: str, cfg: FeatureConfig,
                              max_samples_per_class: int = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    # read mask and channels
    mask = _read_mask(mask_path)
    station = _get_station_from_name(mask_path) or _get_station_from_name(h5_path) or "unknown"
    ch_data, h_vec = _read_h5_channels_with_corrections(h5_path, cfg.keys, station)
    if not ch_data:
        raise RuntimeError(f"No requested channels found in {h5_path}")

    # Ensure shapes consistent
    any_arr = next(iter(ch_data.values()))
    mask = _resize_mask_if_needed(mask, target_shape=any_arr.shape)

    # Cross-channel features use corrected arrays
    cross = derive_cross_channel_features(ch_data)

    # Derive per-channel features
    derived: Dict[str, np.ndarray] = {}
    for k, arr in ch_data.items():
        derived.update(derive_single_channel_features(k, arr, cfg))

    # Merge everything (including original channels as log/robust already present)
    all_feat_maps: Dict[str, np.ndarray] = {}
    all_feat_maps.update(derived)
    all_feat_maps.update(cross)

    # Flatten
    labels = mask.reshape(-1)
    valid = (labels == 0) | (labels == 1)

    # Build a dict of flattened features
    flat_features = {}
    for name, arr in all_feat_maps.items():
        if arr.shape != mask.shape:
            # safety: align by resizing feature map (rare)
            arr = _resize_mask_if_needed(arr, target_shape=mask.shape)
        flat_features[name] = arr.reshape(-1)

    # Stack into DataFrame (only valid labels)
    df = pd.DataFrame({**flat_features, 'label': labels})
    df = df[valid]

    # Subsample per class to balance & limit size
    out_parts = []
    for cls in (0, 1):
        part = df[df['label'] == cls]
        if len(part) == 0:
            continue
        if len(part) > max_samples_per_class:
            idx = np.random.choice(part.index.values, size=max_samples_per_class, replace=False)
            part = part.loc[idx]
        out_parts.append(part)
    if not out_parts:
        return pd.DataFrame()

    out_df = pd.concat(out_parts, axis=0, ignore_index=True)

    # Add meta
    yyyy_mmdd = _get_date_from_parents(mask_path)
    station = _get_station_from_name(mask_path) or _get_station_from_name(h5_path) or "unknown"
    out_df['station'] = station
    if yyyy_mmdd:
        out_df['yyyy'] = yyyy_mmdd[0]
        out_df['mmdd'] = yyyy_mmdd[1]
    out_df['mask_file'] = os.path.basename(mask_path)
    out_df['h5_file'] = os.path.basename(h5_path)
    return out_df


def _append_parquet(df: pd.DataFrame, path: str):
    # Append by reading old then concat (simple & portable); for large-scale, consider pyarrow parquet appends.
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)


def run_pipeline():
    cfg = FeatureConfig(keys=channel_keys)
    mask_files = _list_mask_files(masks_path)

    if not mask_files:
        print(f"No mask files found under: {masks_path}")
        return

    # Process each mask file
    n_pairs = 0
    with open(SKIP_LOG, 'w', encoding='utf-8') as flog:
        for mask_path in mask_files[:5]:
            # station filter if station explicitly in filename
            station = _get_station_from_name(mask_path)
            if station and STATIONS and (station not in STATIONS):
                continue

            ymd = _get_date_from_parents(mask_path)
            if not ymd:
                flog.write(f"[SKIP] Cannot infer YYYY/MMDD from: {mask_path}\n")
                continue
            yyyy, mmdd = ymd
            candidates = _list_h5_candidates(h5s_path, yyyy, mmdd)
            if not candidates:
                flog.write(f"[SKIP] No H5 candidates for date {yyyy}/{mmdd} for mask: {mask_path}\n")
                continue

            h5_path = _choose_h5_for_mask(mask_path, candidates)
            if h5_path is None:
                flog.write(f"[SKIP] Cannot select H5 for mask: {mask_path}\n")
                continue

            try:
                df = extract_features_for_pair(mask_path, h5_path, cfg, max_samples_per_class=MAX_SAMPLES_PER_CLASS)
                if df.empty:
                    flog.write(f"[SKIP] Empty features for pair: {mask_path} | {h5_path}\n")
                    continue
                # Persist incrementally to avoid huge memory
                _append_parquet(df, TABLE_PATH)
                n_pairs += 1
                print(f"OK: {os.path.basename(mask_path)} -> {os.path.basename(h5_path)} | +{len(df)} rows")
            except Exception as e:
                flog.write(f"[ERR] {mask_path} | {h5_path} | {repr(e)}\n")

    if n_pairs == 0:
        print("No valid (mask, H5) pairs processed. See logs/skip.log for details.")
        return

    # KS & stats on the aggregated table
    _compute_stats_and_ks(TABLE_PATH)

    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved stats: {STATS_PATH}")
    print(f"Saved KS: {KS_PATH}")
    print(f"Saved histograms: {HISTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
