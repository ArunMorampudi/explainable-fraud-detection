"""
Explainable Fraud Detection on the UCI Credit Card Fraud Dataset.

This script implements the full experimental pipeline used in the paper,
including data preprocessing, model training, threshold selection under
fixed false-positive constraints, evaluation for extreme class imbalance,
and post-hoc explainability analysis.
"""


from __future__ import annotations

import os
import json
import time
import shutil
import zipfile
import numpy as np
import pandas as pd

# Force non-interactive matplotlib backend to avoid tkinter threading issues in multiprocessing
import matplotlib
matplotlib.use('Agg')

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)

RANDOM_SEED = 42


# -----------------------------
# Data download utility
# -----------------------------
def ensure_creditcard_csv(csv_filename: str = "data/raw/creditcard.csv") -> str:
    """
    Ensure creditcard.csv exists. If not, download from Kaggle using kagglehub.
    Returns the path to the CSV file.
    """
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    if os.path.isfile(csv_filename):
        print(f"[INFO] Found existing {csv_filename}")
        return csv_filename
    
    print(f"[INFO] Dataset not found locally; downloading public benchmark dataset.")
    try:
        import kagglehub
        
        # Download latest version
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print("Original path to dataset files:", path)

        # Determine destination (script directory or current working directory)
        try:
            script_dir = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            script_dir = os.getcwd()

        dest_dir = os.path.join(script_dir, "data", "raw")
        os.makedirs(dest_dir, exist_ok=True)

        def copy_dir_contents(src_dir, dest_dir):
            for item in os.listdir(src_dir):
                s = os.path.join(src_dir, item)
                d = os.path.join(dest_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

        if os.path.isfile(path) and path.lower().endswith('.zip'):
            print("Detected zip file; extracting to:", dest_dir)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(dest_dir)
            print("Extraction complete.")
        elif os.path.isdir(path):
            print("Detected directory; copying contents to:", dest_dir)
            copy_dir_contents(path, dest_dir)
            print("Copy complete.")
        else:
            # Unknown file type: copy the file into the dest dir
            dest = os.path.join(dest_dir, os.path.basename(path))
            shutil.copy2(path, dest)
            print("Copied file to:", dest)

        print("Dataset files are now saved in:", dest_dir)
        
        if os.path.isfile(csv_filename):
            print(f"[INFO] Successfully downloaded and located {csv_filename}")
            return csv_filename
        else:
            raise FileNotFoundError(f"CSV file not found after download. Expected: {csv_filename}")
    
    except ImportError:
        raise ImportError("kagglehub is required to download the dataset. Install it with: pip install kagglehub")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")


# -----------------------------
# Utilities: metrics + thresholding
# -----------------------------
def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) 


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    # Precision among top-k highest-risk predictions
    k = int(k)
    idx = np.argsort(-y_prob)[:k]
    return float(np.mean(y_true[idx])) if k > 0 else 0.0


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """
    Returns (recall, threshold) at or below target FPR.
    Uses ROC curve thresholds.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    # Find thresholds where FPR <= target
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        # No threshold achieves such low FPR; fallback to highest threshold
        return float(tpr[0]), float(thr[0])
    i = ok[-1]
    return float(tpr[i]), float(thr[i])


def fbeta_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> Tuple[float, float]:
    """
    Choose threshold that maximizes F_beta on validation.
    Returns (best_fbeta, threshold).
    """
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns thr of length n-1
    # Align arrays
    precision = precision[:-1]
    recall = recall[:-1]
    thr = thr

    beta2 = beta * beta
    denom = (beta2 * precision + recall)
    denom = np.where(denom == 0, 1e-12, denom)
    fbeta = (1 + beta2) * (precision * recall) / denom

    j = int(np.argmax(fbeta))
    return float(fbeta[j]), float(thr[j])


def confusion_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


# -----------------------------
# Data loading + splitting
# -----------------------------
@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray


def load_creditcard_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # basic checks
    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns {required}. Found: {df.columns.tolist()[:10]} ...")
    return df


def time_based_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15) -> SplitData:
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    y_train = train_df["Class"].to_numpy(dtype=int)
    y_val = val_df["Class"].to_numpy(dtype=int)
    y_test = test_df["Class"].to_numpy(dtype=int)

    X_train = train_df.drop(columns=["Class"])
    X_val = val_df.drop(columns=["Class"])
    X_test = test_df.drop(columns=["Class"])

    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def stratified_split(df: pd.DataFrame, train_size=0.70, val_size=0.15, seed=RANDOM_SEED) -> SplitData:
    X = df.drop(columns=["Class"])
    y = df["Class"].to_numpy(dtype=int)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed
    )
    # val fraction relative to remaining
    val_rel = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, train_size=val_rel, stratify=y_tmp, random_state=seed
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


# -----------------------------
# Models
# -----------------------------
def build_models(y_train: np.ndarray) -> Dict[str, object]:
    # imbalance ratio
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = (n_neg / max(n_pos, 1))

    models: Dict[str, object] = {}

    # Baseline 1: Logistic Regression (with scaling)
    models["logreg"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=None
            )),
        ]
    )

    # Baseline 2: Random Forest (no scaling needed, but fine if you keep consistent)
    models["rf"] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1,
    )

    # HistGradientBoosting
    # Tuned for rare-event detection under extreme class imbalance.
    # sample_weight will be applied in fit_model() for this model only.
    models["hgb"] = HistGradientBoostingClassifier(
        random_state=RANDOM_SEED,
        max_iter=600,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=64,
        min_samples_leaf=20,
        l2_regularization=0.1,
    )

    try:
        import xgboost as xgb  # type: ignore
        models["xgb"] = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
    except Exception:
        pass

    return models


# -----------------------------
# Training / evaluation
# -----------------------------
def fit_model(model, X_train: pd.DataFrame, y_train: np.ndarray, sample_weight: Optional[np.ndarray] = None):
    """
    Fit a model with optional sample_weight.
    sample_weight is only used for HistGradientBoostingClassifier to handle extreme imbalance.
    """
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    # fallback to decision_function if needed.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # convert to (0,1) via sigmoid
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Model has neither predict_proba nor decision_function")


def evaluate_all(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    k_list: List[int] = [100, 500, 1000],
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "pr_auc": pr_auc(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
    }
    # confusion-derived metrics at threshold
    cm = confusion_at_threshold(y_true, y_prob, threshold)
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    out.update({
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tn": float(tn),
    })
    # Precision@K
    for k in k_list:
        out[f"precision@{k}"] = precision_at_k(y_true, y_prob, k)
    return out


# -----
# Sample weight computation (HGB only, for extreme imbalance handling)
# -----
def compute_sample_weight_for_imbalance(y_train: np.ndarray) -> np.ndarray:
    """
    Compute sample weights for HGB: weight_positive = n_neg / n_pos, weight_negative = 1.0.
    This ensures the positive class has appropriate emphasis under extreme imbalance.
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    weight_pos = n_neg / max(n_pos, 1)  # scale positive samples
    weight_neg = 1.0  # keep negative samples at baseline
    
    sample_weight = np.where(y_train == 1, weight_pos, weight_neg)
    return sample_weight


def choose_threshold_validation(
    y_val: np.ndarray,
    p_val: np.ndarray,
    mode: str = "fpr_budget",
    target_fpr: float = 0.01,
    beta: float = 2.0,
) -> Tuple[float, Dict[str, float]]:
    """
    mode:
      - "fpr_budget": choose highest recall with FPR <= target_fpr
      - "fbeta": choose threshold maximizing F_beta
    """
    if mode == "fpr_budget":
        rec, thr = recall_at_fpr(y_val, p_val, target_fpr)
        details = {"mode": "fpr_budget", "target_fpr": float(target_fpr), "val_recall_at_target_fpr": float(rec)}
        return float(thr), details

    if mode == "fbeta":
        best_f, thr = fbeta_best_threshold(y_val, p_val, beta=beta)
        details = {"mode": "fbeta", "beta": float(beta), "val_best_fbeta": float(best_f)}
        return float(thr), details

    raise ValueError("Unknown thresholding mode. Use 'fpr_budget' or 'fbeta'.")


# -----------------------------
# Statistical robustness: bootstrap CI on test set + repeated CV on training
# -----------------------------
def bootstrap_test_metrics(
    y_test: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> Dict[str, Dict[str, float]]:
    """
    Stratified bootstrap (by class) of (y_test, y_scores) pairs.
    Returns dict of metrics -> {mean, std, ci_lower (2.5%), ci_upper (97.5%)}.
    Does NOT retrain models; resamples scores and labels with replacement
    preserving class counts in each bootstrap sample.
    """
    rng = np.random.RandomState(seed)

    idx_pos = np.where(y_test == 1)[0]
    idx_neg = np.where(y_test == 0)[0]
    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    metrics = {"pr_auc": [], "roc_auc": [], "precision": [], "recall": []}

    for i in tqdm(range(int(n_boot)), desc="Bootstrap resamples", unit="sample", leave=False):
        samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        samp_idx = np.concatenate([samp_pos, samp_neg])

        y_s = y_test[samp_idx]
        s_scores = y_scores[samp_idx]

        metrics["pr_auc"].append(pr_auc(y_s, s_scores))
        metrics["roc_auc"].append(roc_auc(y_s, s_scores))

        cm = confusion_at_threshold(y_s, s_scores, threshold)
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

        metrics["precision"].append(float(prec))
        metrics["recall"].append(float(rec))

    out: Dict[str, Dict[str, float]] = {}
    for k, v in metrics.items():
        arr = np.array(v)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        out[k] = {"mean": mean, "std": std, "ci_lower": ci_lower, "ci_upper": ci_upper}

    return out


def run_repeated_cv(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    """
    Perform repeated stratified K-fold CV on the TRAINING portion only.
    For each fold we instantiate a fresh model (via build_models) using
    the fold's training labels so that class imbalance params remain consistent.
    Returns aggregated PR-AUC and ROC-AUC as mean and std across folds.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    pr_list: List[float] = []
    roc_list: List[float] = []

    # Ensure DataFrame for consistent indexing
    X_df = X_train.reset_index(drop=True) if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_arr = np.array(y_train)

    fold_splits = list(rskf.split(X_df, y_arr))
    for train_idx, val_idx in tqdm(fold_splits, desc=f"CV folds ({model_name})", unit="fold", leave=False):
        X_tr = X_df.iloc[train_idx]
        X_va = X_df.iloc[val_idx]
        y_tr = y_arr[train_idx]
        y_va = y_arr[val_idx]

        # Build fresh models for this fold (respecting fold-specific imbalance)
        models = build_models(y_tr)
        if model_name not in models:
            continue

        model = models[model_name]
        # Compute sample_weight only for HGB (imbalance handling)
        sample_weight = None
        if model_name == "hgb":
            sample_weight = compute_sample_weight_for_imbalance(y_tr)
        fit_model(model, X_tr, y_tr, sample_weight=sample_weight)

        p_val = predict_proba(model, X_va)
        pr_list.append(pr_auc(y_va, p_val))
        roc_list.append(roc_auc(y_va, p_val))

    pr_arr = np.array(pr_list)
    roc_arr = np.array(roc_list)

    return {
        "pr_auc_mean": float(pr_arr.mean()) if pr_arr.size > 0 else 0.0,
        "pr_auc_std": float(pr_arr.std(ddof=1)) if pr_arr.size > 1 else 0.0,
        "roc_auc_mean": float(roc_arr.mean()) if roc_arr.size > 0 else 0.0,
        "roc_auc_std": float(roc_arr.std(ddof=1)) if roc_arr.size > 1 else 0.0,
    }


# -----------------------------
# SHAP explainability 
# -----------------------------

def compute_topk_attribution_mass(shap_values: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute Top-K Attribution Mass for each sample.
    
    For each sample, this is the fraction of total absolute SHAP attribution mass
    captured by the top K features (by absolute value).
    
    Args:
        shap_values: array of shape (n_samples, n_features)
        k: number of top features to consider
    
    Returns:
        array of shape (n_samples,) with top-k mass fraction for each sample [0, 1]
    """
    abs_shap = np.abs(shap_values)
    
    total_mass = abs_shap.sum(axis=1, keepdims=True)
    total_mass = np.where(total_mass == 0, 1.0, total_mass)
    topk_mass = np.partition(abs_shap, -k, axis=1)[:, -k:].sum(axis=1, keepdims=True)
    
    # Fraction of total mass in top-k features
    topk_fraction = (topk_mass / total_mass).ravel()
    return topk_fraction


def aggregate_topk_by_tp_fp(
    shap_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_mask: np.ndarray,
    k: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate Top-K Attribution Mass by True Positives and False Positives.
    
    Args:
        shap_values: SHAP values computed on a subset of test data, shape (n_samples, n_features)
        y_true: full test labels (original length)
        y_pred: binary predictions on full test set (0 or 1)
        sample_mask: boolean mask or array of indices indicating which samples were explained
        k: top-k parameter
    
    Returns:
        dict with structure:
        {
            "tp": {"mean": float, "std": float, "count": int},
            "fp": {"mean": float, "std": float, "count": int}
        }
    """
    topk_masses = compute_topk_attribution_mass(shap_values, k=k)
    
    if isinstance(sample_mask, np.ndarray) and sample_mask.dtype == bool:
        mask = sample_mask
    else:
        mask = np.zeros(len(y_true), dtype=bool)
        mask[sample_mask] = True
    
    # Get subset of true/pred labels corresponding to explained samples
    y_true_subset = y_true[mask]
    y_pred_subset = y_pred[mask]
    
    # Identify TP and FP in this subset
    tp_mask = (y_pred_subset == 1) & (y_true_subset == 1)
    fp_mask = (y_pred_subset == 1) & (y_true_subset == 0)
    
    out: Dict[str, Dict[str, float]] = {}
    
    # TP statistics
    if tp_mask.sum() > 0:
        tp_masses = topk_masses[tp_mask]
        out["tp"] = {
            "mean": float(tp_masses.mean()),
            "std": float(tp_masses.std(ddof=1)) if len(tp_masses) > 1 else 0.0,
            "count": int(len(tp_masses))
        }
    else:
        out["tp"] = {"mean": float('nan'), "std": float('nan'), "count": 0}
    
    # FP statistics
    if fp_mask.sum() > 0:
        fp_masses = topk_masses[fp_mask]
        out["fp"] = {
            "mean": float(fp_masses.mean()),
            "std": float(fp_masses.std(ddof=1)) if len(fp_masses) > 1 else 0.0,
            "count": int(len(fp_masses))
        }
    else:
        out["fp"] = {"mean": float('nan'), "std": float('nan'), "count": 0}
    
    return out


def try_shap_explain(model, X_background: pd.DataFrame, X_explain: pd.DataFrame, out_dir: str):
    try:
        import shap
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)

        X_background = X_background.apply(pd.to_numeric, errors="coerce")
        X_explain = X_explain.apply(pd.to_numeric, errors="coerce")

        bg_means = X_background.mean()
        X_background = X_background.fillna(bg_means).fillna(0.0)
        X_explain = X_explain.fillna(bg_means).fillna(0.0)

        # -----------------------------
        # Logistic Regression 
        # -----------------------------
        if isinstance(model, Pipeline):
            scaler = model.named_steps["scaler"]
            clf = model.named_steps["clf"]

            Xb = scaler.transform(X_background)
            Xe = scaler.transform(X_explain)

            explainer = shap.LinearExplainer(clf, Xb)
            shap_values = explainer.shap_values(Xe)

        # -----------------------------
        # XGBoost
        # -----------------------------
        elif model.__class__.__name__.lower().startswith("xgb"):
            if hasattr(model, "get_booster"):
                booster = model.get_booster()

                try:
                    cfg_str = booster.save_config()
                    cfg_json = json.loads(cfg_str)

                    def _clean(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                obj[k] = _clean(v)
                            return obj
                        if isinstance(obj, list):
                            return [_clean(x) for x in obj]
                        if isinstance(obj, str):
                            s = obj.strip()
                            if s.startswith("[") and s.endswith("]"):
                                inner = s[1:-1]
                                # try convert scientific notation or float
                                try:
                                    return float(inner)
                                except Exception:
                                    return obj
                            return obj
                        return obj

                    cfg_clean = _clean(cfg_json)
                    try:
                        booster.load_config(json.dumps(cfg_clean))
                    except Exception:
                        pass
                except Exception:
                    pass

                try:
                    explainer = shap.TreeExplainer(booster)
                    shap_values = explainer.shap_values(X_explain)
                    X_explain_used = X_explain
                except Exception:
                    # Fallback to kernel-based SHAP for robustness across library versions
                    from shap import KernelExplainer
                    bg_small = X_background.sample(n=min(100, len(X_background)), random_state=RANDOM_SEED)
                    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_explain.columns))[:, 1]
                    explainer = KernelExplainer(predict_fn, bg_small)
                    ex_small = X_explain.sample(n=min(200, len(X_explain)), random_state=RANDOM_SEED)
                    shap_values = explainer.shap_values(ex_small, nsamples=100)
                    X_explain_used = ex_small
            else:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_explain)
                    X_explain_used = X_explain
                except Exception:
                    from shap import KernelExplainer
                    bg_small = X_background.sample(n=min(100, len(X_background)), random_state=RANDOM_SEED)
                    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_explain.columns))[:, 1]
                    explainer = KernelExplainer(predict_fn, bg_small)
                    ex_small = X_explain.sample(n=min(200, len(X_explain)), random_state=RANDOM_SEED)
                    shap_values = explainer.shap_values(ex_small, nsamples=100)
                    X_explain_used = ex_small

        # -----------------------------
        # Random Forest
        # -----------------------------
        elif isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values_all = explainer.shap_values(X_explain)

            # Robust handling for different SHAP return types
            sv = shap_values_all
            sv_arr = np.array(sv)

            if isinstance(sv, list):
                if len(sv) >= 2:
                    shap_values = np.array(sv[1])
                else:
                    shap_values = np.array(sv[0])

            elif sv_arr.ndim == 3:
                shap_values = sv_arr[:, :, -1]

            elif sv_arr.ndim == 2:
                if sv_arr.shape == X_explain.shape:
                    shap_values = sv_arr
                elif sv_arr.shape == (X_explain.shape[1], X_explain.shape[0]):
                    shap_values = sv_arr.T
                elif sv_arr.shape[1] == X_explain.shape[1] + 1:
                    shap_values = sv_arr[:, : X_explain.shape[1]]
                else:
                    shap_values = sv_arr

            else:
                raise ValueError(f"Unsupported SHAP output shape for RandomForest: {sv_arr.shape}")


        # -----------------------------
        # HistGradientBoosting
        # -----------------------------
        elif isinstance(model, HistGradientBoostingClassifier):
            print("[INFO] SHAP skipped for HistGradientBoosting (limited support)")
            return False, None, None, None

        else:
            print("[WARN] SHAP not supported for model:", type(model))
            return False, None, None, None

        # -----------------------------
        # Global summary
        # -----------------------------
        X_explain_used = locals().get("X_explain_used", X_explain)

        shap.summary_plot(
            shap_values,
            X_explain_used,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary.png"), dpi=200)
        plt.close()

        # -----------------------------
        # Local explanations
        # -----------------------------
        base_value = explainer.expected_value
        if isinstance(base_value, (list, tuple, np.ndarray)):
            try:
                base_value = float(np.array(base_value).ravel()[-1])
            except Exception:
                base_value = float(np.array(base_value).ravel()[0])

        for i in range(min(5, X_explain_used.shape[0])):
            exp = shap.Explanation(
                values=shap_values[i],
                base_values=base_value,
                data=X_explain_used.iloc[i],
                feature_names=X_explain_used.columns,
            )
            shap.plots.waterfall(exp, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"shap_waterfall_{i}.png"), dpi=200)
            plt.close()

        # Return SHAP values for metric computation
        return True, shap_values, X_explain_used, None

    except Exception as e:
        print(f"[WARN] SHAP explainability skipped: {e}")
        return False, None, None, None


# -----------------------------
# Main runner
# -----------------------------
def main(
    csv_path: str = "data/raw/creditcard.csv",
    split_mode: str = "time",          
    threshold_mode: str = "fpr_budget",
    target_fpr: float = 0.01,
    beta: float = 2.0,
    do_shap: bool = True,
):
    # Ensure CSV file exists; download if missing
    csv_path = ensure_creditcard_csv(csv_path)
    
    df = load_creditcard_csv(csv_path)

    if split_mode == "time":
        data = time_based_split(df, train_frac=0.70, val_frac=0.15)
    else:
        data = stratified_split(df, train_size=0.70, val_size=0.15, seed=RANDOM_SEED)

    # Identify feature columns
    feature_cols = [c for c in data.X_train.columns if c != "Class"]
    X_train = data.X_train[feature_cols].reset_index(drop=True)
    X_val = data.X_val[feature_cols].reset_index(drop=True)
    X_test = data.X_test[feature_cols].reset_index(drop=True)

    models = build_models(data.y_train)

    results = {}
    start_overall = time.time()
    for idx, (name, model) in enumerate(models.items(), 1):
        start_model = time.time()
        print(f"\n[{idx}/{len(models)}] Training: {name} ...", flush=True)
        
        # Compute sample_weight only for HGB (imbalance handling)
        sample_weight = None
        if name == "hgb":
            sample_weight = compute_sample_weight_for_imbalance(data.y_train)
        
        model = fit_model(model, X_train, data.y_train, sample_weight=sample_weight)

        p_val = predict_proba(model, X_val)
        thr, thr_details = choose_threshold_validation(
            data.y_val, p_val, mode=threshold_mode, target_fpr=target_fpr, beta=beta
        )

        p_test = predict_proba(model, X_test)

        val_metrics = evaluate_all(data.y_val, p_val, threshold=thr)
        test_metrics = evaluate_all(data.y_test, p_test, threshold=thr)

        results[name] = {
            "threshold_selection": thr_details,
            "val": val_metrics,
            "test": test_metrics,
        }

        # -----------------------------
        # Statistical robustness analyses
        # 1) Stratified bootstrap CIs on the held-out test set 
        # 2) Repeated stratified CV on the TRAINING portion only
        # -----------------------------
        try:
            print(f"  Computing bootstrap CIs (1000 resamples) ...", flush=True)
            bs = bootstrap_test_metrics(data.y_test, p_test, thr, n_boot=1000, seed=RANDOM_SEED)
            results[name]["test_bootstrap_ci"] = bs
            print(f"  Bootstrap CIs complete", flush=True)
        except Exception as e:
            print(f"[WARN] bootstrap_test_metrics failed for {name}: {e}")

        try:
            print(f"  Running repeated stratified CV (5 splits, 3 repeats, 15 folds total) ...", flush=True)
            cvres = run_repeated_cv(X_train, data.y_train, name, n_splits=5, n_repeats=3, seed=RANDOM_SEED)
            results[name]["cv_metrics"] = cvres
            print(f"  CV complete", flush=True)
        except Exception as e:
            print(f"[WARN] run_repeated_cv failed for {name}: {e}")

        print(f"Threshold chosen (val): {thr:.6f}  |  details: {thr_details}")
        print(f"VAL  PR-AUC={val_metrics['pr_auc']:.4f}, Recall={val_metrics['recall']:.4f}, Precision={val_metrics['precision']:.4f}")
        print(f"TEST PR-AUC={test_metrics['pr_auc']:.4f}, Recall={test_metrics['recall']:.4f}, Precision={test_metrics['precision']:.4f}")
        
        elapsed_model = time.time() - start_model
        print(f"[{name} completed in {elapsed_model:.1f}s]", flush=True)

        # SHAP on alerted samples only 
        if do_shap and name in ["xgb", "rf", "hgb", "logreg"]:
            alert_idx = np.where(p_test >= thr)[0]
            
            # If no alerts at this threshold, skip explainability cleanly
            if len(alert_idx) == 0:
                print(f"[INFO] No alerts at threshold for {name}; skipping SHAP.")
            else:
                ex = X_test.iloc[alert_idx]
                
                # Cap runtime for efficiency
                if len(ex) > 500:
                    ex = ex.sample(n=500, random_state=RANDOM_SEED)
                
                bg = X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_SEED)
                out_dir = os.path.join("results", f"shap_{name}")
                shap_result = try_shap_explain(model, bg, ex, out_dir)
                
                if shap_result[0] and shap_result[1] is not None:  
                    shap_values = shap_result[1]
                    X_explain_used = shap_result[2]
                    
                    sample_mask = np.zeros(len(X_test), dtype=bool)
                    sample_mask[X_explain_used.index.tolist()] = True
                    
                    # Compute Top-K Attribution Mass aggregated by TP/FP
                    try:
                        y_pred_test = (p_test >= thr).astype(int)
                        topk_agg = aggregate_topk_by_tp_fp(
                            shap_values, data.y_test, y_pred_test, sample_mask, k=3
                        )
                        if "explainability_concentration" not in results[name]:
                            results[name]["explainability_concentration"] = {}
                        results[name]["explainability_concentration"]["top3_attribution_mass"] = topk_agg
                        print(f"  Top-3 Attribution Mass computed | TP mean={topk_agg['tp']['mean']:.4f}, FP mean={topk_agg['fp']['mean']:.4f}")
                    except Exception as e:
                        print(f"[WARN] Top-K Attribution Mass computation failed for {name}: {e}")

    elapsed_overall = time.time() - start_overall
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Pipeline complete! Total time: {elapsed_overall:.1f}s ({elapsed_overall/60:.1f}m)")
    print(f"Results saved to: results/results.json")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main(
        csv_path="data/raw/creditcard.csv",
        split_mode="time",
        threshold_mode="fpr_budget",
        target_fpr=0.01,
        beta=2.0,
        do_shap=True,
    )
