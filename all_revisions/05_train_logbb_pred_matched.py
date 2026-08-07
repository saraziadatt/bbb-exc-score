#!/usr/bin/env python3
"""
Matched, fold-safe retraining of the published LogBB_Pred modelling strategy.

This script trains a LightGBM regression comparator on a master CSV containing
fixed outer-fold assignments. It is designed for a fair comparison with other
models trained on the same compounds and folds, rather than for reproducing the
historical performance reported on Shaker et al.'s original dataset.

Within every inner or outer training fit, the script:

    1. Uses only compounds with measured logBB values to fit the regressor.
    2. Removes Mordred descriptors containing any missing value among those
       measured training compounds.
    3. For descriptor pairs with |Pearson r| > 0.8, removes the descriptor with
       the weaker absolute Pearson correlation with training logBB.
    4. Tunes LightGBM using inner-fold mean squared error.
    5. Predicts every compound in the held-out outer fold, including compounds
       having only a BBB classification label.
    6. Converts predicted logBB to BBB class using predicted logBB < -1.0.

No feature scaling or descriptor imputation is applied. LightGBM handles a
held-out or external NaN natively if a descriptor happened to be complete in
training but missing at prediction time.

Published LightGBM tuning information
--------------------------------------
Supplementary Table S1 reports these search bounds and selected optimum:

    n_estimators:      50-700   (optimum 100)
    num_leaves:         5-30    (optimum 15)
    max_depth:          5-25    (optimum 10)
    min_data_in_leaf:  10-15    (optimum 15)
    bagging_fraction: 0.2-0.8   (optimum 0.6)

The supplement does not report the discrete GridSearchCV step sizes. Therefore,
the default matched-retraining mode evaluates a deterministic set of candidates
sampled within exactly those bounds and always includes the published optimum.
Use --tuning-mode published_optimal to fit only the reported optimum, or
--grid-json to provide a fully explicit candidate grid.

Recommended input
-----------------
Use the Mordred table from before global removal of descriptors with missing
values. The fold-local transformer then decides which descriptors are complete
using training compounds only. A globally prefiltered table can still be used,
but it has already used held-out feature availability to determine the columns.

Required input columns
----------------------
    logBB       continuous target; NaN for classification-only compounds
    bbb_class   0 = penetrant, 1 = non-penetrant
    outer_fold  fixed outer-fold assignment
    smiles      molecular SMILES

Optional:
    sample_id
    inner_fold_outer_0, inner_fold_outer_1, ...

All other numeric columns are treated as molecular descriptors, except
ECFP_* and MACCS_* fingerprint columns, which are excluded automatically.

Example
-------
python 05_train_logbb_pred_matched.py \
    --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
    --output-dir logbb_pred_results \
    --tuning-mode published_ranges \
    --n-search-candidates 250 \
    --correlation-threshold 0.8 \
    --n-jobs 4

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


MODEL_NAME = "logbb_pred_matched_retraining"
DEFAULT_TARGET_COL = "logBB"
DEFAULT_CLASS_COL = "bbb_class"
DEFAULT_FOLD_COL = "outer_fold"
DEFAULT_SMILES_COL = "smiles"
DEFAULT_ID_COL = "sample_id"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


class CompleteTrainingDescriptorFilter(BaseEstimator, TransformerMixin):
    """
    Retain descriptors having no missing values in the fitted training data.

    The paper removed descriptors containing missing values. Performing this
    step inside the pipeline avoids consulting the held-out fold. A median
    imputer immediately after this transformer handles the rare case in which
    a descriptor is complete in training but missing in a held-out compound.
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "CompleteTrainingDescriptorFilter":
        del y
        frame = pd.DataFrame(X).copy()
        self.input_columns_ = frame.columns.tolist()
        missing_counts = frame.isna().sum(axis=0)
        self.kept_columns_ = missing_counts.index[missing_counts == 0].tolist()
        self.dropped_columns_ = missing_counts.index[missing_counts > 0].tolist()
        if not self.kept_columns_:
            raise ValueError(
                "Every descriptor contains a missing value in this training fold."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing_columns = sorted(set(self.kept_columns_).difference(frame.columns))
        if missing_columns:
            raise KeyError(
                "Prediction data are missing descriptors required by the fitted "
                f"model. First missing columns: {missing_columns[:20]}"
            )
        return frame.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.kept_columns_, dtype=object)


class TargetAwarePearsonFilter(BaseEstimator, TransformerMixin):
    """
    Remove redundant descriptor pairs using their relationship with logBB.

    For each pair with absolute feature-feature Pearson correlation greater
    than `threshold`, the descriptor with the lower absolute Pearson
    correlation with y is discarded. Pairs are processed deterministically
    from strongest feature-feature correlation to weakest. Ties retain the
    earlier input column.
    """

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> "TargetAwarePearsonFilter":
        if y is None:
            raise ValueError("TargetAwarePearsonFilter requires y during fit.")

        frame = pd.DataFrame(X).copy()
        target = np.asarray(y, dtype=float)
        if len(frame) != len(target):
            raise ValueError("X and y have different lengths.")
        if not 0 < float(self.threshold) < 1:
            raise ValueError("threshold must be between 0 and 1.")

        self.input_columns_ = frame.columns.tolist()
        target_correlations: dict[str, float] = {}
        for column in self.input_columns_:
            value = abs(safe_pearson(frame[column].to_numpy(float), target))
            target_correlations[column] = (
                value if np.isfinite(value) else -math.inf
            )

        pairwise = frame.corr(method="pearson").abs()
        pairs: list[tuple[float, int, int, str, str]] = []
        for i, left in enumerate(self.input_columns_):
            for j in range(i + 1, len(self.input_columns_)):
                right = self.input_columns_[j]
                value = pairwise.iat[i, j]
                if np.isfinite(value) and value > float(self.threshold):
                    pairs.append((float(value), i, j, left, right))

        # Strongest redundant relationships are resolved first.
        pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
        active = set(self.input_columns_)
        decisions: list[dict[str, Any]] = []

        for feature_r, left_i, right_i, left, right in pairs:
            if left not in active or right not in active:
                continue

            left_target_r = target_correlations[left]
            right_target_r = target_correlations[right]

            if left_target_r < right_target_r:
                dropped, retained = left, right
            elif right_target_r < left_target_r:
                dropped, retained = right, left
            else:
                # Deterministic tie: retain the earlier input column.
                dropped, retained = (
                    (right, left) if left_i < right_i else (left, right)
                )

            active.remove(dropped)
            decisions.append(
                {
                    "dropped_feature": dropped,
                    "retained_feature": retained,
                    "abs_feature_feature_r": feature_r,
                    "abs_dropped_target_r": target_correlations[dropped],
                    "abs_retained_target_r": target_correlations[retained],
                }
            )

        self.target_correlations_ = target_correlations
        self.decisions_ = decisions
        self.dropped_columns_ = [
            column for column in self.input_columns_ if column not in active
        ]
        self.kept_columns_ = [
            column for column in self.input_columns_ if column in active
        ]

        if not self.kept_columns_:
            raise ValueError("Target-aware Pearson filtering removed all features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing_columns = sorted(set(self.kept_columns_).difference(frame.columns))
        if missing_columns:
            raise KeyError(
                "Prediction data are missing selected descriptors. "
                f"First missing columns: {missing_columns[:20]}"
            )
        return frame.loc[:, self.kept_columns_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return np.asarray(self.kept_columns_, dtype=object)


def make_pipeline(
    *,
    seed: int,
    correlation_threshold: float,
) -> Pipeline:
    return Pipeline(
        [
            ("complete_descriptors", CompleteTrainingDescriptorFilter()),
            (
                "target_pcc_filter",
                TargetAwarePearsonFilter(
                    threshold=correlation_threshold,
                ),
            ),
            (
                "model",
                lgb.LGBMRegressor(
                    objective="regression_l2",
                    boosting_type="gbdt",
                    learning_rate=0.1,
                    subsample_freq=1,
                    random_state=seed,
                    n_jobs=1,
                    verbosity=-1,
                ),
            ),
        ]
    )


PUBLISHED_OPTIMUM = {
    "model__n_estimators": 100,
    "model__num_leaves": 15,
    "model__max_depth": 10,
    "model__min_child_samples": 15,
    "model__subsample": 0.6,
}

PUBLISHED_BOUNDS = {
    "n_estimators": [50, 700],
    "num_leaves": [5, 30],
    "max_depth": [5, 25],
    "min_data_in_leaf": [10, 15],
    "bagging_fraction": [0.2, 0.8],
}


def singleton_grid(candidate: dict[str, Any]) -> dict[str, list[Any]]:
    """Convert one candidate into a GridSearchCV-compatible singleton grid."""
    return {key: [value] for key, value in candidate.items()}


def published_range_candidates(
    *,
    n_candidates: int,
    seed: int,
) -> list[dict[str, list[Any]]]:
    """
    Build deterministic candidates within the exact reported search bounds.

    Supplementary Table S1 gives parameter bounds but not the discrete values
    passed to GridSearchCV. We therefore sample a reproducible candidate set
    inside those bounds and prepend the exact published optimum. GridSearchCV
    then evaluates each candidate using the saved or generated inner folds.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1.")

    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = [dict(PUBLISHED_OPTIMUM)]
    seen = {tuple(sorted(PUBLISHED_OPTIMUM.items()))}

    while len(candidates) < n_candidates:
        candidate = {
            "model__n_estimators": int(rng.integers(50, 701)),
            "model__num_leaves": int(rng.integers(5, 31)),
            "model__max_depth": int(rng.integers(5, 26)),
            "model__min_child_samples": int(rng.integers(10, 16)),
            "model__subsample": float(np.round(rng.uniform(0.2, 0.8), 3)),
        }
        signature = tuple(sorted(candidate.items()))
        if signature not in seen:
            candidates.append(candidate)
            seen.add(signature)

    return [singleton_grid(candidate) for candidate in candidates]


def load_candidate_grid(
    *,
    tuning_mode: str,
    n_search_candidates: int,
    seed: int,
    grid_json: Optional[Path],
) -> list[dict[str, list[Any]]]:
    if grid_json is not None:
        raw = json.loads(grid_json.read_text())
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ValueError("--grid-json must contain a non-empty dict or list.")

        output: list[dict[str, list[Any]]] = []
        for block in raw:
            if not isinstance(block, dict):
                raise TypeError("Every grid block must be a JSON object.")
            normalized: dict[str, list[Any]] = {}
            for key, values in block.items():
                parameter = (
                    key if str(key).startswith("model__") else f"model__{key}"
                )
                normalized[parameter] = (
                    values if isinstance(values, list) else [values]
                )
            output.append(normalized)
        return output

    if tuning_mode == "published_optimal":
        return [singleton_grid(PUBLISHED_OPTIMUM)]
    if tuning_mode == "smoke":
        return [
            singleton_grid(
                {
                    "model__n_estimators": 50,
                    "model__num_leaves": 5,
                    "model__max_depth": 5,
                    "model__min_child_samples": 10,
                    "model__subsample": 0.8,
                }
            )
        ]
    if tuning_mode == "published_ranges":
        return published_range_candidates(
            n_candidates=n_search_candidates,
            seed=seed,
        )
    raise ValueError(f"Unknown tuning mode: {tuning_mode}")


def infer_descriptor_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    class_col: str,
    fold_col: str,
    smiles_col: str,
    id_col: str,
) -> list[str]:
    metadata = {
        target_col,
        class_col,
        fold_col,
        smiles_col,
        id_col,
        "has_logbb",
        "logbb_is_measured",
        "classification_only",
        "outer_stratum",
        "row_index",
    }
    metadata = {str(column).strip().lower() for column in metadata}
    excluded_prefixes = (
        "inner_fold_outer_",
        "pred_",
        "prediction",
        "oof_",
        "shap_",
        "ecfp_",
        "maccs_",
    )

    columns: list[str] = []
    for column in df.columns:
        name = str(column)
        lower = name.strip().lower()
        if lower in metadata:
            continue
        if lower.startswith(excluded_prefixes):
            continue
        if lower.startswith("unnamed:"):
            continue
        if lower in {"index", "level_0", "fold", "target", "label", "class"}:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)

    if not columns:
        raise ValueError("No numeric molecular descriptor columns were found.")
    return columns


@dataclass
class DataBundle:
    frame: pd.DataFrame
    X: pd.DataFrame
    y_reg: np.ndarray
    y_class: np.ndarray
    outer_fold: np.ndarray
    sample_id: np.ndarray
    smiles: np.ndarray
    feature_columns: list[str]


def load_data(args: argparse.Namespace) -> DataBundle:
    path = args.input.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)
    required = {
        args.target_col,
        args.class_col,
        args.fold_col,
        args.smiles_col,
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    if args.id_col not in df.columns:
        df[args.id_col] = np.arange(len(df), dtype=int)

    y_reg = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)
    y_class = pd.to_numeric(df[args.class_col], errors="raise").to_numpy(int)
    outer_fold = pd.to_numeric(
        df[args.fold_col],
        errors="raise",
    ).to_numpy(int)

    if not set(np.unique(y_class)).issubset({0, 1}):
        raise ValueError(
            f"{args.class_col} must use 0=penetrant and 1=non-penetrant."
        )

    measured = np.isfinite(y_reg)
    implied_class = (y_reg[measured] < args.logbb_threshold).astype(int)
    disagreement = int(np.sum(implied_class != y_class[measured]))
    if disagreement and not args.allow_label_disagreement:
        raise ValueError(
            f"{disagreement} measured rows disagree with the logBB threshold "
            f"{args.logbb_threshold}. Use --allow-label-disagreement only if "
            "this is intentional."
        )

    feature_columns = infer_descriptor_columns(
        df,
        target_col=args.target_col,
        class_col=args.class_col,
        fold_col=args.fold_col,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
    )

    forbidden_features = {
        "has_logbb",
        "logbb_is_measured",
        "classification_only",
        "outer_fold",
        "outer_stratum",
        "row_index",
        "logbb",
        "bbb_class",
    }

    leaked_features = [
        column
        for column in feature_columns
        if str(column).strip().lower() in forbidden_features
    ]

    if leaked_features:
        raise RuntimeError(
            f"Metadata or target columns entered the descriptor set: "
            f"{leaked_features}"
        )

    print(f"Using {len(feature_columns)} molecular descriptors.")
    print(f"First descriptors: {feature_columns[:10]}")

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    globally_all_nan = X.columns[X.isna().all()].tolist()
    if globally_all_nan:
        warnings.warn(
            f"Dropping {len(globally_all_nan)} globally all-NaN descriptors. "
            f"Examples: {globally_all_nan[:10]}"
        )
        X = X.drop(columns=globally_all_nan)
        feature_columns = X.columns.tolist()

    return DataBundle(
        frame=df,
        X=X,
        y_reg=y_reg,
        y_class=y_class,
        outer_fold=outer_fold,
        sample_id=df[args.id_col].to_numpy(),
        smiles=df[args.smiles_col].astype(str).to_numpy(),
        feature_columns=feature_columns,
    )


def saved_inner_splits(
    df: pd.DataFrame,
    *,
    outer_fold: int,
    fit_indices: np.ndarray,
) -> Optional[list[tuple[np.ndarray, np.ndarray]]]:
    column = f"inner_fold_outer_{outer_fold}"
    if column not in df.columns:
        return None

    assignments = pd.to_numeric(df.loc[fit_indices, column], errors="coerce")
    if assignments.isna().any():
        return None

    values = assignments.to_numpy(int)
    unique = sorted(np.unique(values))
    if len(unique) < 2:
        return None

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_fold in unique:
        validation = np.flatnonzero(values == inner_fold)
        training = np.flatnonzero(values != inner_fold)
        if len(training) == 0 or len(validation) == 0:
            return None
        splits.append((training, validation))
    return splits


def generated_inner_splits(
    n_samples: int,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    resolved = min(int(n_splits), int(n_samples))
    if resolved < 2:
        raise ValueError("At least two measured training compounds are required.")
    splitter = KFold(n_splits=resolved, shuffle=True, random_state=seed)
    indices = np.arange(n_samples)
    return [(train, valid) for train, valid in splitter.split(indices)]


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_class: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(mask):
        return {
            "n_evaluable": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "median_absolute_error": np.nan,
            "r2": np.nan,
            "pearson_r": np.nan,
            "np_mae": np.nan,
            "penetrant_mae": np.nan,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    yc = (
        np.asarray(y_class, dtype=float)[mask]
        if y_class is not None
        else np.full(len(yt), np.nan)
    )
    mse = float(mean_squared_error(yt, yp))
    output = {
        "n_evaluable": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "median_absolute_error": float(median_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "pearson_r": safe_pearson(yt, yp),
        "np_mae": np.nan,
        "penetrant_mae": np.nan,
    }
    if np.any(yc == 1):
        output["np_mae"] = float(
            mean_absolute_error(yt[yc == 1], yp[yc == 1])
        )
    if np.any(yc == 0):
        output["penetrant_mae"] = float(
            mean_absolute_error(yt[yc == 0], yp[yc == 0])
        )
    return output


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_nonpenetrant: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score = np.asarray(score_nonpenetrant, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(score)

    if not np.any(mask):
        return {
            "n_evaluable": 0,
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "np_recall": np.nan,
            "np_precision": np.nan,
            "np_f1": np.nan,
            "penetrant_specificity": np.nan,
            "roc_auc": np.nan,
            "pr_auc_nonpenetrant": np.nan,
            "cohen_kappa": np.nan,
            "mcc": np.nan,
            "wrong_side_count": 0,
            "wrong_side_rate": np.nan,
        }

    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    ys = score[mask]
    wrong = yt != yp
    has_both = len(np.unique(yt)) == 2
    penetrant = yt == 0

    return {
        "n_evaluable": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "np_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "np_precision": float(
            precision_score(yt, yp, pos_label=1, zero_division=0)
        ),
        "np_f1": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "penetrant_specificity": (
            float(np.mean(yp[penetrant] == 0))
            if np.any(penetrant)
            else np.nan
        ),
        "roc_auc": float(roc_auc_score(yt, ys)) if has_both else np.nan,
        "pr_auc_nonpenetrant": (
            float(average_precision_score(yt, ys)) if has_both else np.nan
        ),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "wrong_side_count": int(np.sum(wrong)),
        "wrong_side_rate": float(np.mean(wrong)),
    }


def prediction_frame(
    data: DataBundle,
    test_indices: np.ndarray,
    predicted_logbb: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predicted_logbb = np.asarray(predicted_logbb, dtype=float)
    predicted_class = (predicted_logbb < threshold).astype(int)
    score_np = -predicted_logbb
    true_logbb = data.y_reg[test_indices]
    true_class = data.y_class[test_indices]

    absolute_error = np.where(
        np.isfinite(true_logbb),
        np.abs(predicted_logbb - true_logbb),
        np.nan,
    )
    squared_error = np.where(
        np.isfinite(true_logbb),
        np.square(predicted_logbb - true_logbb),
        np.nan,
    )

    return pd.DataFrame(
        {
            "sample_id": data.sample_id[test_indices],
            "row_index": test_indices,
            "smiles": data.smiles[test_indices],
            "outer_fold": data.outer_fold[test_indices],
            "true_logBB": true_logbb,
            "bbb_class": true_class,
            "has_logBB": np.isfinite(true_logbb),
            "classification_only": ~np.isfinite(true_logbb),
            "model": MODEL_NAME,
            "model_task": "regression",
            "predicted_logBB": predicted_logbb,
            "score_nonpenetrant": score_np,
            "predicted_bbb_class": predicted_class,
            "classification_threshold": threshold,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "wrong_side": predicted_class != true_class,
        }
    )


def metrics_table(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    def add_rows(
        frame: pd.DataFrame,
        outer_fold: Any,
        destination: list[dict[str, Any]],
    ) -> None:
        reg = regression_metrics(
            frame["true_logBB"].to_numpy(float),
            frame["predicted_logBB"].to_numpy(float),
            frame["bbb_class"].to_numpy(int),
        )
        for metric, value in reg.items():
            destination.append(
                {
                    "model": MODEL_NAME,
                    "evaluation_task": "regression",
                    "outer_fold": outer_fold,
                    "metric": metric,
                    "value": value,
                }
            )

        cls = classification_metrics(
            frame["bbb_class"].to_numpy(int),
            frame["predicted_bbb_class"].to_numpy(int),
            frame["score_nonpenetrant"].to_numpy(float),
        )
        for metric, value in cls.items():
            destination.append(
                {
                    "model": MODEL_NAME,
                    "evaluation_task": "classification",
                    "outer_fold": outer_fold,
                    "metric": metric,
                    "value": value,
                }
            )

    add_rows(predictions, "pooled", pooled_rows)
    for outer_fold, frame in predictions.groupby("outer_fold", sort=True):
        add_rows(frame, int(outer_fold), fold_rows)

    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def mean_sd_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = fold_metrics.copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    summary = (
        numeric.groupby(
            ["model", "evaluation_task", "metric"],
            as_index=False,
        )["value"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["mean_sd"] = summary.apply(
        lambda row: (
            f"{row['mean']:.6f} ± {row['std']:.6f}"
            if np.isfinite(row["mean"]) and np.isfinite(row["std"])
            else (
                f"{row['mean']:.6f}"
                if np.isfinite(row["mean"])
                else ""
            )
        ),
        axis=1,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logbb_pred_matched_results"),
    )
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--class-col", default=DEFAULT_CLASS_COL)
    parser.add_argument("--fold-col", default=DEFAULT_FOLD_COL)
    parser.add_argument("--smiles-col", default=DEFAULT_SMILES_COL)
    parser.add_argument("--id-col", default=DEFAULT_ID_COL)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--logbb-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--tuning-mode",
        choices=["published_ranges", "published_optimal", "smoke"],
        default="published_ranges",
        help=(
            "published_ranges performs fold-local tuning within the exact "
            "reported bounds; published_optimal uses the reported optimum "
            "without tuning."
        ),
    )
    parser.add_argument(
        "--n-search-candidates",
        type=int,
        default=250,
        help=(
            "Number of deterministic candidates evaluated when "
            "--tuning-mode=published_ranges. The published optimum is always "
            "included."
        ),
    )
    parser.add_argument(
        "--grid-json",
        type=Path,
        default=None,
        help=(
            "Optional explicit GridSearchCV grid. This overrides --tuning-mode. "
            "Keys may use LightGBM names with or without model__."
        ),
    )
    parser.add_argument("--skip-model-saving", action="store_true")
    parser.add_argument("--allow-label-disagreement", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is non-empty. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    prepare_output_directory(args.output_dir, args.overwrite)
    model_dir = args.output_dir / "fitted_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args)
    grid = load_candidate_grid(
        tuning_mode=args.tuning_mode,
        n_search_candidates=args.n_search_candidates,
        seed=args.seed,
        grid_json=args.grid_json,
    )
    outer_folds = sorted(np.unique(data.outer_fold))

    print(f"Rows: {len(data.frame)}")
    print(f"Measured logBB rows: {np.sum(np.isfinite(data.y_reg))}")
    print(f"Descriptor columns: {data.X.shape[1]}")
    print(f"Outer folds: {outer_folds}")
    print(f"Correlation threshold: {args.correlation_threshold}")
    print(f"Tuning mode: {args.tuning_mode}")
    print(f"Candidate parameter settings: {len(grid)}")

    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for outer_fold in outer_folds:
        print(f"\n=== Outer fold {outer_fold} ===")
        train_mask = data.outer_fold != outer_fold
        test_indices = np.flatnonzero(data.outer_fold == outer_fold)
        fit_indices = np.flatnonzero(train_mask & np.isfinite(data.y_reg))

        if len(fit_indices) < 2:
            raise ValueError(
                f"Outer fold {outer_fold} has fewer than two measured "
                "training compounds."
            )

        X_fit = data.X.iloc[fit_indices]
        y_fit = data.y_reg[fit_indices]

        inner_cv = saved_inner_splits(
            data.frame,
            outer_fold=int(outer_fold),
            fit_indices=fit_indices,
        )
        if inner_cv is None:
            inner_cv = generated_inner_splits(
                n_samples=len(fit_indices),
                n_splits=args.inner_folds,
                seed=args.seed + int(outer_fold),
            )

        pipeline = make_pipeline(
            seed=args.seed + int(outer_fold),
            correlation_threshold=args.correlation_threshold,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="neg_mean_squared_error",
            cv=inner_cv,
            refit=True,
            n_jobs=args.n_jobs,
            verbose=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_fit, y_fit)
        fitted = search.best_estimator_

        predicted_logbb = np.asarray(
            fitted.predict(data.X.iloc[test_indices]),
            dtype=float,
        )
        prediction_parts.append(
            prediction_frame(
                data,
                test_indices,
                predicted_logbb,
                args.logbb_threshold,
            )
        )

        complete_filter = fitted.named_steps["complete_descriptors"]
        pcc_filter = fitted.named_steps["target_pcc_filter"]

        parameter_rows.append(
            {
                "model": MODEL_NAME,
                "outer_fold": int(outer_fold),
                "n_outer_train_measured": int(len(fit_indices)),
                "best_inner_neg_mse": float(search.best_score_),
                "best_inner_mse": float(-search.best_score_),
                "n_input_descriptors": int(data.X.shape[1]),
                "n_complete_training_descriptors": int(
                    len(complete_filter.kept_columns_)
                ),
                "n_selected_descriptors": int(
                    len(pcc_filter.kept_columns_)
                ),
                "correlation_threshold": float(args.correlation_threshold),
                "best_parameters_json": json.dumps(
                    json_safe(search.best_params_),
                    sort_keys=True,
                ),
            }
        )

        for feature in complete_filter.dropped_columns_:
            selection_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "feature": feature,
                    "status": "dropped_training_missing",
                    "abs_target_r": np.nan,
                    "retained_in_pair": "",
                    "abs_feature_feature_r": np.nan,
                }
            )

        decision_by_dropped = {
            row["dropped_feature"]: row
            for row in pcc_filter.decisions_
        }
        for feature in pcc_filter.input_columns_:
            if feature in pcc_filter.kept_columns_:
                selection_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "feature": feature,
                        "status": "selected",
                        "abs_target_r": pcc_filter.target_correlations_[feature],
                        "retained_in_pair": "",
                        "abs_feature_feature_r": np.nan,
                    }
                )
            else:
                decision = decision_by_dropped[feature]
                selection_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "feature": feature,
                        "status": "dropped_redundant",
                        "abs_target_r": pcc_filter.target_correlations_[feature],
                        "retained_in_pair": decision["retained_feature"],
                        "abs_feature_feature_r": decision[
                            "abs_feature_feature_r"
                        ],
                    }
                )

        if not args.skip_model_saving:
            bundle = {
                "pipeline": fitted,
                "metadata": {
                    "model_name": MODEL_NAME,
                    "outer_fold": int(outer_fold),
                    "target_col": args.target_col,
                    "class_col": args.class_col,
                    "fold_col": args.fold_col,
                    "smiles_col": args.smiles_col,
                    "id_col": args.id_col,
                    "logbb_threshold": float(args.logbb_threshold),
                    "correlation_threshold": float(
                        args.correlation_threshold
                    ),
                    "best_inner_mse": float(-search.best_score_),
                    "best_parameters": json_safe(search.best_params_),
                    "input_feature_columns": data.feature_columns,
                    "selected_feature_columns": pcc_filter.kept_columns_,
                },
            }
            joblib.dump(
                bundle,
                model_dir
                / f"logbb_pred_matched_outer_{outer_fold}.joblib",
                compress=3,
            )

    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        .sort_values(["row_index", "outer_fold"])
        .reset_index(drop=True)
    )
    if predictions["row_index"].duplicated().any():
        raise RuntimeError("A row received more than one outer-fold prediction.")
    if len(predictions) != len(data.frame):
        raise RuntimeError(
            f"Expected {len(data.frame)} OOF predictions, got {len(predictions)}."
        )

    pooled_metrics, fold_metrics = metrics_table(predictions)
    mean_sd = mean_sd_table(fold_metrics)
    best_parameters = pd.DataFrame(parameter_rows)
    feature_selection = pd.DataFrame(selection_rows)

    predictions.to_csv(
        args.output_dir / "logbb_pred_matched_predictions.csv",
        index=False,
    )
    pooled_metrics.to_csv(
        args.output_dir / "logbb_pred_matched_pooled_metrics.csv",
        index=False,
    )
    fold_metrics.to_csv(
        args.output_dir / "logbb_pred_matched_fold_metrics.csv",
        index=False,
    )
    mean_sd.to_csv(
        args.output_dir / "logbb_pred_matched_mean_sd.csv",
        index=False,
    )
    best_parameters.to_csv(
        args.output_dir / "logbb_pred_matched_best_parameters.csv",
        index=False,
    )
    feature_selection.to_csv(
        args.output_dir / "logbb_pred_matched_feature_selection.csv",
        index=False,
    )

    manifest = {
        "model_name": MODEL_NAME,
        "interpretation": (
            "Matched, fold-safe retraining of the published LogBB_Pred modelling "
            "strategy on the supplied dataset and fixed folds."
        ),
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "n_rows": int(len(data.frame)),
        "n_measured_logbb": int(np.sum(np.isfinite(data.y_reg))),
        "n_descriptors": int(data.X.shape[1]),
        "outer_folds": [int(value) for value in outer_folds],
        "correlation_threshold": float(args.correlation_threshold),
        "missing_descriptor_policy": (
            "drop descriptors with any missing values among measured compounds in "
            "each training fit; no imputation"
        ),
        "feature_pair_policy": (
            "for |feature-feature Pearson r| above threshold, drop the feature "
            "with lower absolute Pearson correlation with training logBB"
        ),
        "scaling": "none",
        "tuning": (
            "GridSearchCV scored by negative mean squared error; default "
            "candidates lie within Supplementary Table S1 bounds and include "
            "the published optimum"
        ),
        "tuning_mode": args.tuning_mode,
        "n_search_candidates_requested": int(args.n_search_candidates),
        "grid_json": str(args.grid_json.resolve()) if args.grid_json else None,
        "published_parameter_bounds": PUBLISHED_BOUNDS,
        "published_optimum": json_safe(PUBLISHED_OPTIMUM),
        "parameter_grid": json_safe(grid),
        "logbb_classification_threshold": float(args.logbb_threshold),
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "joblib": joblib.__version__,
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\nCompleted successfully.")
    print(
        "Predictions:",
        args.output_dir / "logbb_pred_matched_predictions.csv",
    )
    print(
        "Pooled metrics:",
        args.output_dir / "logbb_pred_matched_pooled_metrics.csv",
    )
    print(
        "Mean ± SD:",
        args.output_dir / "logbb_pred_matched_mean_sd.csv",
    )


if __name__ == "__main__":
    main()
