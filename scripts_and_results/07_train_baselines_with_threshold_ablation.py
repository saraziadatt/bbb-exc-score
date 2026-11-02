
#!/usr/bin/env python3
"""
Train the four BBB baseline comparators.

Models
    1. dummy_mean_regression
    2. lightgbm_mse_regression
    3. lightgbm_mse_regression_tuned_threshold
       (same fitted MSE regressor; inner-OOF threshold maximizes balanced accuracy)
    4. dummy_prior_classification

"""

from __future__ import annotations

import argparse
import hashlib
import json
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
import optuna
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,mean_absolute_error,
                            mean_squared_error,median_absolute_error,precision_score,r2_score,recall_score,roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline


DEFAULT_TARGET_COL = "logBB"
DEFAULT_CLASS_COL = "bbb_class"
DEFAULT_FOLD_COL = "outer_fold"
DEFAULT_SMILES_COL = "smiles"
DEFAULT_ID_COL = "sample_id"

TRAINABLE_MODEL_TASKS = {
    "dummy_mean_regression": "regression",
    "lightgbm_mse_regression": "regression",
    "dummy_prior_classification": "classification",
}

THRESHOLD_TUNED_MODEL = "lightgbm_mse_regression_threshold"

MODEL_TASKS = {
    **TRAINABLE_MODEL_TASKS,
    THRESHOLD_TUNED_MODEL: "regression",
}

# Useful functions 
def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
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
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def parse_boolean_series(series: pd.Series, column_name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="raise").to_numpy(float)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError(f"{column_name!r} contains values other than 0/1.")
        return values.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    unknown = sorted(set(normalized).difference(mapping))
    if unknown:
        raise ValueError(
            f"{column_name!r} contains unrecognized booleans: {unknown[:20]}"
        )
    return normalized.map(mapping).to_numpy(bool)


def class_score_from_estimator(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    """Return a score where larger values mean more likely non-penetrant."""
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(estimator.classes_)
        if probabilities.ndim == 2 and 1 in classes:
            index = int(np.flatnonzero(classes == 1)[0])
            return probabilities[:, index]
        return np.zeros(len(X), dtype=float)
    if hasattr(estimator, "decision_function"):
        score = np.asarray(estimator.decision_function(X), dtype=float)
        if score.ndim == 2:
            classes = np.asarray(estimator.classes_)
            if 1 in classes:
                index = int(np.flatnonzero(classes == 1)[0])
                return score[:, index]
        return score.ravel()
    return np.asarray(estimator.predict(X), dtype=float)


def regression_to_classification(
    predicted_logbb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_logbb = np.asarray(predicted_logbb, dtype=float)
    return (predicted_logbb < threshold).astype(int), -predicted_logbb


# ---------------------------------------------------------------------------
# Metrics
def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_class: Optional[np.ndarray] = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_class_array = (
        np.full(len(y_true), np.nan)
        if y_class is None
        else np.asarray(y_class, dtype=float)
    )
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    yc = y_class_array[mask]
    if len(yt) == 0:
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
    np_mask = yc == 1
    p_mask = yc == 0
    if np.any(np_mask):
        output["np_mae"] = float(mean_absolute_error(yt[np_mask], yp[np_mask]))
    if np.any(p_mask):
        output["penetrant_mae"] = float(
            mean_absolute_error(yt[p_mask], yp[p_mask])
        )
    return output


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_nonpenetrant: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    score_nonpenetrant = np.asarray(score_nonpenetrant, dtype=float)
    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(score_nonpenetrant)
    )
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    ys = score_nonpenetrant[mask]
    if len(yt) == 0:
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
            "np_wrong_side_count": 0,
            "penetrant_wrong_side_count": 0,
        }
    wrong = yp != yt
    np_mask = yt == 1
    p_mask = yt == 0
    has_both = len(np.unique(yt)) == 2
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
            float(np.mean(yp[p_mask] == 0)) if np.any(p_mask) else np.nan
        ),
        "roc_auc": float(roc_auc_score(yt, ys)) if has_both else np.nan,
        "pr_auc_nonpenetrant": (
            float(average_precision_score(yt, ys)) if has_both else np.nan
        ),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "wrong_side_count": int(np.sum(wrong)),
        "wrong_side_rate": float(np.mean(wrong)),
        "np_wrong_side_count": int(np.sum(wrong & np_mask)),
        "penetrant_wrong_side_count": int(np.sum(wrong & p_mask)),
    }


# ---------------------------------------------------------------------------
# Data loading and folds
# ---------------------------------------------------------------------------

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


def check_feature_target_correlation(
    X: pd.DataFrame,
    y_reg: np.ndarray,
) -> pd.DataFrame:
    measured = np.isfinite(y_reg)
    rows: list[dict[str, Any]] = []
    if np.sum(measured) < 3:
        return pd.DataFrame(columns=["feature", "pearson_r", "abs_pearson_r"])
    target = y_reg[measured]
    for column in X.columns:
        values = pd.to_numeric(X.loc[measured, column], errors="coerce").to_numpy(
            float
        )
        mask = np.isfinite(values) & np.isfinite(target)
        correlation = (
            safe_pearson(values[mask], target[mask]) if np.sum(mask) >= 3 else np.nan
        )
        rows.append(
            {
                "feature": column,
                "pearson_r": correlation,
                "abs_pearson_r": (
                    abs(correlation) if np.isfinite(correlation) else np.nan
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_pearson_r", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


@dataclass
class LoadedData:
    frame: pd.DataFrame
    X: pd.DataFrame
    feature_columns: list[str]
    y_reg: np.ndarray
    y_class: np.ndarray
    outer_fold: np.ndarray
    sample_id: np.ndarray
    smiles: np.ndarray


def load_master_data(args: argparse.Namespace) -> LoadedData:
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path, low_memory=False)
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
    outer_fold = pd.to_numeric(df[args.fold_col], errors="raise").to_numpy(int)
    if not set(np.unique(y_class)).issubset({0, 1}):
        raise ValueError(
            f"{args.class_col} must use 0=penetrant and 1=non-penetrant."
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
    all_nan = X.columns[X.isna().all()].tolist()
    if all_nan:
        warnings.warn(
            f"Dropping {len(all_nan)} globally all-NaN descriptors. "
            f"Examples: {all_nan[:10]}"
        )
        X = X.drop(columns=all_nan)
        feature_columns = X.columns.tolist()

    measured = np.isfinite(y_reg)
    implied = (y_reg[measured] < args.logbb_threshold).astype(int)
    disagreement = int(np.sum(implied != y_class[measured]))
    if disagreement:
        message = (
            f"{disagreement} measured rows disagree between {args.class_col} "
            f"and logBB threshold {args.logbb_threshold}."
        )
        if args.allow_label_disagreement:
            warnings.warn(message)
        else:
            raise ValueError(
                message + " Use --allow-label-disagreement only if intentional."
            )

    return LoadedData(
        frame=df,
        X=X,
        feature_columns=feature_columns,
        y_reg=y_reg,
        y_class=y_class,
        outer_fold=outer_fold,
        sample_id=df[args.id_col].to_numpy(),
        smiles=df[args.smiles_col].astype(str).to_numpy(),
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
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for value in sorted(np.unique(values)):
        validation = np.flatnonzero(values == value)
        training = np.flatnonzero(values != value)
        if len(training) and len(validation):
            splits.append((training, validation))
    return splits if len(splits) >= 2 else None


def adaptive_regression_inner_splits(
    y: np.ndarray,
    classes: Optional[np.ndarray],
    n_splits: int,
    seed: int,
    max_bins: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=float)
    if len(y) < n_splits:
        raise ValueError(
            f"Only {len(y)} regression-labelled samples are available for "
            f"{n_splits}-fold inner CV."
        )
    candidate_strata: list[np.ndarray] = []
    classes_array = None if classes is None else np.asarray(classes)
    for n_bins in range(max_bins, 1, -1):
        try:
            bins = np.asarray(
                pd.qcut(y, q=n_bins, labels=False, duplicates="drop"),
                dtype=int,
            )
        except ValueError:
            continue
        if classes_array is not None:
            candidate_strata.append(
                np.asarray(
                    [f"class_{c}_bin_{b}" for c, b in zip(classes_array, bins)],
                    dtype=object,
                )
            )
        candidate_strata.append(
            np.asarray([f"bin_{value}" for value in bins], dtype=object)
        )
    if classes_array is not None:
        candidate_strata.append(
            np.asarray([f"class_{value}" for value in classes_array], dtype=object)
        )
    placeholder = np.zeros((len(y), 1))
    for strata in candidate_strata:
        counts = pd.Series(strata).value_counts()
        if not counts.empty and counts.min() >= n_splits:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            return list(cv.split(placeholder, strata))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(placeholder))


def classification_inner_splits(
    y_class: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = pd.Series(y_class).value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"Insufficient samples per class for {n_splits}-fold CV: "
            f"{counts.to_dict()}"
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros((len(y_class), 1)), y_class))


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def basic_numeric_pipeline(model: Any) -> Pipeline:
    """Wrap a model without modifying or imputing descriptor values."""
    return Pipeline([("model", model)])


def restrict_inner_splits_to_measured(
    inner_cv_all: list[tuple[np.ndarray, np.ndarray]],
    measured_mask: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project all-row inner folds onto the measured-logBB row subset."""
    measured_mask = np.asarray(measured_mask, dtype=bool)
    measured_positions = np.flatnonzero(measured_mask)
    full_to_measured = np.full(len(measured_mask), -1, dtype=int)
    full_to_measured[measured_positions] = np.arange(len(measured_positions))

    projected: list[tuple[np.ndarray, np.ndarray]] = []
    for fit_all, val_all in inner_cv_all:
        fit = full_to_measured[np.asarray(fit_all, dtype=int)]
        val = full_to_measured[np.asarray(val_all, dtype=int)]
        fit = fit[fit >= 0]
        val = val[val >= 0]
        if len(fit) == 0 or len(val) == 0:
            raise ValueError(
                "An inner fold contains no measured-logBB training or "
                "validation compounds. Regenerate compatible inner folds."
            )
        projected.append((fit, val))
    return projected


def tune_and_fit_mse_regressor(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    inner_cv: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_trials: int,
    boost_rounds: int,
) -> tuple[Pipeline, dict[str, Any], float, int]:
    """Tune the MSE model with the same TPE sampler and model space as MTL."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 50
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            ),
        }
        fold_mse: list[float] = []
        for inner_number, (fit_rel, val_rel) in enumerate(inner_cv):
            inner_seed = seed + 100 * inner_number
            estimator = basic_numeric_pipeline(
                lgb.LGBMRegressor(
                    objective="regression_l2",
                    boosting_type="gbdt",
                    n_estimators=boost_rounds,
                    subsample_freq=1,
                    random_state=inner_seed,
                    n_jobs=1,
                    verbosity=-1,
                    force_col_wise=True,
                    **parameters,
                )
            )
            estimator.fit(X.iloc[fit_rel], y[fit_rel])
            prediction = np.asarray(
                estimator.predict(X.iloc[val_rel]), dtype=float
            )
            fold_mse.append(float(mean_squared_error(y[val_rel], prediction)))
        return float(np.mean(fold_mse))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_parameters = dict(study.best_trial.params)
    fitted = basic_numeric_pipeline(
        lgb.LGBMRegressor(
            objective="regression_l2",
            boosting_type="gbdt",
            n_estimators=boost_rounds,
            subsample_freq=1,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            force_col_wise=True,
            **best_parameters,
        )
    )
    fitted.fit(X, y)
    return (
        fitted,
        best_parameters,
        float(study.best_value),
        int(study.best_trial.number),
    )


def select_balanced_accuracy_threshold(
    *,
    estimator: Any,
    X: pd.DataFrame,
    y_reg: np.ndarray,
    y_class: np.ndarray,
    inner_cv: list[tuple[np.ndarray, np.ndarray]],
    reference_threshold: float = -1.0,
) -> tuple[float, float, float, float]:
    """Select the classification threshold from inner-fold OOF predictions.

    The LightGBM hyperparameters are already fixed by inner-CV MSE. This
    function refits that exact estimator on measured compounds in each
    inner-training split, predicts every classification-labelled compound in
    the corresponding inner-validation split, and chooses only the scalar
    logBB threshold that maximizes balanced accuracy.

    The outer test fold is never used.
    """
    y_reg = np.asarray(y_reg, dtype=float)
    y_class = np.asarray(y_class, dtype=int)

    if len(X) != len(y_reg) or len(y_reg) != len(y_class):
        raise ValueError("X, y_reg, and y_class must have the same length.")
    oof_prediction = np.full(len(y_reg), np.nan, dtype=float)
    oof_count = np.zeros(len(y_reg), dtype=int)

    for inner_number, (fit_rel, val_rel) in enumerate(inner_cv):
        del inner_number
        measured_fit = np.asarray(fit_rel, dtype=int)[
            np.isfinite(y_reg[np.asarray(fit_rel, dtype=int)])
        ]
        if len(measured_fit) < 2:
            raise ValueError(
                "An inner-training fold has fewer than two measured-logBB rows."
            )
        fold_estimator = clone(estimator)
        fold_estimator.fit(X.iloc[measured_fit], y_reg[measured_fit])
        oof_prediction[val_rel] = np.asarray(
            fold_estimator.predict(X.iloc[val_rel]),
            dtype=float,
        )
        oof_count[val_rel] += 1

    if not np.all(oof_count == 1):
        values, counts = np.unique(oof_count, return_counts=True)
        raise RuntimeError(
            "Inner CV did not produce exactly one OOF prediction per labelled "
            f"sample. OOF-count distribution: {dict(zip(values.tolist(), counts.tolist()))}"
        )
    if not np.all(np.isfinite(oof_prediction)):
        raise RuntimeError("Non-finite inner-OOF predictions were produced.")

    unique_predictions = np.sort(np.unique(oof_prediction))
    scale = max(1.0, float(np.max(np.abs(unique_predictions))))
    epsilon = 1e-10 * scale

    if len(unique_predictions) == 1:
        candidates = np.asarray(
            [
                unique_predictions[0] - epsilon,
                reference_threshold,
                unique_predictions[0] + epsilon,
            ],
            dtype=float,
        )
    else:
        midpoints = (
            unique_predictions[:-1] + unique_predictions[1:]
        ) / 2.0
        candidates = np.concatenate(
            [
                [unique_predictions[0] - epsilon],
                midpoints,
                [unique_predictions[-1] + epsilon],
                [reference_threshold],
            ]
        )

    candidates = np.unique(candidates)
    threshold_scores: list[tuple[float, float]] = []

    for threshold in candidates:
        predicted_class = (oof_prediction < threshold).astype(int)
        score = balanced_accuracy_score(y_class, predicted_class)
        threshold_scores.append((float(threshold), float(score)))

    best_score = max(score for _, score in threshold_scores)
    tied_thresholds = [
        threshold
        for threshold, score in threshold_scores
        if np.isclose(score, best_score, rtol=0.0, atol=1e-12)
    ]

    # Conservative tie-break: among equally good thresholds, stay as close as
    # possible to the conventional logBB = -1 boundary.
    best_threshold = min(
        tied_thresholds,
        key=lambda value: abs(value - reference_threshold),
    )

    reference_class = (
        oof_prediction < reference_threshold
    ).astype(int)
    reference_ba = float(
        balanced_accuracy_score(y_class, reference_class)
    )

    tuned_class = (oof_prediction < best_threshold).astype(int)
    tuned_np_rate = float(np.mean(tuned_class == 1))

    if best_score + 1e-12 < reference_ba:
        raise RuntimeError(
            "Threshold optimization returned a lower balanced accuracy than the "
            "reference threshold, which should be impossible because the "
            "reference threshold is included among the candidates."
        )

    return (
        float(best_threshold),
        float(best_score),
        reference_ba,
        tuned_np_rate,
    )


def validate_threshold_ablation_predictions(
    predictions: pd.DataFrame,
    *,
    fixed_model: str = "lightgbm_mse_regression",
    tuned_model: str = THRESHOLD_TUNED_MODEL,
) -> None:
    """Hard validation of quantities that threshold tuning cannot change."""
    fixed = predictions.loc[predictions["model"] == fixed_model].copy()
    tuned = predictions.loc[predictions["model"] == tuned_model].copy()

    if fixed.empty or tuned.empty:
        raise RuntimeError(
            "Could not validate threshold ablation because one of the paired "
            "MSE prediction sets is missing."
        )

    fixed = fixed.sort_values(["outer_fold", "row_index"]).reset_index(drop=True)
    tuned = tuned.sort_values(["outer_fold", "row_index"]).reset_index(drop=True)

    if len(fixed) != len(tuned):
        raise RuntimeError(
            f"Fixed/tuned MSE prediction counts differ: {len(fixed)} vs {len(tuned)}."
        )

    identity_columns = [
        "sample_id",
        "row_index",
        "outer_fold",
        "bbb_class",
        "has_logBB",
        "classification_only",
    ]
    for column in identity_columns:
        if not fixed[column].equals(tuned[column]):
            raise RuntimeError(
                f"Threshold-ablation pairing failed: column {column!r} differs."
            )

    fixed_logbb = fixed["predicted_logBB"].to_numpy(float)
    tuned_logbb = tuned["predicted_logBB"].to_numpy(float)
    if not np.array_equal(fixed_logbb, tuned_logbb, equal_nan=True):
        max_diff = float(np.nanmax(np.abs(fixed_logbb - tuned_logbb)))
        raise RuntimeError(
            "Threshold tuning changed continuous MSE predictions. "
            f"Maximum absolute difference: {max_diff}."
        )

    fixed_score = fixed["score_nonpenetrant"].to_numpy(float)
    tuned_score = tuned["score_nonpenetrant"].to_numpy(float)
    if not np.array_equal(fixed_score, tuned_score, equal_nan=True):
        raise RuntimeError(
            "Threshold tuning changed score_nonpenetrant, which should remain "
            "exactly -predicted_logBB."
        )

    # Confirm each row's class is exactly the result of applying its recorded
    # threshold to the shared continuous prediction.
    for name, frame in [(fixed_model, fixed), (tuned_model, tuned)]:
        expected_class = (
            frame["predicted_logBB"].to_numpy(float)
            < frame["classification_threshold"].to_numpy(float)
        ).astype(int)
        observed_class = frame["predicted_bbb_class"].to_numpy(int)
        if not np.array_equal(expected_class, observed_class):
            raise RuntimeError(
                f"Recorded classifications for {name} do not match its recorded "
                "thresholds."
            )

    # Regression metrics and ranking metrics must be invariant to a threshold.
    subsets = {
        "classification_all": np.ones(len(fixed), dtype=bool),
        "classification_measured": fixed["has_logBB"].to_numpy(bool),
        "classification_only": fixed["classification_only"].to_numpy(bool),
    }

    fixed_reg = regression_metrics(
        fixed["true_logBB"].to_numpy(float),
        fixed_logbb,
        fixed["bbb_class"].to_numpy(float),
    )
    tuned_reg = regression_metrics(
        tuned["true_logBB"].to_numpy(float),
        tuned_logbb,
        tuned["bbb_class"].to_numpy(float),
    )
    invariant_regression_metrics = [
        "mae",
        "mse",
        "rmse",
        "median_absolute_error",
        "r2",
        "pearson_r",
        "np_mae",
        "penetrant_mae",
    ]
    for metric in invariant_regression_metrics:
        if not np.isclose(
            fixed_reg[metric],
            tuned_reg[metric],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Invariant regression metric {metric!r} differs between fixed "
                "and tuned-threshold MSE predictions."
            )

    for subset_name, mask in subsets.items():
        yt_fixed = fixed.loc[mask, "bbb_class"].to_numpy(int)
        yt_tuned = tuned.loc[mask, "bbb_class"].to_numpy(int)
        ys_fixed = fixed.loc[mask, "score_nonpenetrant"].to_numpy(float)
        ys_tuned = tuned.loc[mask, "score_nonpenetrant"].to_numpy(float)

        if not np.array_equal(yt_fixed, yt_tuned):
            raise RuntimeError(f"True classes differ in {subset_name}.")
        if not np.array_equal(ys_fixed, ys_tuned, equal_nan=True):
            raise RuntimeError(f"Ranking scores differ in {subset_name}.")

        if len(np.unique(yt_fixed)) == 2:
            roc_fixed = roc_auc_score(yt_fixed, ys_fixed)
            roc_tuned = roc_auc_score(yt_tuned, ys_tuned)
            pr_fixed = average_precision_score(yt_fixed, ys_fixed)
            pr_tuned = average_precision_score(yt_tuned, ys_tuned)

            if not np.isclose(roc_fixed, roc_tuned, rtol=0.0, atol=0.0):
                raise RuntimeError(
                    f"ROC AUC changed after threshold tuning in {subset_name}."
                )
            if not np.isclose(pr_fixed, pr_tuned, rtol=0.0, atol=0.0):
                raise RuntimeError(
                    f"PR AUC changed after threshold tuning in {subset_name}."
                )

    print(
        "Threshold-ablation validation passed: continuous predictions, "
        "regression metrics, ROC AUC, and PR AUC are unchanged."
    )



# ---------------------------------------------------------------------------
# Prediction formatting
# ---------------------------------------------------------------------------

def prediction_frame(
    *,
    data: LoadedData,
    test_indices: np.ndarray,
    model: str,
    model_task: str,
    predicted_logbb: Optional[np.ndarray],
    predicted_class: np.ndarray,
    score_nonpenetrant: np.ndarray,
    threshold: Optional[float],
) -> pd.DataFrame:
    indices = np.asarray(test_indices, dtype=int)
    logbb_values = (
        np.full(len(indices), np.nan)
        if predicted_logbb is None
        else np.asarray(predicted_logbb, dtype=float)
    )
    result = pd.DataFrame(
        {
            "sample_id": data.sample_id[indices],
            "row_index": indices,
            "smiles": data.smiles[indices],
            "outer_fold": data.outer_fold[indices],
            "true_logBB": data.y_reg[indices],
            "bbb_class": data.y_class[indices],
            "has_logBB": np.isfinite(data.y_reg[indices]),
            "classification_only": ~np.isfinite(data.y_reg[indices]),
            "model": model,
            "model_task": model_task,
            "predicted_logBB": logbb_values,
            "score_nonpenetrant": np.asarray(score_nonpenetrant, dtype=float),
            "predicted_bbb_class": np.asarray(predicted_class, dtype=int),
            "classification_threshold": threshold,
        }
    )
    result["absolute_error"] = np.where(
        result["has_logBB"],
        np.abs(result["true_logBB"] - result["predicted_logBB"]),
        np.nan,
    )
    result["squared_error"] = np.where(
        result["has_logBB"],
        (result["true_logBB"] - result["predicted_logBB"]) ** 2,
        np.nan,
    )
    result["wrong_side"] = (
        result["predicted_bbb_class"].astype(int)
        != result["bbb_class"].astype(int)
    )
    return result


# ---------------------------------------------------------------------------
# Summary tables and outputs
# ---------------------------------------------------------------------------

def evaluate_prediction_group(
    group: pd.DataFrame,
    *,
    model_task: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if model_task == "regression":
        rows.append(
            {
                "evaluation_subset": "measured_regression",
                "evaluation_task": "regression",
                **regression_metrics(
                    group["true_logBB"].to_numpy(float),
                    group["predicted_logBB"].to_numpy(float),
                    group["bbb_class"].to_numpy(float),
                ),
            }
        )
    subsets = {
        "classification_all": np.ones(len(group), dtype=bool),
        "classification_measured": group["has_logBB"].to_numpy(bool),
        "classification_only": group["classification_only"].to_numpy(bool),
    }
    for subset_name, mask in subsets.items():
        subset = group.loc[mask]
        rows.append(
            {
                "evaluation_subset": subset_name,
                "evaluation_task": "classification",
                **classification_metrics(
                    subset["bbb_class"].to_numpy(float),
                    subset["predicted_bbb_class"].to_numpy(float),
                    subset["score_nonpenetrant"].to_numpy(float),
                ),
            }
        )
    return rows


def build_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    for (model, outer_fold), group in predictions.groupby(
        ["model", "outer_fold"], sort=True
    ):
        model_task = MODEL_TASKS[model]
        for values in evaluate_prediction_group(group, model_task=model_task):
            fold_rows.append(
                {
                    "model": model,
                    "model_task": model_task,
                    "outer_fold": outer_fold,
                    **values,
                }
            )
    for model, group in predictions.groupby("model", sort=True):
        model_task = MODEL_TASKS[model]
        for values in evaluate_prediction_group(group, model_task=model_task):
            pooled_rows.append(
                {"model": model, "model_task": model_task, **values}
            )
    fold_metrics = pd.DataFrame(fold_rows)
    pooled_metrics = pd.DataFrame(pooled_rows)
    identifier_columns = {
        "model",
        "model_task",
        "outer_fold",
        "evaluation_subset",
        "evaluation_task",
    }
    metric_columns = [
        column for column in fold_metrics.columns if column not in identifier_columns
    ]
    summary_rows: list[dict[str, Any]] = []
    for (model, model_task, subset, task), group in fold_metrics.groupby(
        ["model", "model_task", "evaluation_subset", "evaluation_task"],
        sort=True,
    ):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            summary_rows.append(
                {
                    "model": model,
                    "model_task": model_task,
                    "evaluation_subset": subset,
                    "evaluation_task": task,
                    "metric": metric,
                    "n_outer_folds": int(len(values)),
                    "mean": mean,
                    "sd": sd,
                    "mean_sd": (
                        f"{mean:.6f} ± {sd:.6f}"
                        if np.isfinite(sd)
                        else f"{mean:.6f}"
                    ),
                }
            )
    return fold_metrics, pooled_metrics, pd.DataFrame(summary_rows)


def write_excel_workbook(
    path: Path,
    *,
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    mean_sd: pd.DataFrame,
    best_parameters: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            predictions.to_excel(writer, sheet_name="predictions", index=False)
            pooled_metrics.to_excel(writer, sheet_name="pooled_metrics", index=False)
            fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
            mean_sd.to_excel(writer, sheet_name="mean_sd", index=False)
            best_parameters.to_excel(
                writer, sheet_name="best_parameters", index=False
            )
            pd.DataFrame(
                {
                    "key": list(manifest.keys()),
                    "value": [
                        json.dumps(json_safe(value))
                        if isinstance(value, (dict, list))
                        else value
                        for value in manifest.values()
                    ],
                }
            ).to_excel(writer, sheet_name="run_manifest", index=False)
    except ImportError as exc:
        warnings.warn(
            "openpyxl is not installed, so the XLSX workbook was not written. "
            f"CSV outputs are complete. Original error: {exc}"
        )


def prepare_output_directory(
    output_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} is not empty. Use --overwrite or choose a new directory."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "fitted_models"
    checkpoint_dir = output_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, model_dir, checkpoint_dir


def save_checkpoint(
    checkpoint_dir: Path,
    prediction_parts: list[pd.DataFrame],
    parameter_rows: list[dict[str, Any]],
) -> None:
    if prediction_parts:
        pd.concat(prediction_parts, ignore_index=True).to_csv(
            checkpoint_dir / "trained_models_predictions.partial.csv",
            index=False,
        )
    if parameter_rows:
        pd.DataFrame(parameter_rows).to_csv(
            checkpoint_dir / "best_parameters.partial.csv",
            index=False,
        )


# ---------------------------------------------------------------------------
# Arguments and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_model_results"))
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--class-col", default=DEFAULT_CLASS_COL)
    parser.add_argument("--fold-col", default=DEFAULT_FOLD_COL)
    parser.add_argument("--smiles-col", default=DEFAULT_SMILES_COL)
    parser.add_argument("--id-col", default=DEFAULT_ID_COL)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Retained for command-line compatibility. Optuna trials and "
            "LightGBM fits run sequentially for deterministic matching to script 04."
        ),
    )
    parser.add_argument("--logbb-threshold", type=float, default=-1.0)
    parser.add_argument("--lgbm-search-iterations", type=int, default=30)
    parser.add_argument(
        "--lgbm-boost-rounds",
        type=int,
        default=1000,
        help="Fixed boosting rounds, matched to --mtl-boost-rounds in script 04.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated trainable model names, or 'all'. Choices: "
            + ", ".join(TRAINABLE_MODEL_TASKS)
        ),
    )
    parser.add_argument("--skip-model-saving", action="store_true")
    parser.add_argument("--allow-label-disagreement", action="store_true")
    parser.add_argument(
        "--allow-high-target-correlation",
        action="store_true",
        help=(
            "Allow descriptors with |Pearson r| >= 0.98 against measured logBB."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_models(specification: str) -> list[str]:
    if specification.strip().lower() == "all":
        return list(TRAINABLE_MODEL_TASKS)
    requested = [token.strip() for token in specification.split(",") if token.strip()]
    unknown = sorted(set(requested).difference(TRAINABLE_MODEL_TASKS))
    if unknown:
        raise ValueError(
            f"Unknown trainable models: {unknown}. "
            f"Choices: {sorted(TRAINABLE_MODEL_TASKS)}"
        )
    return requested


def main() -> None:
    args = parse_args()
    models = selected_models(args.models)
    
    output_dir, model_dir, checkpoint_dir = prepare_output_directory(args.output_dir, args.overwrite)
    data = load_master_data(args)

    correlation_check = check_feature_target_correlation(data.X, data.y_reg)
    correlation_check.to_csv(output_dir / "feature_target_correlation_check.csv", index=False)
    
    suspicious = correlation_check.loc[correlation_check["abs_pearson_r"] >= 0.98]
    
    if len(suspicious) and not args.allow_high_target_correlation:
        examples = suspicious[["feature", "pearson_r"]].head(10)
        raise RuntimeError(
            "Potential target leakage detected. At least one descriptor has "
            "|Pearson r| >= 0.98 against measured logBB:\n"
            + examples.to_string(index=False)
            + "\nUse --allow-high-target-correlation only after checking."
        )

    outer_folds = sorted(np.unique(data.outer_fold).tolist())
    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    for fold_position, outer_fold in enumerate(outer_folds):
        print(f"outer fold: {outer_fold}")

        train_mask = data.outer_fold != outer_fold
        test_mask = data.outer_fold == outer_fold
        train_indices_all = np.flatnonzero(train_mask)
        test_indices = np.flatnonzero(test_mask)
        X_test_df = data.X.iloc[test_indices]
        seed = args.seed + 1000 * fold_position

        if "dummy_mean_regression" in models:
            fit_indices = np.flatnonzero(train_mask & np.isfinite(data.y_reg))
            estimator = basic_numeric_pipeline(DummyRegressor(strategy="mean"))
            estimator.fit(data.X.iloc[fit_indices], data.y_reg[fit_indices])
            predicted_logbb = np.asarray(estimator.predict(X_test_df), dtype=float)
            predicted_class, score_np = regression_to_classification(
                predicted_logbb, args.logbb_threshold
            )
            prediction_parts.append(
                prediction_frame(
                    data=data,
                    test_indices=test_indices,
                    model="dummy_mean_regression",
                    model_task="regression",
                    predicted_logbb=predicted_logbb,
                    predicted_class=predicted_class,
                    score_nonpenetrant=score_np,
                    threshold=args.logbb_threshold,
                )
            )
            parameter_rows.append(
                {
                    "model": "dummy_mean_regression",
                    "outer_fold": outer_fold,
                    "best_inner_score": np.nan,
                    "parameters_json": json.dumps({"strategy": "mean"}),
                    "training_location": "this_script",
                }
            )
            if not args.skip_model_saving:
                joblib.dump(
                    estimator,
                    model_dir / f"dummy_mean_regression_threshold_outer_{outer_fold}.joblib",
                    compress=3,
                )
            save_checkpoint(checkpoint_dir, prediction_parts, parameter_rows)

        if "lightgbm_mse_regression" in models:
            X_train_all = data.X.iloc[train_indices_all].reset_index(drop=True)
            usable_features = X_train_all.columns[
                ~X_train_all.isna().all(axis=0)
            ].tolist()
            if not usable_features:
                raise ValueError(
                    f"No usable descriptors remain in outer fold {outer_fold}."
                )
            X_train_all = X_train_all.loc[:, usable_features]
            X_test_mse = X_test_df.loc[:, usable_features]
            y_reg_train_all = data.y_reg[train_indices_all]
            y_class_train_all = data.y_class[train_indices_all]
            measured_train_mask = np.isfinite(y_reg_train_all)
            measured_positions = np.flatnonzero(measured_train_mask)
            X_fit = X_train_all.iloc[measured_positions]
            y_fit = y_reg_train_all[measured_positions]

            inner_cv_all = saved_inner_splits(
                data.frame,
                outer_fold=outer_fold,
                fit_indices=train_indices_all,
            )
            if inner_cv_all is None:
                inner_cv_all = classification_inner_splits(
                    y_class_train_all,
                    n_splits=args.inner_folds,
                    seed=seed,
                )
            inner_cv_measured = restrict_inner_splits_to_measured(
                inner_cv_all,
                measured_train_mask,
            )

            # -------------------------------------------------------------
            # A. Select MSE hyperparameters with the same structural search
            #    space, TPE sampler, folds, seeds, and trial budget as MTL.
            # -------------------------------------------------------------
            (
                fitted,
                best_mse_parameters,
                best_inner_mse,
                best_trial_number,
            ) = tune_and_fit_mse_regressor(
                X=X_fit,
                y=y_fit,
                inner_cv=inner_cv_measured,
                seed=seed,
                n_trials=args.lgbm_search_iterations,
                boost_rounds=args.lgbm_boost_rounds,
            )

            # -------------------------------------------------------------
            # B. Freeze those MSE-selected hyperparameters and tune ONLY the
            #    scalar classification threshold using inner-fold OOF
            #    predictions from every classification-labelled compound.
            # -------------------------------------------------------------
            (
                tuned_threshold,
                tuned_inner_ba,
                fixed_inner_ba,
                tuned_inner_np_rate,
            ) = select_balanced_accuracy_threshold(
                estimator=fitted,
                X=X_train_all,
                y_reg=y_reg_train_all,
                y_class=y_class_train_all,
                inner_cv=inner_cv_all,
                reference_threshold=args.logbb_threshold,
            )

            print(
                f"  threshold: fixed={args.logbb_threshold:.6f}, "
                f"inner-OOF tuned={tuned_threshold:.6f}"
            )
            print(
                f"  inner-OOF balanced accuracy: "
                f"fixed={fixed_inner_ba:.6f}, tuned={tuned_inner_ba:.6f}"
            )
            print(
                f"  inner-OOF predicted non-penetrant rate at tuned "
                f"threshold={tuned_inner_np_rate:.6f}"
            )

            # -------------------------------------------------------------
            # C. Generate ONE set of continuous outer-test predictions.
            #    Both comparator rows below use this exact same array.
            # -------------------------------------------------------------
            predicted_logbb = np.asarray(
                fitted.predict(X_test_mse), dtype=float
            )
            score_np_shared = -predicted_logbb

            # Original MSE comparator: conventional threshold = -1.
            predicted_class_fixed = (
                predicted_logbb < args.logbb_threshold
            ).astype(int)
            prediction_parts.append(
                prediction_frame(
                    data=data,
                    test_indices=test_indices,
                    model="lightgbm_mse_regression",
                    model_task="regression",
                    predicted_logbb=predicted_logbb,
                    predicted_class=predicted_class_fixed,
                    score_nonpenetrant=score_np_shared,
                    threshold=args.logbb_threshold,
                )
            )
            parameter_rows.append(
                {
                    "model": "lightgbm_mse_regression",
                    "outer_fold": outer_fold,
                    "best_inner_score": float(-best_inner_mse),
                    "best_inner_mse": float(best_inner_mse),
                    "selected_trial_number": int(best_trial_number),
                    "parameters_json": json.dumps(
                        json_safe(best_mse_parameters), sort_keys=True
                    ),
                    "n_estimators": int(args.lgbm_boost_rounds),
                    "n_selected_features": int(len(usable_features)),
                    "classification_threshold": float(args.logbb_threshold),
                    "threshold_selection": "fixed",
                    "training_location": "this_script",
                }
            )

            # Threshold-ablation comparator: SAME MSE model and SAME continuous
            # predictions; only the decision threshold changes.
            predicted_class_tuned = (
                predicted_logbb < tuned_threshold
            ).astype(int)
            prediction_parts.append(
                prediction_frame(
                    data=data,
                    test_indices=test_indices,
                    model=THRESHOLD_TUNED_MODEL,
                    model_task="regression",
                    predicted_logbb=predicted_logbb,
                    predicted_class=predicted_class_tuned,
                    score_nonpenetrant=score_np_shared,
                    threshold=tuned_threshold,
                )
            )
            parameter_rows.append(
                {
                    "model": THRESHOLD_TUNED_MODEL,
                    "outer_fold": outer_fold,
                    "best_inner_score": float(-best_inner_mse),
                    "best_inner_mse": float(best_inner_mse),
                    "selected_trial_number": int(best_trial_number),
                    "parameters_json": json.dumps(
                        json_safe(best_mse_parameters), sort_keys=True
                    ),
                    "n_estimators": int(args.lgbm_boost_rounds),
                    "n_selected_features": int(len(usable_features)),
                    "classification_threshold": float(tuned_threshold),
                    "threshold_selection": "inner_oof_max_balanced_accuracy",
                    "inner_balanced_accuracy_fixed_threshold": float(
                        fixed_inner_ba
                    ),
                    "inner_balanced_accuracy_tuned_threshold": float(
                        tuned_inner_ba
                    ),
                    "inner_predicted_nonpenetrant_rate_tuned": float(
                        tuned_inner_np_rate
                    ),
                    "training_location": "this_script",
                }
            )

            if not args.skip_model_saving:
                # Only one fitted regressor is saved because the two MSE
                # comparators are deliberately the exact same model.
                joblib.dump(
                    fitted,
                    model_dir
                    / f"lightgbm_mse_regression_threshold_outer_{outer_fold}.joblib",
                    compress=3,
                )
                threshold_metadata = {
                    "outer_fold": int(outer_fold),
                    "fixed_threshold": float(args.logbb_threshold),
                    "tuned_threshold": float(tuned_threshold),
                    "threshold_selection": "inner_oof_max_balanced_accuracy",
                    "inner_balanced_accuracy_fixed": float(fixed_inner_ba),
                    "inner_balanced_accuracy_tuned": float(tuned_inner_ba),
                    "inner_predicted_nonpenetrant_rate_tuned": float(
                        tuned_inner_np_rate
                    ),
                }
                (
                    model_dir
                    / f"lightgbm_mse_regression_threshold_outer_{outer_fold}.json"
                ).write_text(
                    json.dumps(json_safe(threshold_metadata), indent=2)
                )

            save_checkpoint(checkpoint_dir, prediction_parts, parameter_rows)

        if "dummy_prior_classification" in models:
            estimator = basic_numeric_pipeline(DummyClassifier(strategy="prior"))
            estimator.fit(
                data.X.iloc[train_indices_all], data.y_class[train_indices_all]
            )
            predicted_class = np.asarray(estimator.predict(X_test_df), dtype=int)
            score_np = class_score_from_estimator(estimator, X_test_df)
            prediction_parts.append(
                prediction_frame(
                    data=data,
                    test_indices=test_indices,
                    model="dummy_prior_classification",
                    model_task="classification",
                    predicted_logbb=None,
                    predicted_class=predicted_class,
                    score_nonpenetrant=score_np,
                    threshold=0.5,
                )
            )
            prior = float(np.mean(data.y_class[train_indices_all] == 1))
            parameter_rows.append(
                {
                    "model": "dummy_prior_classification",
                    "outer_fold": outer_fold,
                    "best_inner_score": np.nan,
                    "parameters_json": json.dumps(
                        {
                            "strategy": "prior",
                            "training_nonpenetrant_prior": prior,
                        }
                    ),
                    "training_location": "this_script",
                }
            )
            if not args.skip_model_saving:
                joblib.dump(
                    estimator,
                    model_dir / f"dummy_prior_classification_threshold_outer_{outer_fold}.joblib",
                    compress=3,
                )
            save_checkpoint(checkpoint_dir, prediction_parts, parameter_rows)

    if not prediction_parts:
        raise RuntimeError("No predictions were generated.")

    trained_predictions = pd.concat(prediction_parts, ignore_index=True)
    trained_predictions = trained_predictions.sort_values(
        ["model", "row_index"], kind="stable"
    ).reset_index(drop=True)
    expected = len(data.frame)
    for model in models:
        group = trained_predictions.loc[trained_predictions["model"] == model]
        if len(group) != expected or group["row_index"].duplicated().any():
            raise RuntimeError(
                f"{model} does not have exactly {expected} unique OOF predictions."
            )
    trained_predictions.to_csv(
        output_dir / "trained_models_predictions_.csv", index=False
    )

    output_models = list(models)
    if "lightgbm_mse_regression" in models:
        output_models.append(THRESHOLD_TUNED_MODEL)
    predictions = trained_predictions.copy()
    predictions = predictions.sort_values(
        ["model", "row_index"], kind="stable"
    ).reset_index(drop=True)
    for model in output_models:
        group = predictions.loc[predictions["model"] == model]
        if len(group) != expected or group["row_index"].duplicated().any():
            raise RuntimeError(
                f"{model} does not have exactly {expected} unique OOF predictions."
            )

    if "lightgbm_mse_regression" in models:
        validate_threshold_ablation_predictions(predictions)

    fold_metrics, pooled_metrics, mean_sd = build_metric_tables(predictions)
    best_parameters = pd.DataFrame(parameter_rows).sort_values(
        ["model", "outer_fold"], kind="stable"
    )
    manifest = {
        "input_file": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input.resolve()),
        "output_directory": str(output_dir),
        "n_samples": int(len(data.frame)),
        "n_measured_logbb": int(np.sum(np.isfinite(data.y_reg))),
        "n_classification_only": int(np.sum(~np.isfinite(data.y_reg))),
        "n_features_loaded": int(len(data.feature_columns)),
        "outer_folds": outer_folds,
        "models_trained_here": models,
        "models_in_final_output": output_models,
        "class_encoding": {"0": "BBB penetrant", "1": "BBB non-penetrant"},
        "logbb_threshold": args.logbb_threshold,
        "seed": args.seed,
        "inner_folds": args.inner_folds,
        "lgbm_search_iterations": args.lgbm_search_iterations,
        "lgbm_boost_rounds": args.lgbm_boost_rounds,
        "mse_threshold_ablation": {
            "enabled": "lightgbm_mse_regression" in models,
            "tuned_model_name": THRESHOLD_TUNED_MODEL,
            "selection_data": (
                "inner-fold OOF predictions on all classification-labelled "
                "outer-training rows; each inner model fitted only on measured-logBB rows"
            ),
            "selection_metric": "balanced_accuracy",
            "continuous_model_identical_to_fixed_threshold_mse": True,
        },
        "missing_value_policy": "native LightGBM NaN handling; no imputation",
        "matched_mse_ablation": {
            "structural_search_space": "matched to custom-MTL script 04",
            "sampler": "Optuna TPESampler",
            "boost_rounds": int(args.lgbm_boost_rounds),
            "search_iterations": int(args.lgbm_search_iterations),
            "note": "Use the same iteration count as --mtl-trials for the paired comparison.",
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "optuna": optuna.__version__,
            "joblib": joblib.__version__,
        },
    }

    predictions.to_csv(output_dir / "all_models_predictions.csv", index=False)
    pooled_metrics.to_csv(
        output_dir / "all_models_pooled_metrics.csv", index=False
    )
    fold_metrics.to_csv(output_dir / "all_models_fold_metrics.csv", index=False)
    mean_sd.to_csv(output_dir / "all_models_mean_sd.csv", index=False)
    best_parameters.to_csv(
        output_dir / "all_models_best_parameters.csv", index=False
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2)
    )
    write_excel_workbook(
        output_dir / "all_models_predictions.xlsx",
        predictions=predictions,
        pooled_metrics=pooled_metrics,
        fold_metrics=fold_metrics,
        mean_sd=mean_sd,
        best_parameters=best_parameters,
        manifest=manifest,
    )

    print("\nCompleted successfully.")
    print("Trained-only predictions:", output_dir / "trained_models_predictions.csv")
    print("Combined predictions:", output_dir / "all_models_predictions.csv")
    print("Pooled metrics:", output_dir / "all_models_pooled_metrics.csv")
    print("Mean ± SD:", output_dir / "all_models_mean_sd.csv")


if __name__ == "__main__":
    main()
