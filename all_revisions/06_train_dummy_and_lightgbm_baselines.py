#!/usr/bin/env python3
"""
Train the three requested BBB baseline comparators.

Models
    1. dummy_mean_regression
    2. lightgbm_mse_regression
    3. dummy_prior_classification

python ./06_train_dummy_and_lightgbm_baselines.py \
    --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
    --output-dir baseline_model_results \
    --n-jobs 4 
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score,average_precision_score,balanced_accuracy_score,cohen_kappa_score,f1_score,matthews_corrcoef,mean_absolute_error,
                            mean_squared_error,median_absolute_error,precision_score,r2_score,recall_score,roc_auc_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
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

MODEL_TASKS = dict(TRAINABLE_MODEL_TASKS)

LGBM_PARAMETER_DISTRIBUTIONS = {
    "model__n_estimators": randint(150, 1500),
    "model__learning_rate": loguniform(0.01, 0.2),
    "model__num_leaves": randint(7, 128),
    "model__max_depth": [-1, 4, 6, 8, 10, 12],
    "model__min_child_samples": randint(5, 51),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.5, 0.5),
    "model__reg_alpha": loguniform(1e-8, 10.0),
    "model__reg_lambda": loguniform(1e-8, 10.0),
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
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("model", model),
        ]
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
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
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
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--logbb-threshold", type=float, default=-1.0)
    parser.add_argument("--lgbm-search-iterations", type=int, default=30)
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
                    model_dir / f"dummy_mean_regression_outer_{outer_fold}.joblib",
                    compress=3,
                )
            save_checkpoint(checkpoint_dir, prediction_parts, parameter_rows)

        if "lightgbm_mse_regression" in models:
            fit_indices = np.flatnonzero(train_mask & np.isfinite(data.y_reg))
            X_fit = data.X.iloc[fit_indices]
            y_fit = data.y_reg[fit_indices]
            class_fit = data.y_class[fit_indices]
            inner_cv = saved_inner_splits(
                data.frame, outer_fold=outer_fold, fit_indices=fit_indices
            )
            if inner_cv is None:
                inner_cv = adaptive_regression_inner_splits(
                    y=y_fit,
                    classes=class_fit,
                    n_splits=args.inner_folds,
                    seed=seed,
                )
            estimator = basic_numeric_pipeline(
                lgb.LGBMRegressor(
                    objective="regression_l2",
                    boosting_type="gbdt",
                    random_state=seed,
                    n_jobs=1,
                    verbosity=-1,
                )
            )
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=LGBM_PARAMETER_DISTRIBUTIONS,
                n_iter=args.lgbm_search_iterations,
                scoring="neg_mean_squared_error",
                cv=inner_cv,
                refit=True,
                n_jobs=args.n_jobs,
                random_state=seed,
                verbose=1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_fit, y_fit)
            fitted = search.best_estimator_
            predicted_logbb = np.asarray(fitted.predict(X_test_df), dtype=float)
            predicted_class, score_np = regression_to_classification(
                predicted_logbb, args.logbb_threshold
            )
            prediction_parts.append(
                prediction_frame(
                    data=data,
                    test_indices=test_indices,
                    model="lightgbm_mse_regression",
                    model_task="regression",
                    predicted_logbb=predicted_logbb,
                    predicted_class=predicted_class,
                    score_nonpenetrant=score_np,
                    threshold=args.logbb_threshold,
                )
            )
            parameter_rows.append(
                {
                    "model": "lightgbm_mse_regression",
                    "outer_fold": outer_fold,
                    "best_inner_score": float(search.best_score_),
                    "parameters_json": json.dumps(
                        json_safe(search.best_params_), sort_keys=True
                    ),
                    "training_location": "this_script",
                }
            )
            if not args.skip_model_saving:
                joblib.dump(
                    fitted,
                    model_dir / f"lightgbm_mse_regression_outer_{outer_fold}.joblib",
                    compress=3,
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
                    model_dir / f"dummy_prior_classification_outer_{outer_fold}.joblib",
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
        output_dir / "trained_models_predictions.csv", index=False
    )

    output_models = list(models)
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
