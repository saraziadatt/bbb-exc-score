#!/usr/bin/env python3
"""
Paired model-comparison analysis for five prespecified BBB models.

Inferential procedure
---------------------
- Statistical significance is evaluated only with a paired, two-sided
  permutation test. For each permutation, the two models' predictions are
  randomly exchanged within each compound and the metric difference is
  recalculated.
- Holm correction is applied separately within every unique model pair and
  evaluation dataset (subset), across all metrics evaluated for that pair and
  dataset.
- Paired bootstrap resampling, stratified by original outer fold × BBB class,
  is used only to estimate confidence intervals for metric differences. It is
  not used to calculate p-values.
- Only the prespecified manuscript metrics are evaluated:
  balanced accuracy, non-penetrant recall, non-penetrant precision,
  non-penetrant F1, ROC AUC, non-penetrant PR AUC, MCC, non-penetrant
  wrong-side count, MAE, MSE, Pearson r, R², and non-penetrant MAE.

Models
------
1. Dummy mean regression
2. Dummy prior classification
3. LightGBM MSE regression
4. Clean custom-MTL LightGBM
5. Retrained logBB_pred (Shaker et al.)

python ./07_evaluation_and_significance_testing.py --predictions-file ./all_models/all_models_predictions.csv

"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MARGIN = -1.0

REGRESSION_METRICS = [
    "mae",
    "mse",
    "pearson_r",
    "r2",
    "np_mae",
]

CLASSIFICATION_METRICS = [
    "balanced_accuracy",
    "np_recall",
    "np_precision",
    "np_f1",
    "roc_auc",
    "pr_auc_nonpenetrant",
    "mcc",
    "cohens_kappa",
    "np_wrong_side_count",
]

LOWER_IS_BETTER = {
    "mae",
    "mse",
    "np_mae",
    "np_wrong_side_count",
}

HIGHER_IS_BETTER = {
    "pearson_r",
    "r2",
    "balanced_accuracy",
    "np_recall",
    "np_precision",
    "np_f1",
    "roc_auc",
    "pr_auc_nonpenetrant",
    "mcc",
    "cohens_kappa"
}

SUBSET_LABELS = {
    "measured_regression": "Measured compounds: regression",
    "measured_threshold": "Measured compounds: threshold-derived",
    "all_threshold": "All labelled compounds: threshold-derived",
    "classification_only_threshold": (
        "Classification-only compounds: threshold-derived"
    ),
}

MODEL_ORDER = [
    "dummy_mean_regression",
    "dummy_prior_classification",
    "lightgbm_mse_regression",
    "custom_mtl_original",
    "logbb_pred_shaker_retrained",
]

MODEL_DISPLAY = {
    "dummy_mean_regression": "Dummy regression mean",
    "dummy_prior_classification": "Dummy classification baseline",
    "lightgbm_mse_regression": "LightGBM MSE",
    "custom_mtl_original": "Custom MTL LightGBM clean",
    "logbb_pred_shaker_retrained": (
        "Matched-retrained logBB_pred strategy (Shaker et al.)"
    ),
}

REGRESSION_MODELS = {
    "dummy_mean_regression",
    "lightgbm_mse_regression",
    "custom_mtl_original",
    "logbb_pred_shaker_retrained",
}

DUMMY_CLASSIFIER_MODEL = "dummy_prior_classification"

LOGBB_MODEL_ALIASES = {
    "logbb_pred_shaker_retrained": "logbb_pred_shaker_retrained",
    "logbb_pred_matched_retraining": "logbb_pred_shaker_retrained",
    "logbb_pred_reproduction_fold_safe": "logbb_pred_shaker_retrained",
    "logbb_pred_reproduction": "logbb_pred_shaker_retrained",
}



def first_existing_column(frame: pd.DataFrame, candidates: list[str]):
    lower = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def metric_suffix(metric: str) -> str:
    for prefix in ("np_", "penetrant_"):
        if metric.startswith(prefix):
            return metric[len(prefix) :]
    return metric


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    suffix = metric_suffix(metric)
    if suffix in LOWER_IS_BETTER:
        return "lower"
    if suffix in HIGHER_IS_BETTER:
        return "higher"
    raise KeyError(f"No direction defined for metric {metric!r}.")


def metric_improvement(
    focus_value: np.ndarray | float,
    comparator_value: np.ndarray | float,
    metric: str,
):
    """Positive means focus is better than comparator."""
    if metric_direction(metric) == "lower":
        return np.asarray(comparator_value) - np.asarray(focus_value)
    return np.asarray(focus_value) - np.asarray(comparator_value)


def holm_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(adjusted, index=values.index)

    ordered = valid[np.argsort(p[valid])]
    running_max = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return pd.Series(adjusted, index=values.index)


def stable_seed(base_seed: int, *values: str) -> int:
    digest = hashlib.sha256("::".join(values).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def make_stratified_bootstrap_matrix(
    folds: np.ndarray,
    classes: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Return paired bootstrap indices stratified by outer fold × class."""
    folds = np.asarray(folds, dtype=int)
    classes = np.asarray(classes, dtype=int)
    if len(folds) != len(classes):
        raise ValueError("folds and classes have different lengths.")

    labels = np.asarray(
        [
            f"fold_{fold}_class_{label}"
            for fold, label in zip(folds, classes)
        ],
        dtype=object,
    )
    strata = [
        np.flatnonzero(labels == label)
        for label in np.unique(labels)
    ]
    if any(len(indices) == 0 for indices in strata):
        raise RuntimeError("An empty bootstrap stratum was generated.")

    rng = np.random.default_rng(seed)
    index_dtype = np.uint16 if len(folds) < 65535 else np.uint32
    sampled_parts = [
        rng.choice(
            indices,
            size=(n_bootstrap, len(indices)),
            replace=True,
        ).astype(index_dtype)
        for indices in strata
    ]
    matrix = np.concatenate(sampled_parts, axis=1)
    if matrix.shape != (n_bootstrap, len(folds)):
        raise RuntimeError(
            f"Unexpected bootstrap shape {matrix.shape}; "
            f"expected {(n_bootstrap, len(folds))}."
        )
    return matrix


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return np.where(denominator == 0, np.nan, result)


def pearson_rows(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    p_centered = prediction - np.mean(
        prediction, axis=1, keepdims=True
    )
    numerator = np.sum(y_centered * p_centered, axis=1)
    denominator = np.sqrt(
        np.sum(y_centered**2, axis=1)
        * np.sum(p_centered**2, axis=1)
    )
    return safe_divide(numerator, denominator)


def regression_metrics_rows(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate only the four prespecified overall regression metrics."""
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = prediction - y
    absolute_error = np.abs(error)
    mse = np.mean(error**2, axis=1)
    y_mean = np.mean(y, axis=1, keepdims=True)
    sst = np.sum((y - y_mean) ** 2, axis=1)
    sse = np.sum(error**2, axis=1)
    return {
        "mae": np.mean(absolute_error, axis=1),
        "mse": mse,
        "pearson_r": pearson_rows(y, prediction),
        "r2": 1.0 - safe_divide(sse, sst),
    }


def observed_regression_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    values = regression_metrics_rows(
        np.asarray(y, dtype=float)[None, :],
        np.asarray(prediction, dtype=float)[None, :],
    )
    return {key: float(value[0]) for key, value in values.items()}


def bootstrap_regression_metrics(
    *,
    y_true: np.ndarray,
    y_class: np.ndarray,
    prediction: np.ndarray,
    bootstrap_index: np.ndarray,
    chunk_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Bootstrap only MAE, MSE, Pearson r, R², and non-penetrant MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_class = np.asarray(y_class, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    if not (len(y_true) == len(y_class) == len(prediction)):
        raise ValueError("Regression arrays have different lengths.")

    np_mask = y_class == 1
    if not np.any(np_mask):
        raise ValueError("Non-penetrant MAE requires at least one NP compound.")

    observed = observed_regression_metrics(y_true, prediction)
    observed["np_mae"] = observed_regression_metrics(
        y_true[np_mask], prediction[np_mask]
    )["mae"]
    distributions = {
        metric: np.empty(len(bootstrap_index), dtype=np.float32)
        for metric in REGRESSION_METRICS
    }

    for start in range(0, len(bootstrap_index), chunk_size):
        stop = min(start + chunk_size, len(bootstrap_index))
        index = bootstrap_index[start:stop]
        y_boot = y_true[index]
        class_boot = y_class[index]
        prediction_boot = prediction[index]

        overall = regression_metrics_rows(y_boot, prediction_boot)
        for metric, values in overall.items():
            distributions[metric][start:stop] = values.astype(np.float32)

        np_boot_mask = class_boot == 1
        np_counts = np.sum(np_boot_mask, axis=1)
        if not np.all(np_counts == np_counts[0]):
            raise RuntimeError(
                "Class-stratified bootstrap did not preserve NP count."
            )
        np_size = int(np_counts[0])
        y_np = y_boot[np_boot_mask].reshape(stop - start, np_size)
        prediction_np = prediction_boot[np_boot_mask].reshape(
            stop - start, np_size
        )
        distributions["np_mae"][start:stop] = np.mean(
            np.abs(prediction_np - y_np), axis=1
        ).astype(np.float32)

    return observed, distributions


def average_precision_rows(
    score: np.ndarray,
    positive: np.ndarray,
) -> np.ndarray:
    """
    Tie-aware average precision for rows of samples.

    This reproduces the threshold-group definition used by average precision:
    each distinct score contributes its recall increment multiplied by precision
    at the end of that tied score group. A constant dummy score therefore gives
    AP equal to the positive prevalence.
    """
    order = np.argsort(-score, axis=1, kind="stable")
    sorted_score = np.take_along_axis(score, order, axis=1)
    sorted_positive = np.take_along_axis(
        positive.astype(np.int8), order, axis=1
    )
    cumulative_tp = np.cumsum(sorted_positive, axis=1).astype(float)
    ranks = np.arange(1, score.shape[1] + 1, dtype=float)[None, :]

    starts = np.ones_like(sorted_positive, dtype=bool)
    starts[:, 1:] = sorted_score[:, 1:] != sorted_score[:, :-1]
    ends = np.ones_like(sorted_positive, dtype=bool)
    ends[:, :-1] = sorted_score[:, :-1] != sorted_score[:, 1:]

    positions = np.arange(score.shape[1], dtype=int)[None, :]
    start_positions = np.maximum.accumulate(
        np.where(starts, positions, 0),
        axis=1,
    )
    previous_positions = np.maximum(start_positions - 1, 0)
    previous_tp = np.take_along_axis(
        cumulative_tp, previous_positions, axis=1
    )
    previous_tp = np.where(start_positions == 0, 0.0, previous_tp)
    group_positive = cumulative_tp - previous_tp

    n_positive = np.sum(positive, axis=1).astype(float)
    precision_at_end = safe_divide(cumulative_tp, ranks)
    recall_increment = safe_divide(group_positive, n_positive[:, None])
    return np.sum(
        np.where(ends, precision_at_end * recall_increment, 0.0),
        axis=1,
    )


def classification_metrics_rows(
    *,
    y_class: np.ndarray,
    predicted_class: np.ndarray,
    np_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate only the eight prespecified classification metrics."""
    y_class = np.asarray(y_class, dtype=int)
    predicted_class = np.asarray(predicted_class, dtype=int)
    np_score = np.asarray(np_score, dtype=float)

    if not (y_class.shape == predicted_class.shape == np_score.shape):
        raise ValueError("Classification metric arrays have different shapes.")

    positive = y_class == 1
    negative = ~positive
    predicted_positive = predicted_class == 1
    predicted_negative = ~predicted_positive

    tp = np.sum(positive & predicted_positive, axis=1).astype(float)
    fn = np.sum(positive & predicted_negative, axis=1).astype(float)
    fp = np.sum(negative & predicted_positive, axis=1).astype(float)
    tn = np.sum(negative & predicted_negative, axis=1).astype(float)

    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = safe_divide(mcc_numerator, mcc_denominator)

    n_samples = tp + fn + fp + tn

    observed_agreement = safe_divide(
        tp + tn,
        n_samples,
    )

    expected_agreement = safe_divide(
        (tp + fn) * (tp + fp)
        + (tn + fp) * (tn + fn),
        n_samples**2,
    )

    cohens_kappa = safe_divide(
        observed_agreement - expected_agreement,
        1.0 - expected_agreement,
)

    score_ranks = rankdata(np_score, axis=1, method="average")
    n_positive = tp + fn
    n_negative = tn + fp
    positive_rank_sum = np.sum(score_ranks * positive, axis=1)
    roc_auc = safe_divide(
        positive_rank_sum - n_positive * (n_positive + 1.0) / 2.0,
        n_positive * n_negative,
    )
    average_precision = average_precision_rows(np_score, positive)

    return {
        "balanced_accuracy": balanced_accuracy,
        "np_recall": recall,
        "np_precision": precision,
        "np_f1": f1,
        "roc_auc": roc_auc,
        "pr_auc_nonpenetrant": average_precision,
        "mcc": mcc,
        "cohens_kappa": cohens_kappa,
        "np_wrong_side_count": fn,
    }


def observed_classification_metrics(
    *,
    y_class: np.ndarray,
    predicted_class: np.ndarray,
    np_score: np.ndarray,
) -> dict[str, float]:
    values = classification_metrics_rows(
        y_class=np.asarray(y_class, dtype=int)[None, :],
        predicted_class=np.asarray(predicted_class, dtype=int)[None, :],
        np_score=np.asarray(np_score, dtype=float)[None, :],
    )
    return {key: float(value[0]) for key, value in values.items()}


def bootstrap_classification_metrics(
    *,
    y_class: np.ndarray,
    predicted_class: np.ndarray,
    np_score: np.ndarray,
    bootstrap_index: np.ndarray,
    chunk_size: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    observed = observed_classification_metrics(
        y_class=y_class,
        predicted_class=predicted_class,
        np_score=np_score,
    )
    distributions = {
        metric: np.empty(len(bootstrap_index), dtype=np.float32)
        for metric in CLASSIFICATION_METRICS
    }

    y_class = np.asarray(y_class, dtype=int)
    predicted_class = np.asarray(predicted_class, dtype=int)
    np_score = np.asarray(np_score, dtype=float)

    for start in range(0, len(bootstrap_index), chunk_size):
        stop = min(start + chunk_size, len(bootstrap_index))
        index = bootstrap_index[start:stop]
        values = classification_metrics_rows(
            y_class=y_class[index],
            predicted_class=predicted_class[index],
            np_score=np_score[index],
        )
        for metric, metric_values in values.items():
            distributions[metric][start:stop] = metric_values.astype(
                np.float32
            )
    return observed, distributions


def bootstrap_difference_summary(
    values: np.ndarray,
    observed_difference: float,
    confidence: float,
) -> dict[str, float | int | bool]:
    """Summarize a paired bootstrap distribution of model A minus model B."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "difference_a_minus_b": float(observed_difference),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_ci_excludes_zero": False,
            "n_bootstrap_valid": 0,
        }

    alpha = 1.0 - confidence
    ci_lower, ci_upper = np.quantile(
        values, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return {
        "difference_a_minus_b": float(observed_difference),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap_ci_excludes_zero": bool(
            (ci_lower > 0.0) or (ci_upper < 0.0)
        ),
        "n_bootstrap_valid": int(len(values)),
    }


def regression_metrics_rows_with_groups(
    *,
    y_true: np.ndarray,
    y_class: np.ndarray,
    prediction_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the five prespecified regression metrics by permutation row."""
    y_true = np.asarray(y_true, dtype=float)
    y_class = np.asarray(y_class, dtype=int)
    prediction_rows = np.asarray(prediction_rows, dtype=float)
    y_rows = np.broadcast_to(y_true, prediction_rows.shape)

    metrics = regression_metrics_rows(y_rows, prediction_rows)
    np_mask = y_class == 1
    metrics["np_mae"] = np.mean(
        np.abs(prediction_rows[:, np_mask] - y_rows[:, np_mask]), axis=1
    )
    return metrics


def _initialize_permutation_counts(
    metrics: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {metric: 0 for metric in metrics},
        {metric: 0 for metric in metrics},
    )


def _update_two_sided_permutation_counts(
    *,
    permuted_a: dict[str, np.ndarray],
    permuted_b: dict[str, np.ndarray],
    observed_difference: dict[str, float],
    metrics: list[str],
    extreme_counts: dict[str, int],
    valid_counts: dict[str, int],
) -> None:
    """Accumulate |permuted difference| >= |observed difference| counts."""
    for metric in metrics:
        difference = np.asarray(
            permuted_a[metric] - permuted_b[metric], dtype=float
        )
        valid = np.isfinite(difference)
        valid_counts[metric] += int(np.sum(valid))
        if np.any(valid):
            threshold = abs(float(observed_difference[metric]))
            extreme_counts[metric] += int(
                np.sum(np.abs(difference[valid]) >= threshold - 1e-15)
            )


def _finalize_two_sided_permutation_pvalues(
    *,
    metrics: list[str],
    extreme_counts: dict[str, int],
    valid_counts: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    results: dict[str, dict[str, float | int]] = {}
    for metric in metrics:
        n_valid = valid_counts[metric]
        p_value = (
            np.nan
            if n_valid == 0
            else (extreme_counts[metric] + 1.0) / (n_valid + 1.0)
        )
        results[metric] = {
            "permutation_p_two_sided": float(p_value),
            "n_valid_permutations": int(n_valid),
            "n_extreme_permutations": int(extreme_counts[metric]),
        }
    return results


def paired_regression_permutation_pvalues(
    *,
    y_true: np.ndarray,
    y_class: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    observed_difference: dict[str, float],
    metrics: list[str],
    n_permutations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    """Paired two-sided prediction-swap permutation test for regression."""
    prediction_a = np.asarray(prediction_a, dtype=float)
    prediction_b = np.asarray(prediction_b, dtype=float)
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Regression prediction arrays have different shapes.")

    rng = np.random.default_rng(seed)
    extreme_counts, valid_counts = _initialize_permutation_counts(metrics)
    n_samples = len(prediction_a)

    for start in range(0, n_permutations, chunk_size):
        current = min(chunk_size, n_permutations - start)
        swap = rng.integers(
            0, 2, size=(current, n_samples), dtype=np.int8
        ).astype(bool)
        permuted_prediction_a = np.where(
            swap, prediction_b[None, :], prediction_a[None, :]
        )
        permuted_prediction_b = np.where(
            swap, prediction_a[None, :], prediction_b[None, :]
        )
        permuted_a = regression_metrics_rows_with_groups(
            y_true=y_true,
            y_class=y_class,
            prediction_rows=permuted_prediction_a,
        )
        permuted_b = regression_metrics_rows_with_groups(
            y_true=y_true,
            y_class=y_class,
            prediction_rows=permuted_prediction_b,
        )
        _update_two_sided_permutation_counts(
            permuted_a=permuted_a,
            permuted_b=permuted_b,
            observed_difference=observed_difference,
            metrics=metrics,
            extreme_counts=extreme_counts,
            valid_counts=valid_counts,
        )

    return _finalize_two_sided_permutation_pvalues(
        metrics=metrics,
        extreme_counts=extreme_counts,
        valid_counts=valid_counts,
    )


def paired_classification_permutation_pvalues(
    *,
    y_class: np.ndarray,
    class_prediction_a: np.ndarray,
    class_prediction_b: np.ndarray,
    np_score_a: np.ndarray,
    np_score_b: np.ndarray,
    observed_difference: dict[str, float],
    metrics: list[str],
    n_permutations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, dict[str, float | int]]:
    """Paired two-sided prediction-swap test for selected classification metrics."""
    y_class = np.asarray(y_class, dtype=int)
    class_prediction_a = np.asarray(class_prediction_a, dtype=int)
    class_prediction_b = np.asarray(class_prediction_b, dtype=int)
    np_score_a = np.asarray(np_score_a, dtype=float)
    np_score_b = np.asarray(np_score_b, dtype=float)

    expected_shape = y_class.shape
    for name, array in [
        ("class_prediction_a", class_prediction_a),
        ("class_prediction_b", class_prediction_b),
        ("np_score_a", np_score_a),
        ("np_score_b", np_score_b),
    ]:
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {array.shape}; expected {expected_shape}."
            )

    rng = np.random.default_rng(seed)
    extreme_counts, valid_counts = _initialize_permutation_counts(metrics)
    n_samples = len(y_class)

    for start in range(0, n_permutations, chunk_size):
        current = min(chunk_size, n_permutations - start)
        swap = rng.integers(
            0, 2, size=(current, n_samples), dtype=np.int8
        ).astype(bool)

        permuted_class_a = np.where(
            swap, class_prediction_b[None, :], class_prediction_a[None, :]
        )
        permuted_class_b = np.where(
            swap, class_prediction_a[None, :], class_prediction_b[None, :]
        )
        permuted_score_a = np.where(
            swap, np_score_b[None, :], np_score_a[None, :]
        )
        permuted_score_b = np.where(
            swap, np_score_a[None, :], np_score_b[None, :]
        )

        y_rows = np.broadcast_to(y_class, permuted_class_a.shape)
        permuted_a = classification_metrics_rows(
            y_class=y_rows,
            predicted_class=permuted_class_a,
            np_score=permuted_score_a,
        )
        permuted_b = classification_metrics_rows(
            y_class=y_rows,
            predicted_class=permuted_class_b,
            np_score=permuted_score_b,
        )
        _update_two_sided_permutation_counts(
            permuted_a=permuted_a,
            permuted_b=permuted_b,
            observed_difference=observed_difference,
            metrics=metrics,
            extreme_counts=extreme_counts,
            valid_counts=valid_counts,
        )

    return _finalize_two_sided_permutation_pvalues(
        metrics=metrics,
        extreme_counts=extreme_counts,
        valid_counts=valid_counts,
    )


def parse_boolean_series(
    series: pd.Series,
    column_name: str,
) -> np.ndarray:
    """Parse a CSV boolean column without treating nonempty strings as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise").to_numpy(float)
        if not np.isin(numeric, [0.0, 1.0]).all():
            raise ValueError(
                f"{column_name!r} contains numeric values other than 0/1."
            )
        return numeric.astype(bool)

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
            f"{column_name!r} contains unrecognized boolean values: "
            f"{unknown[:20]}"
        )
    return normalized.map(mapping).to_numpy(bool)


def resolve_predictions_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_raw_prediction_table(
    prediction_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    """
    Load and combine long-format raw OOF prediction tables.

    The primary file may contain the four faster models, while a second file
    may contain the independently trained LogBB_Pred comparator. For backward
    compatibility, a single file containing all five models is also accepted.

    Expected columns:
        sample_id, row_index, smiles, outer_fold, true_logBB, bbb_class,
        has_logBB, classification_only, model, model_task,
        predicted_logBB, score_nonpenetrant, predicted_bbb_class,
        classification_threshold, absolute_error, squared_error, wrong_side.
    """
    if not prediction_paths:
        raise ValueError("At least one raw prediction file is required.")

    source_frames: list[pd.DataFrame] = []
    for prediction_path in prediction_paths:
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"Raw prediction file not found: {prediction_path}"
            )
        source = pd.read_csv(prediction_path)
        source = source.copy()
        source["__source_path"] = str(prediction_path)
        source_frames.append(source)

    frame = pd.concat(source_frames, ignore_index=True, sort=False)
    required_columns = [
        "sample_id",
        "row_index",
        "smiles",
        "outer_fold",
        "true_logBB",
        "bbb_class",
        "has_logBB",
        "classification_only",
        "model",
        "model_task",
        "predicted_logBB",
        "score_nonpenetrant",
        "predicted_bbb_class",
        "classification_threshold",
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "Raw prediction table is missing required columns: "
            f"{missing_columns}. Observed columns: {frame.columns.tolist()}"
        )

    frame = frame.copy()
    frame["sample_id"] = pd.to_numeric(
        frame["sample_id"], errors="raise"
    ).astype(int)
    frame["row_index"] = pd.to_numeric(
        frame["row_index"], errors="raise"
    ).astype(int)
    frame["outer_fold"] = pd.to_numeric(
        frame["outer_fold"], errors="raise"
    ).astype(int)
    frame["bbb_class"] = pd.to_numeric(
        frame["bbb_class"], errors="raise"
    ).astype(int)
    frame["predicted_bbb_class"] = pd.to_numeric(
        frame["predicted_bbb_class"], errors="raise"
    ).astype(int)
    frame["classification_threshold"] = pd.to_numeric(
        frame["classification_threshold"], errors="raise"
    ).astype(float)
    frame["true_logBB"] = pd.to_numeric(
        frame["true_logBB"], errors="coerce"
    )
    frame["predicted_logBB"] = pd.to_numeric(
        frame["predicted_logBB"], errors="coerce"
    )
    frame["score_nonpenetrant"] = pd.to_numeric(
        frame["score_nonpenetrant"], errors="coerce"
    )
    frame["has_logBB"] = parse_boolean_series(
        frame["has_logBB"], "has_logBB"
    )
    frame["classification_only"] = parse_boolean_series(
        frame["classification_only"], "classification_only"
    )
    if "wrong_side" in frame.columns:
        frame["wrong_side"] = parse_boolean_series(
            frame["wrong_side"], "wrong_side"
        )

    if set(frame["bbb_class"].unique()).difference({0, 1}):
        raise ValueError("bbb_class must contain only 0 and 1.")
    if set(frame["predicted_bbb_class"].unique()).difference({0, 1}):
        raise ValueError(
            "predicted_bbb_class must contain only 0 and 1."
        )

    original_model_name = frame["model"].astype(str)
    frame["__source_model_name"] = original_model_name
    frame["model"] = original_model_name.replace(LOGBB_MODEL_ALIASES)

    observed_models = sorted(
        frame["model"].dropna().astype(str).unique().tolist()
    )
    missing_models = sorted(set(MODEL_ORDER).difference(observed_models))
    extra_models = sorted(set(observed_models).difference(MODEL_ORDER))
    if missing_models:
        raise ValueError(
            f"Raw prediction table is missing models: {missing_models}. "
            f"Observed models: {observed_models}"
        )
    if extra_models:
        warnings.warn(
            "Ignoring models not included in this fixed analysis: "
            f"{extra_models}",
            RuntimeWarning,
        )

    working = frame[frame["model"].isin(MODEL_ORDER)].copy()

    # When a separate LogBB_Pred file is supplied after a combined prediction
    # file, the same model can appear twice. Treat the last supplied source as
    # the explicit override for LogBB_Pred, while retaining strict duplicate
    # checks for every other model.
    duplicate_mask = working.duplicated(
        ["model", "row_index"], keep=False
    )
    if duplicate_mask.any():
        duplicate_models = set(
            working.loc[duplicate_mask, "model"].astype(str).unique()
        )
        logbb_name = "logbb_pred_shaker_retrained"

        if (
            duplicate_models == {logbb_name}
            and len(prediction_paths) > 1
        ):
            preferred_source = str(prediction_paths[-1])
            preferred_rows = working[
                (working["model"] == logbb_name)
                & (working["__source_path"] == preferred_source)
            ].copy()

            expected_rows = working.loc[
                working["model"] != logbb_name, "row_index"
            ].nunique()
            if expected_rows == 0:
                expected_rows = preferred_rows["row_index"].nunique()

            if (
                preferred_rows.empty
                or preferred_rows["row_index"].duplicated().any()
                or preferred_rows["row_index"].nunique() != expected_rows
            ):
                raise ValueError(
                    "The last supplied prediction file was intended to "
                    "override LogBB_Pred, but it does not contain one complete "
                    "unique OOF prediction per row. Preferred source: "
                    f"{preferred_source}"
                )

            removed_sources = sorted(
                working.loc[
                    (working["model"] == logbb_name)
                    & (working["__source_path"] != preferred_source),
                    "__source_path",
                ]
                .astype(str)
                .unique()
                .tolist()
            )
            warnings.warn(
                "LogBB_Pred was present in more than one input file. Using the "
                "last supplied --logbb-predictions-file as the explicit "
                f"override and ignoring LogBB_Pred rows from: {removed_sources}",
                RuntimeWarning,
            )
            working = pd.concat(
                [
                    working[working["model"] != logbb_name],
                    preferred_rows,
                ],
                ignore_index=True,
                sort=False,
            )
        else:
            examples = (
                working.loc[duplicate_mask, ["model", "row_index", "__source_path"]]
                .head(20)
                .to_dict("records")
            )
            raise ValueError(
                "Expected exactly one row per model × row_index. "
                f"Duplicate examples: {examples}"
            )

    # Confirm that the override logic, if used, resolved all duplicates.
    duplicate_mask = working.duplicated(
        ["model", "row_index"], keep=False
    )
    if duplicate_mask.any():
        examples = (
            working.loc[
                duplicate_mask,
                ["model", "row_index", "__source_path"],
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValueError(
            "Duplicate model × row_index rows remain after resolving the "
            f"explicit LogBB_Pred override: {examples}"
        )

    # Construct canonical rows from the first requested model.
    reference_name = MODEL_ORDER[0]
    reference = (
        working[working["model"] == reference_name]
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    if reference.empty:
        raise ValueError(f"No rows found for {reference_name!r}.")
    if len(reference) != reference["row_index"].nunique():
        raise ValueError("Reference model row_index is not unique.")
    expected_row_index = np.arange(len(reference), dtype=int)
    actual_row_index = reference["row_index"].to_numpy(int)
    if not np.array_equal(actual_row_index, expected_row_index):
        raise ValueError(
            "row_index must be a complete zero-based sequence after sorting. "
            f"First observed values: {actual_row_index[:20].tolist()}"
        )
    if not np.array_equal(
        reference["sample_id"].to_numpy(int),
        reference["row_index"].to_numpy(int),
    ):
        raise ValueError(
            "sample_id and row_index differ in the reference model. "
            "Use one stable canonical row identifier consistently."
        )

    measured = reference["has_logBB"].to_numpy(bool)
    class_only = reference["classification_only"].to_numpy(bool)
    if not np.array_equal(class_only, ~measured):
        mismatch = np.flatnonzero(class_only != ~measured)
        raise ValueError(
            "classification_only is not the inverse of has_logBB. "
            f"First mismatches: {mismatch[:20].tolist()}"
        )
    true_logbb = reference["true_logBB"].to_numpy(float)
    if not np.isfinite(true_logbb[measured]).all():
        raise ValueError(
            "Measured rows contain missing or nonfinite true_logBB values."
        )
    if np.isfinite(true_logbb[class_only]).any():
        warnings.warn(
            "Some classification-only rows contain true_logBB. They remain "
            "excluded from regression metrics because has_logBB is False.",
            RuntimeWarning,
        )

    canonical = pd.DataFrame(
        {
            "row_position": actual_row_index,
            "sample_id": reference["sample_id"].to_numpy(int),
            "smiles": reference["smiles"].astype(str).to_numpy(),
            "outer_fold": reference["outer_fold"].to_numpy(int),
            "logBB": true_logbb,
            "bbb_class": reference["bbb_class"].to_numpy(int),
            "measured": measured,
            "classification_only": class_only,
        }
    )

    selected_models: dict[str, dict[str, Any]] = {}
    audit_rows = []
    truth_columns = [
        "sample_id",
        "row_index",
        "smiles",
        "outer_fold",
        "true_logBB",
        "bbb_class",
        "has_logBB",
        "classification_only",
    ]

    for model_name in MODEL_ORDER:
        group = (
            working[working["model"] == model_name]
            .sort_values("row_index")
            .reset_index(drop=True)
        )
        if len(group) != len(canonical):
            raise ValueError(
                f"{model_name!r} has {len(group)} rows; expected "
                f"{len(canonical)} complete OOF rows."
            )

        # Verify exact pairing and truth/fold consistency.
        for column in truth_columns:
            observed = group[column]
            expected = reference[column]
            if column == "true_logBB":
                equal = np.allclose(
                    observed.to_numpy(float),
                    expected.to_numpy(float),
                    equal_nan=True,
                    atol=1e-12,
                    rtol=0,
                )
            else:
                equal = np.array_equal(
                    observed.astype(str).to_numpy(),
                    expected.astype(str).to_numpy(),
                )
            if not equal:
                raise ValueError(
                    f"{model_name!r} does not match the canonical rows in "
                    f"column {column!r}."
                )

        task_values = sorted(
            group["model_task"].dropna().astype(str).str.lower().unique()
        )
        expected_task = (
            "classification"
            if model_name == DUMMY_CLASSIFIER_MODEL
            else "regression"
        )
        if len(task_values) != 1 or expected_task not in task_values[0]:
            raise ValueError(
                f"{model_name!r} has model_task values {task_values}; "
                f"expected {expected_task!r}."
            )

        class_prediction = group[
            "predicted_bbb_class"
        ].to_numpy(int)
        np_score = group["score_nonpenetrant"].to_numpy(float)
        if not np.isfinite(np_score).all():
            raise ValueError(
                f"{model_name!r} has missing/nonfinite "
                "score_nonpenetrant values."
            )

        threshold_values = group[
            "classification_threshold"
        ].to_numpy(float)
        finite_thresholds = threshold_values[np.isfinite(threshold_values)]
        if not len(finite_thresholds):
            raise ValueError(
                f"{model_name!r} has no finite classification threshold."
            )
        if not np.allclose(
            finite_thresholds, finite_thresholds[0], atol=1e-12, rtol=0
        ):
            raise ValueError(
                f"{model_name!r} uses multiple classification thresholds."
            )
        threshold = float(finite_thresholds[0])

        if model_name in REGRESSION_MODELS:
            # Regression models classify by thresholding predicted logBB at -1.
            if not np.isclose(
                threshold, MARGIN, atol=1e-12, rtol=0
            ):
                raise ValueError(
                    f"{model_name!r} regression threshold is {threshold}; "
                    f"expected the logBB threshold {MARGIN}."
                )

            regression_prediction = group[
                "predicted_logBB"
            ].to_numpy(float)
            if not np.isfinite(regression_prediction).all():
                raise ValueError(
                    f"{model_name!r} lacks complete predicted_logBB values."
                )
            implied_class = (
                regression_prediction < threshold
            ).astype(int)
            if not np.array_equal(implied_class, class_prediction):
                mismatch = np.flatnonzero(
                    implied_class != class_prediction
                )
                raise ValueError(
                    f"{model_name!r} predicted_bbb_class disagrees with "
                    "predicted_logBB < classification_threshold. First "
                    f"mismatches: {mismatch[:20].tolist()}"
                )

            # score_nonpenetrant is expected to increase as logBB decreases.
            if not np.allclose(
                np_score,
                -regression_prediction,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"{model_name!r} score_nonpenetrant is not "
                    "-predicted_logBB."
                )
            distance_prediction = regression_prediction

        else:
            # The dummy classifier uses a probability/score cutoff, normally
            # 0.5. This threshold is not a logBB threshold and must not be
            # compared with MARGIN = -1.
            if not np.isfinite(threshold):
                raise ValueError(
                    f"{model_name!r} has a nonfinite classification threshold."
                )
            implied_class = (np_score >= threshold).astype(int)
            if not np.array_equal(implied_class, class_prediction):
                mismatch = np.flatnonzero(
                    implied_class != class_prediction
                )
                raise ValueError(
                    f"{model_name!r} predicted_bbb_class disagrees with "
                    "score_nonpenetrant >= classification_threshold. First "
                    f"mismatches: {mismatch[:20].tolist()}"
                )

            regression_prediction = None
            distance_prediction = None

        expected_wrong = (
            class_prediction != canonical["bbb_class"].to_numpy(int)
        )
        if "wrong_side" in group.columns:
            stored_wrong = group["wrong_side"].to_numpy(bool)
            if not np.array_equal(stored_wrong, expected_wrong):
                mismatch = np.flatnonzero(
                    stored_wrong != expected_wrong
                )
                raise ValueError(
                    f"{model_name!r} stored wrong_side is inconsistent with "
                    "predicted_bbb_class and bbb_class. First mismatches: "
                    f"{mismatch[:20].tolist()}"
                )

        if (
            model_name in REGRESSION_MODELS
            and "absolute_error" in group.columns
        ):
            stored_absolute = pd.to_numeric(
                group["absolute_error"], errors="coerce"
            ).to_numpy(float)
            expected_absolute = np.abs(
                regression_prediction[measured] - true_logbb[measured]
            )
            if not np.allclose(
                stored_absolute[measured],
                expected_absolute,
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{model_name!r} stored absolute_error is inconsistent "
                    "with true_logBB and predicted_logBB."
                )

        source_paths = sorted(group["__source_path"].astype(str).unique())
        source_model_names = sorted(
            group["__source_model_name"].astype(str).unique()
        )
        if len(source_paths) != 1:
            raise ValueError(
                f"{model_name!r} appears in multiple prediction files: "
                f"{source_paths}. Supply exactly one source for each model."
            )

        selected_models[model_name] = {
            "model": model_name,
            "display_name": MODEL_DISPLAY[model_name],
            "regression_prediction": regression_prediction,
            "class_prediction": class_prediction,
            "np_score": np_score,
            "distance_prediction": distance_prediction,
            "source_path": source_paths[0],
            "source_model_name": ",".join(source_model_names),
            "score_orientation": (
                "-predicted_logBB"
                if model_name in REGRESSION_MODELS
                else "score_nonpenetrant"
            ),
        }
        audit_rows.append(
            {
                "model": model_name,
                "display_name": MODEL_DISPLAY[model_name],
                "source_path": source_paths[0],
                "source_model_name": ",".join(source_model_names),
                "model_task": expected_task,
                "n_rows": len(group),
                "n_measured": int(measured.sum()),
                "n_classification_only": int(class_only.sum()),
                "regression_available": (
                    regression_prediction is not None
                ),
                "threshold_available": True,
                "wrong_side_distance_available": (
                    distance_prediction is not None
                ),
                "classification_threshold": threshold,
                "status": "loaded_and_validated",
            }
        )

    return canonical, selected_models, pd.DataFrame(audit_rows)


def metric_catalog() -> pd.DataFrame:
    rows = []
    for metric in REGRESSION_METRICS:
        rows.append(
            {
                "subset": "measured_regression",
                "metric": metric,
                "metric_group": (
                    "measured_nonpenetrant" if metric == "np_mae" else "overall"
                ),
                "direction": metric_direction(metric),
                "positive_improvement_definition": (
                    "comparator - focus"
                    if metric_direction(metric) == "lower"
                    else "focus - comparator"
                ),
                "dummy_classifier_available": False,
            }
        )

    for subset in [
        "measured_threshold",
        "all_threshold",
        "classification_only_threshold",
    ]:
        for metric in CLASSIFICATION_METRICS:
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "metric_group": "threshold_derived",
                    "direction": metric_direction(metric),
                    "positive_improvement_definition": (
                        "comparator - focus"
                        if metric_direction(metric) == "lower"
                        else "focus - comparator"
                    ),
                    "dummy_classifier_available": True,
                }
            )
    return pd.DataFrame(rows)


def add_pair_dataset_holm_adjustment(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Holm across metrics within each unique model pair and subset."""
    result = frame.copy()
    result["permutation_p_two_sided_holm_pair_dataset"] = (
        result.groupby(
            ["model_a", "model_b", "subset"],
            group_keys=False,
        )["permutation_p_two_sided"]
        .transform(lambda values: holm_adjust(values).to_numpy())
    )
    result["holm_family_size"] = result.groupby(
        ["model_a", "model_b", "subset"]
    )["metric"].transform("size")
    result["significant_permutation_holm_0.05"] = (
        result["permutation_p_two_sided_holm_pair_dataset"] < 0.05
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--predictions-file",
        required=True,
        help=(
            "Long-format raw OOF prediction CSV containing the four faster "
            "models, or all five models for backward compatibility. Relative "
            "paths are resolved under --root."
        ),
    )
    parser.add_argument(
        "--logbb-predictions-file",
        default=None,
        help=(
            "Optional separate OOF prediction CSV produced by "
            "train_logbb_pred_matched.py. Its model name is normalized to "
            "logbb_pred_shaker_retrained after validation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./all_models/fixed_five_model_paired_permutation",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of paired prediction-swap permutations.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of paired stratified bootstrap replicates for CIs only.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()

    if args.permutations < 1:
        parser.error("--permutations must be at least 1.")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1.")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be strictly between 0 and 1.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1.")
    return args


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_paths = [
        resolve_predictions_path(root, args.predictions_file)
    ]
    if args.logbb_predictions_file is not None:
        prediction_paths.append(
            resolve_predictions_path(root, args.logbb_predictions_file)
        )

    canonical, selected_models, selected_audit = load_raw_prediction_table(
        prediction_paths
    )
    selected_audit.to_csv(
        output_dir / "selected_model_input_audit.csv", index=False
    )
    metric_catalog().to_csv(output_dir / "metric_catalog.csv", index=False)

    y_reg_all = canonical["logBB"].to_numpy(float)
    y_class_all = canonical["bbb_class"].to_numpy(int)
    folds_all = canonical["outer_fold"].to_numpy(int)
    measured = canonical["measured"].to_numpy(bool)
    class_only = ~measured

    subset_masks = {
        "measured_regression": measured,
        "measured_threshold": measured,
        "all_threshold": np.ones(len(canonical), dtype=bool),
        "classification_only_threshold": class_only,
    }

    model_values: dict[str, dict[str, dict[str, Any]]] = {
        model: {} for model in MODEL_ORDER
    }

    # Bootstrap distributions are calculated once per model using shared,
    # paired indices. They are used only for confidence intervals.
    bootstrap_indices = {
        "measured": make_stratified_bootstrap_matrix(
            folds_all[measured],
            y_class_all[measured],
            args.bootstrap,
            stable_seed(args.seed, "bootstrap", "measured"),
        ),
        "all": make_stratified_bootstrap_matrix(
            folds_all,
            y_class_all,
            args.bootstrap,
            stable_seed(args.seed, "bootstrap", "all"),
        ),
        "classification_only": make_stratified_bootstrap_matrix(
            folds_all[class_only],
            y_class_all[class_only],
            args.bootstrap,
            stable_seed(args.seed, "bootstrap", "classification_only"),
        ),
    }

    for model_name, record in selected_models.items():
        regression_prediction = record["regression_prediction"]
        if regression_prediction is not None:
            observed, bootstrap = bootstrap_regression_metrics(
                y_true=y_reg_all[measured],
                y_class=y_class_all[measured],
                prediction=regression_prediction[measured],
                bootstrap_index=bootstrap_indices["measured"],
                chunk_size=args.chunk_size,
            )
            model_values[model_name]["measured_regression"] = {
                "n_samples": int(measured.sum()),
                "observed": observed,
                "bootstrap": bootstrap,
            }

        for subset, index_name in [
            ("measured_threshold", "measured"),
            ("all_threshold", "all"),
            ("classification_only_threshold", "classification_only"),
        ]:
            mask = subset_masks[subset]
            observed, bootstrap = bootstrap_classification_metrics(
                y_class=y_class_all[mask],
                predicted_class=record["class_prediction"][mask],
                np_score=record["np_score"][mask],
                bootstrap_index=bootstrap_indices[index_name],
                chunk_size=args.chunk_size,
            )
            model_values[model_name][subset] = {
                "n_samples": int(mask.sum()),
                "observed": observed,
                "bootstrap": bootstrap,
            }

    del bootstrap_indices

    pooled_rows = []
    for model_name in MODEL_ORDER:
        for subset, subset_values in model_values[model_name].items():
            for metric, value in subset_values["observed"].items():
                pooled_rows.append(
                    {
                        "model": model_name,
                        "display_name": MODEL_DISPLAY[model_name],
                        "subset": subset,
                        "subset_label": SUBSET_LABELS[subset],
                        "metric": metric,
                        "value": value,
                        "direction": metric_direction(metric),
                    }
                )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(
        output_dir / "pooled_metrics_selected_models_long.csv", index=False
    )
    pooled.pivot_table(
        index=["model", "display_name"],
        columns=["subset", "metric"],
        values="value",
        aggfunc="first",
    ).to_csv(output_dir / "pooled_metrics_selected_models_wide.csv")

    pairwise_rows: list[dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        common_subsets = sorted(
            set(model_values[model_a]).intersection(model_values[model_b])
        )
        for subset in common_subsets:
            values_a = model_values[model_a][subset]
            values_b = model_values[model_b][subset]
            metrics = sorted(
                set(values_a["observed"]).intersection(values_b["observed"])
            )
            observed_difference = {
                metric: float(
                    values_a["observed"][metric]
                    - values_b["observed"][metric]
                )
                for metric in metrics
            }

            mask = subset_masks[subset]
            if subset == "measured_regression":
                permutation_results = paired_regression_permutation_pvalues(
                    y_true=y_reg_all[mask],
                    y_class=y_class_all[mask],
                    prediction_a=selected_models[model_a][
                        "regression_prediction"
                    ][mask],
                    prediction_b=selected_models[model_b][
                        "regression_prediction"
                    ][mask],
                    observed_difference=observed_difference,
                    metrics=metrics,
                    n_permutations=args.permutations,
                    seed=stable_seed(
                        args.seed, "permutation", model_a, model_b, subset
                    ),
                    chunk_size=args.chunk_size,
                )
            else:
                permutation_results = (
                    paired_classification_permutation_pvalues(
                        y_class=y_class_all[mask],
                        class_prediction_a=selected_models[model_a][
                            "class_prediction"
                        ][mask],
                        class_prediction_b=selected_models[model_b][
                            "class_prediction"
                        ][mask],
                        np_score_a=selected_models[model_a]["np_score"][mask],
                        np_score_b=selected_models[model_b]["np_score"][mask],
                        observed_difference=observed_difference,
                        metrics=metrics,
                        n_permutations=args.permutations,
                        seed=stable_seed(
                            args.seed,
                            "permutation",
                            model_a,
                            model_b,
                            subset,
                        ),
                        chunk_size=args.chunk_size,
                    )
                )

            for metric in metrics:
                bootstrap_difference = (
                    np.asarray(values_a["bootstrap"][metric], dtype=float)
                    - np.asarray(values_b["bootstrap"][metric], dtype=float)
                )
                ci_summary = bootstrap_difference_summary(
                    bootstrap_difference,
                    observed_difference[metric],
                    args.confidence,
                )
                direction = metric_direction(metric)
                effect_in_favor_of_a = float(
                    metric_improvement(
                        values_a["observed"][metric],
                        values_b["observed"][metric],
                        metric,
                    )
                )
                pairwise_rows.append(
                    {
                        "model_a": model_a,
                        "model_a_display_name": MODEL_DISPLAY[model_a],
                        "model_b": model_b,
                        "model_b_display_name": MODEL_DISPLAY[model_b],
                        "subset": subset,
                        "subset_label": SUBSET_LABELS[subset],
                        "metric": metric,
                        "n_paired": values_a["n_samples"],
                        "model_a_value": values_a["observed"][metric],
                        "model_b_value": values_b["observed"][metric],
                        **ci_summary,
                        "better_direction": direction,
                        "effect_in_favor_of_a": effect_in_favor_of_a,
                        "observed_better_model": (
                            model_a
                            if effect_in_favor_of_a > 0
                            else model_b
                            if effect_in_favor_of_a < 0
                            else "tie"
                        ),
                        **permutation_results[metric],
                        "n_permutations_requested": args.permutations,
                    }
                )

    results = add_pair_dataset_holm_adjustment(pd.DataFrame(pairwise_rows))
    preferred_columns = [
        "model_a",
        "model_a_display_name",
        "model_b",
        "model_b_display_name",
        "subset",
        "subset_label",
        "metric",
        "n_paired",
        "model_a_value",
        "model_b_value",
        "difference_a_minus_b",
        "ci_lower",
        "ci_upper",
        "bootstrap_ci_excludes_zero",
        "n_bootstrap_valid",
        "better_direction",
        "effect_in_favor_of_a",
        "observed_better_model",
        "permutation_p_two_sided",
        "permutation_p_two_sided_holm_pair_dataset",
        "significant_permutation_holm_0.05",
        "holm_family_size",
        "n_extreme_permutations",
        "n_valid_permutations",
        "n_permutations_requested",
    ]
    results = results[preferred_columns].sort_values(
        ["model_a", "model_b", "subset", "metric"]
    )
    results.to_csv(
        output_dir
        / "paired_two_sided_permutation_holm_by_pair_and_dataset.csv",
        index=False,
    )

    for subset in SUBSET_LABELS:
        results[results["subset"] == subset].to_csv(
            output_dir / f"paired_permutation_{subset}.csv", index=False
        )

    metadata = {
        "prediction_source_files": [str(path) for path in prediction_paths],
        "models": MODEL_ORDER,
        "permutation_test": (
            "paired two-sided within-compound prediction-label swap"
        ),
        "permutation_replicates": args.permutations,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_purpose": "percentile confidence intervals only",
        "bootstrap_stratification": "original outer fold × BBB class",
        "holm_family": (
            "the 5 selected regression metrics or 8 selected classification "
            "metrics within each unique model pair and evaluation dataset"
        ),
        "confidence": args.confidence,
        "seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    readme = f"""# Paired permutation significance results

The only hypothesis test in this analysis is a paired, two-sided permutation
 test based on randomly exchanging the two models' predictions within each
 compound. Holm correction is applied separately across the five selected
 regression metrics or 9 selected classification metrics within each unique
 model pair and evaluation dataset.

Paired bootstrap resampling stratified by original outer fold × BBB class is
 used only for the {args.confidence:.1%} percentile confidence intervals.

Primary output:
- `paired_two_sided_permutation_holm_by_pair_and_dataset.csv`

Primary adjusted p-value column:
- `permutation_p_two_sided_holm_pair_dataset`
"""
    (output_dir / "README_RESULTS.md").write_text(readme)

    print("\nPaired two-sided permutation results:")
    print(
        results[
            [
                "model_a",
                "model_b",
                "subset",
                "metric",
                "permutation_p_two_sided",
                "permutation_p_two_sided_holm_pair_dataset",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
